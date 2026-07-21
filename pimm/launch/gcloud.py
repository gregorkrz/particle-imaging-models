"""Google Cloud Batch executor for ``pimm submit --site gcloud``.

Cloud Batch is GCP's managed, queue-based batch service -- the closest analog to
a Slurm queue. A submitted job queues, provisions an A100 VM, runs the pimm dev
container image, and tears the VM down. Output artifacts are written to a GCS
bucket that Batch mounts on the VM via gcsfuse, so the training/checkpoint code
sees an ordinary local directory and needs no cloud-storage awareness.

The user writes ``paths.exp_root: gs://<bucket>/<prefix>`` in the site config;
this module derives the bucket, mounts it at ``gcs_mount_path`` on the VM, and
rewrites ``EXP_ROOT`` to the mounted local path before rendering the training
script. Everything else reuses the shared renderers in ``local.py``.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import tempfile
from typing import Any

from .local import build_train_sh_command, redact_script, render_script
from .utils import scheduler, slurm_time_to_minutes, write_text

DEFAULT_MOUNT_PATH = "/mnt/gcs"
DEFAULT_PROVISIONING_MODEL = "STANDARD"
DEFAULT_BOOT_DISK_GB = 200


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI into ``(bucket, prefix)``.

    ``prefix`` has no leading/trailing slashes and may be empty (bucket root).
    """
    if not uri.startswith("gs://"):
        raise SystemExit(f"Expected a gs:// URI, got {uri!r}")
    rest = uri[len("gs://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise SystemExit(f"gs:// URI is missing a bucket: {uri!r}")
    return bucket, prefix.strip("/")


def sanitize_job_name(run_name: str) -> str:
    """Coerce a run name into a valid Cloud Batch job id.

    Batch job names must match ``^[a-z]([a-z0-9-]{0,61})?$`` (<=63 chars,
    lowercase, start with a letter). Run names contain uppercase, underscores
    (timestamp), and dots, so normalize them.
    """
    name = re.sub(r"[^a-z0-9-]", "-", run_name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    if not name or not name[0].isalpha():
        name = f"pimm-{name}".strip("-")
    return name[:63].rstrip("-")


def build_batch_job(cfg: dict[str, Any], run_name: str) -> tuple[dict[str, Any], str]:
    """Return ``(job_spec, inner_script)`` for a single-node A100 Batch job."""
    opts = cfg.get("resources", {}).get("scheduler_options", {}) or {}
    mount_path = str(opts.get("gcs_mount_path") or DEFAULT_MOUNT_PATH).rstrip("/")

    exp_root = str(cfg.get("paths", {}).get("exp_root", ""))
    bucket, prefix = parse_gs_uri(exp_root)
    # EXP_ROOT on the VM points into the gcsfuse mount, not the gs:// URI.
    exp_root_local = "/".join([mount_path, prefix]).rstrip("/") if prefix else mount_path

    # Cloud Batch is the container runner: render the *inner* training script
    # with no nested `docker run` (runtime="none") and the mounted EXP_ROOT.
    render_cfg = copy.deepcopy(cfg)
    render_cfg.setdefault("container", {})["runtime"] = "none"
    render_cfg.setdefault("paths", {})["exp_root"] = exp_root_local
    train_cmd = build_train_sh_command(render_cfg, run_name)
    script = render_script(render_cfg, train_cmd, run_name)

    image = cfg.get("container", {}).get("image")
    provisioning = str(opts.get("provisioning_model") or DEFAULT_PROVISIONING_MODEL)
    boot_disk_gb = int(opts.get("boot_disk_gb") or DEFAULT_BOOT_DISK_GB)
    minutes = slurm_time_to_minutes(cfg.get("resources", {}).get("time", "24:00:00"))
    max_run_seconds = f"{minutes * 60}s"

    instance_policy: dict[str, Any] = {
        "machineType": opts["machine_type"],
        "provisioningModel": provisioning,
        "bootDisk": {"sizeGb": boot_disk_gb},
    }
    # a2-* machine types bundle their A100s automatically. For other machine
    # types (e.g. n1-*), require an explicit accelerator spec.
    accel_type = opts.get("accelerator_type")
    accel_count = opts.get("accelerator_count")
    if accel_type and accel_count:
        instance_policy["accelerators"] = [
            {"type": accel_type, "count": int(accel_count)}
        ]

    task_spec: dict[str, Any] = {
        "runnables": [
            {
                "container": {
                    "imageUri": image,
                    "entrypoint": "/bin/bash",
                    "commands": ["-lc", script],
                    # gcsfuse mount lives on the host; expose it to the container.
                    "volumes": [f"{mount_path}:{mount_path}"],
                }
            }
        ],
        "volumes": [
            {
                "gcs": {"remotePath": bucket},
                "mountPath": mount_path,
            }
        ],
        "maxRunDuration": max_run_seconds,
        "maxRetryCount": 0,
    }

    allocation_instance: dict[str, Any] = {
        "policy": instance_policy,
        "installGpuDrivers": True,
    }
    network = opts.get("network")
    subnetwork = opts.get("subnetwork")
    allocation_policy: dict[str, Any] = {"instances": [allocation_instance]}
    if opts.get("service_account"):
        allocation_policy["serviceAccount"] = {"email": opts["service_account"]}
    if network or subnetwork:
        interface: dict[str, Any] = {}
        if network:
            interface["network"] = network
        if subnetwork:
            interface["subnetwork"] = subnetwork
        allocation_policy["network"] = {"networkInterfaces": [interface]}

    job_spec: dict[str, Any] = {
        "taskGroups": [{"taskCount": 1, "taskSpec": task_spec}],
        "allocationPolicy": allocation_policy,
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    return job_spec, script


def run_gcloud(
    cfg: dict[str, Any],
    run_name: str,
    *,
    dry_run: bool,
    output: str | None,
) -> int:
    """Render and submit a Google Cloud Batch job for a training run."""
    if scheduler(cfg) != "gcloud":
        raise SystemExit("run_gcloud requires resources.scheduler='gcloud'")

    opts = cfg.get("resources", {}).get("scheduler_options", {}) or {}
    project = opts["project"]
    location = opts["location"]

    job_spec, script = build_batch_job(cfg, run_name)
    job_name = sanitize_job_name(run_name)
    job_json = json.dumps(job_spec, indent=2)

    # The inner training script is embedded as a JSON string, so its `export`
    # lines can't be redacted after serialization. Redact the script itself and
    # splice it into a display-only copy of the spec.
    display_spec = copy.deepcopy(job_spec)
    display_spec["taskGroups"][0]["taskSpec"]["runnables"][0]["container"][
        "commands"
    ][1] = redact_script(script)
    display_json = json.dumps(display_spec, indent=2)
    if output:
        path = write_text(output, display_json)
        print(f"# wrote Cloud Batch job spec: {path}")
    if dry_run:
        print(f"# gcloud batch jobs submit {job_name} "
              f"--project {project} --location {location} --config <spec>")
        print(display_json)
        return 0

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=f"{job_name}-", delete=False
    ) as handle:
        handle.write(job_json)
        config_path = handle.name

    argv = [
        "gcloud",
        "batch",
        "jobs",
        "submit",
        job_name,
        "--project",
        project,
        "--location",
        location,
        "--config",
        config_path,
    ]
    print(f"# submitting Cloud Batch job {job_name} to {project}/{location}")
    return subprocess.run(argv).returncode
