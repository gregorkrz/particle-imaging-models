import json

import pytest

from pimm.launch.config import finalize_config, load_config, validate_launch_config
from pimm.launch.gcloud import build_batch_job, parse_gs_uri, sanitize_job_name


TIMESTAMP = "2026-01-02_03-04-05"


def _config(**overrides):
    cfg = load_config(site="gcloud", recipe=None, launch_timestamp=TIMESTAMP)
    cfg["executor"] = "batch"
    cfg["resources"]["scheduler_options"].update(
        project="my-proj", location="us-central1", machine_type="a2-highgpu-1g"
    )
    cfg["paths"]["exp_root"] = "gs://my-bucket/pimm_exp"
    cfg["run"] = {"name": "gcloud-render", "timestamp": False}
    cfg["train"]["config"] = "tests/tiny_semseg"
    for key, value in overrides.items():
        cfg[key] = value
    return finalize_config(cfg, launch_timestamp=TIMESTAMP, require_config=True)


def test_parse_gs_uri():
    assert parse_gs_uri("gs://b/p/q") == ("b", "p/q")
    assert parse_gs_uri("gs://b") == ("b", "")
    assert parse_gs_uri("gs://b/") == ("b", "")
    with pytest.raises(SystemExit):
        parse_gs_uri("/local/path")


def test_sanitize_job_name():
    raw = "detector-v5_FT.joint-2026-07-20_15-16-57"
    name = sanitize_job_name(raw)
    assert name == "detector-v5-ft-joint-2026-07-20-15-16-57"
    assert len(name) <= 63
    assert name[0].isalpha()
    # a leading digit gets a letter prefix
    assert sanitize_job_name("2026-run").startswith("pimm-")


def test_build_batch_job_structure():
    cfg = _config()
    job, script = build_batch_job(cfg, cfg["run"]["name"])

    task_spec = job["taskGroups"][0]["taskSpec"]
    container = task_spec["runnables"][0]["container"]
    instance = job["allocationPolicy"]["instances"][0]

    # gs:// bucket is mounted; EXP_ROOT points at the mount, not the URI.
    assert task_spec["volumes"][0]["gcs"]["remotePath"] == "my-bucket"
    assert task_spec["volumes"][0]["mountPath"] == "/mnt/gcs"
    assert "export EXP_ROOT=/mnt/gcs/pimm_exp" in script
    assert "gs://" not in script

    # Cloud Batch runs the image directly: no nested docker, train.sh invoked.
    assert container["imageUri"] == "docker.io/gkrz/pimm_dev:v1"
    assert container["entrypoint"] == "/bin/bash"
    assert "docker run" not in script
    assert "sh /opt/pimm/src/scripts/train.sh -m 1 -g 1" in script

    # A100 VM with drivers installed, and the time budget becomes a duration.
    assert instance["policy"]["machineType"] == "a2-highgpu-1g"
    assert instance["installGpuDrivers"] is True
    assert task_spec["maxRunDuration"] == "86400s"

    # spec is JSON-serializable
    json.dumps(job)


def test_explicit_accelerator_for_non_a2():
    cfg = _config()
    cfg["resources"]["scheduler_options"].update(
        machine_type="n1-standard-8",
        accelerator_type="nvidia-tesla-a100",
        accelerator_count=1,
    )
    job, _ = build_batch_job(cfg, cfg["run"]["name"])
    accel = job["allocationPolicy"]["instances"][0]["policy"]["accelerators"][0]
    assert accel == {"type": "nvidia-tesla-a100", "count": 1}


def test_non_gs_exp_root_rejected():
    cfg = _config()
    cfg["paths"]["exp_root"] = "/local/exp"
    with pytest.raises(SystemExit, match="gs://"):
        validate_launch_config(cfg)


def test_missing_project_rejected():
    cfg = _config()
    del cfg["resources"]["scheduler_options"]["project"]
    with pytest.raises(SystemExit, match="project"):
        validate_launch_config(cfg)
