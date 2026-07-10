"""Run the pimm test suite on Modal GPUs for trusted branch commits.

GitHub-hosted runners have no GPU, so this app ships the committed checkout to
Modal, builds the locked wheel-based environment (no native compile), bakes in
the commit-pinned PILArNet-M-mini dataset and released models, and runs the
shared pytest suite on CPU plus A100, L40S, and H100 workers.

Run locally with:

    uvx modal run .github/scripts/modal_ci.py
"""

import json
import os
import subprocess
from pathlib import Path

import modal


# remote containers re-import this module from a shallower path
_here = Path(__file__).resolve()
REPO_ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
REMOTE_ROOT = "/opt/pimm"
# venv lives outside the copied source dir: writes under an add_local_dir
# target during a build function do not persist into the image layer
VENV = "/opt/pimm-venv"
DATA_ROOT = "/opt/pimm-data/pilarnet"
HF_CACHE = "/opt/pimm-data/hf-cache"
MANIFEST = "tests/integration/fixtures/metric_baselines.json"

UV_VERSION = "0.11.28"
INTEGRATION_TIMEOUT = 1800
OUTPUT_TAIL = 8000

RUNTIME_ENV = {
    "HF_HUB_DISABLE_XET": "1",
    "PILARNET_DATA_ROOT_V2": DATA_ROOT,
    "PIMM_HF_CACHE": HF_CACHE,
    "PIMM_TEST_DATA_ROOT": DATA_ROOT,
    "PYTHONPATH": REMOTE_ROOT,
    "UV_PROJECT_ENVIRONMENT": VENV,
    "WANDB_MODE": "disabled",
}


def _sync_environment():
    """Install the locked environment into the image.

    A build function rather than a run_commands layer: uv's ``--locked`` check
    passes in the function sandbox but rejects the lock in a build layer.
    """
    subprocess.run(
        ["uv", "sync", "--locked", "--group", "dev"],
        cwd=REMOTE_ROOT,
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": VENV,
            "UV_CACHE_DIR": "/tmp/uv-cache",
        },
        check=True,
    )


def _prime_cache():
    """Bake the pinned dataset and released models into the image."""
    import hashlib

    from huggingface_hub import hf_hub_download, snapshot_download

    manifest = json.loads(Path(REMOTE_ROOT, MANIFEST).read_text())
    dataset = manifest["dataset"]

    for filename, info in dataset["files"].items():
        path = Path(
            hf_hub_download(
                repo_id=dataset["repo_id"],
                repo_type=dataset["repo_type"],
                revision=dataset["revision"],
                filename=filename,
                local_dir=DATA_ROOT,
            )
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise SystemExit(f"sha256 mismatch for {filename}: {digest}")

    for baseline in manifest["baselines"].values():
        repo_id = baseline["model_uri"].removeprefix("hf://").split("@", 1)[0]
        snapshot_download(
            repo_id=repo_id,
            revision=baseline["model_revision"],
            cache_dir=HF_CACHE,
        )


image = (
    # base CUDA/cuDNN must match the locked torch stack (torch 2.13.0, cu126)
    modal.Image.from_registry(
        "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel",
        add_python="3.10",
    )
    .pip_install("huggingface_hub", f"uv=={UV_VERSION}")
    .add_local_dir(
        REPO_ROOT,
        REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".venv*",
            "**/__pycache__",
            "**/*.pyc",
            "**/build",
            "**/*.egg-info",
            "**/*.so",
            "docs/build",
        ],
    )
    .run_function(_sync_environment)
    .run_function(_prime_cache)
    .env(RUNTIME_ENV)
)

app = modal.App("pimm-ci", image=image)


def _pytest(*args):
    """Run pytest with the given args in the locked environment and report the result."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "pytest", "-v", *args],
        cwd=REMOTE_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output)
    return {"returncode": result.returncode, "tail": output[-OUTPUT_TAIL:]}


# single-GPU cells skip the distributed tests; the A10:2 cell runs only those
SINGLE_GPU_ARGS = ("tests/integration", "-m", "not distributed")


@app.function()
def unit():
    return _pytest("tests/unit")


@app.function(gpu="A100", timeout=INTEGRATION_TIMEOUT)
def integration_a100():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="L40S", timeout=INTEGRATION_TIMEOUT)
def integration_l40s():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="H100", timeout=INTEGRATION_TIMEOUT)
def integration_h100():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="A10:2", timeout=INTEGRATION_TIMEOUT)
def integration_distributed():
    return _pytest("tests/integration", "-m", "distributed")


@app.local_entrypoint()
def main():
    calls = {
        "cpu-unit": unit.spawn(),
        "a100-integration": integration_a100.spawn(),
        "l40s-integration": integration_l40s.spawn(),
        "h100-integration": integration_h100.spawn(),
        "a10x2-distributed": integration_distributed.spawn(),
    }
    results = {name: call.get() for name, call in calls.items()}

    failed = []
    for name, result in results.items():
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        print(f"[{status}] {name} (returncode={result['returncode']})")
        if result["returncode"] != 0:
            print(result["tail"])
            failed.append(name)

    if failed:
        raise SystemExit("modal ci failed: " + ", ".join(sorted(failed)))
    print("modal ci passed on cpu, a100, l40s, h100, and a10x2")
