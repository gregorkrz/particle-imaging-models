import hashlib
import os
import signal
import subprocess
from pathlib import Path
from typing import Callable

import pytest


Environment = dict[str, str]
Command = list[str]


class ProcessResult:
    def __init__(self, process, output=None):
        self.returncode = process.returncode
        self.pid = process.pid
        self.output = output

    def __repr__(self):
        return f"ProcessResult(returncode={self.returncode}, pid={self.pid})"


@pytest.fixture(scope="module")
def run_process() -> Callable[..., ProcessResult]:
    def _run_process(command, env=None, timeout=None, capture=False):
        stream = subprocess.PIPE if capture else None
        process = subprocess.Popen(
            command,
            env={**os.environ, **(env or {})},
            start_new_session=True,
            stdout=stream,
            stderr=subprocess.STDOUT if capture else None,
            text=True if capture else None,
        )
        output = None
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
        return ProcessResult(process, output)

    return _run_process


PILARNET_MINI_REPO = "DeepLearnPhysics/PILArNet-M-mini"
PILARNET_MINI_REVISION = "c218be4a76948bb44c3dcd7964f4b22c6c0052f4"
PILARNET_MINI_FILES = {
    "train/generic_v2_80_v2.h5": (
        "c8056fa915efb50bff1e06e746240ed0cce166b41693502b361b208fe3296f1f",
        80,
    ),
    "val/generic_v2_20_v2.h5": (
        "4881c1651e4bcfbd91e39c7a18de907aa2fd2d8389de0ace8b96c1a22926ae64",
        20,
    ),
    "test/generic_v2_20_v2.h5": (
        "8c092b797736f64130826a63c28fa2493a00e5efbe1c38d62326b676b00cf8a5",
        20,
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pilarnet_mini(root):
    root = Path(root)
    for filename, (expected_sha256, _) in PILARNET_MINI_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing PILArNet-M-mini file: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
            )
    return root


@pytest.fixture(scope="session")
def pilarnet_mini_root(pytestconfig):
    configured = os.environ.get("PIMM_TEST_DATA_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        candidates = (root, root / "PILArNet-M-mini")
        root = next(
            (candidate for candidate in candidates if (candidate / "train").is_dir()),
            root,
        )
        return _validate_pilarnet_mini(root)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub is required for PILArNet-M-mini")

    cache_dir = pytestconfig.cache.mkdir("pimm-huggingface")
    paths = []
    try:
        for filename in PILARNET_MINI_FILES:
            paths.append(
                Path(
                    hf_hub_download(
                        repo_id=PILARNET_MINI_REPO,
                        repo_type="dataset",
                        revision=PILARNET_MINI_REVISION,
                        filename=filename,
                        cache_dir=cache_dir,
                    )
                )
            )
    except Exception as exc:
        pytest.skip(f"PILArNet-M-mini is unavailable: {exc}")

    roots = {path.parents[1].resolve() for path in paths}
    if len(roots) != 1:
        raise RuntimeError(f"PILArNet-M-mini files resolved to multiple roots: {roots}")
    return _validate_pilarnet_mini(roots.pop())
