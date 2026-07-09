"""Optional CUDA backend for point serialization.

Wraps the ``serialize_cuda`` extension vendored under ``libs/serialize_cuda``
(from https://github.com/ChristianSchott/point_serialization_cuda). The
extension provides hand-written CUDA kernels for Morton (z-order) and Hilbert
encoding that replace the pure-PyTorch implementations in
:mod:`pimm.models.utils.serialization` on the hot path (roughly a 20%
end-to-end speedup on PTv3 inference per the upstream repo).

The backend is purely opt-in and degrades gracefully:

- If the extension was never built, :data:`HAS_SERIALIZE_CUDA` is ``False`` and
  callers fall back to the PyTorch path.
- It only ever activates for CUDA tensors; CPU tensors always use PyTorch so
  results stay correct on machines without a GPU.
- Set ``PIMM_DISABLE_SERIALIZE_CUDA=1`` to force the PyTorch path even when the
  extension is installed (useful for debugging or A/B benchmarking).

It is built automatically by the container and uv environment; see
``libs/serialize_cuda/README.md`` for the manual build.
"""

import os

import torch

try:
    import serialize_cuda as _serialize_cuda

    HAS_SERIALIZE_CUDA = True
except ImportError:
    _serialize_cuda = None
    HAS_SERIALIZE_CUDA = False


def _disabled() -> bool:
    return os.environ.get("PIMM_DISABLE_SERIALIZE_CUDA", "0") not in ("0", "", "false", "False")


def available(grid_coord: torch.Tensor) -> bool:
    """Return ``True`` if the CUDA kernels should handle ``grid_coord``."""
    return HAS_SERIALIZE_CUDA and not _disabled() and grid_coord.is_cuda


def morton_encode(grid_coord: torch.Tensor) -> torch.Tensor:
    """Morton (z-order) encode ``(N, 3)`` integer coords into ``(N,)`` int64 codes."""
    coords = grid_coord.to(torch.uint32).contiguous()
    return _serialize_cuda.morton_encode(coords).long()


def hilbert_encode(grid_coord: torch.Tensor, depth: int, approx: bool = False) -> torch.Tensor:
    """Hilbert encode ``(N, 3)`` integer coords into ``(N,)`` int64 codes.

    ``approx=True`` uses the faster but non-exact kernel (a slightly different
    space-filling curve); leave it ``False`` to match the PyTorch reference.
    """
    coords = grid_coord.to(torch.uint32).contiguous()
    if approx:
        return _serialize_cuda.hilbert_encode_approx(coords, depth).long()
    return _serialize_cuda.hilbert_encode(coords, depth).long()
