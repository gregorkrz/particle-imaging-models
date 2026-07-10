# serialize_cuda

CUDA kernels for point-cloud serialization (Morton / z-order and Hilbert
space-filling curves), used as an optional drop-in accelerator for
`pimm.models.utils.serialization`.

Vendored from
[ChristianSchott/point_serialization_cuda](https://github.com/ChristianSchott/point_serialization_cuda)
at commit `ceebfd14814fb19d11bc622eb1375013362f6327`. Only the CUDA extension is
vendored — pimm keeps its own PyTorch serialization front-end. The kernel
sources (`*.cu`, `bindings.cpp`) are upstream's; `setup.py` was rewritten to
auto-detect the GPU arch / honor `TORCH_CUDA_ARCH_LIST` instead of hard-coding
`sm_86`.

## Build

Built automatically with the other local extensions by `uv sync`.
To force a rebuild:

```bash
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" \
  uv sync --locked --reinstall-package serialize-cuda
```

## Use

Importing it is enough — `pimm.models.utils.serialization` detects the
`serialize_cuda` module and routes CUDA tensors through these kernels, falling
back to the pure-PyTorch path otherwise. Set `PIMM_DISABLE_SERIALIZE_CUDA=1` to
force the PyTorch path. See `tests/test_serialize_cuda.py` for the parity check.

## Exposed ops

- `serialize_cuda.morton_encode(coords: uint32[N,3]) -> int64[N]`
- `serialize_cuda.hilbert_encode(coords: uint32[N,3], num_bits) -> int64[N]`
- `serialize_cuda.hilbert_encode_approx(coords: uint32[N,3], num_bits) -> int64[N]`
  (faster, non-exact curve)
