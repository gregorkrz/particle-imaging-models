import importlib

import pytest


pytestmark = pytest.mark.gpu

CUDA_MODULES = (
    "cnms",
    "pointgroup_ops",
    "pointops",
    "pointrope",
    "pytorch3d_ops",
    "serialize_cuda",
)


def test_cuda_runtime_and_serialization():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    for module in CUDA_MODULES:
        importlib.import_module(module)

    from pimm.models.utils.serialization import _cuda_backend
    from pimm.models.utils.serialization.default import encode, encode_batch
    from pimm.models.utils.serialization.hilbert import encode_int
    from pimm.models.utils.serialization.z_order import xyz2key

    value = torch.arange(128, device="cuda", dtype=torch.float32)
    assert torch.equal(value.square(), value * value)
    assert _cuda_backend.HAS_SERIALIZE_CUDA

    depth = 12
    generator = torch.Generator(device="cuda").manual_seed(0)
    grid = torch.randint(
        0,
        2**depth,
        (8192, 3),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    batch = torch.arange(grid.shape[0], device="cuda") % 4

    morton = _cuda_backend.morton_encode(grid)
    morton_reference = xyz2key(
        grid[:, 0].long(),
        grid[:, 1].long(),
        grid[:, 2].long(),
        depth=depth,
    )
    assert torch.equal(morton, morton_reference)

    hilbert = _cuda_backend.hilbert_encode(grid, depth)
    hilbert_reference = encode_int(grid, num_dims=3, num_bits=depth)
    assert torch.equal(hilbert, hilbert_reference)

    orders = ("z", "hilbert", "z-trans", "hilbert-trans")
    batched = encode_batch(grid, batch, depth=depth, orders=orders)
    assert batched.shape == (len(orders), grid.shape[0])
    for index, order in enumerate(orders):
        assert torch.equal(
            batched[index],
            encode(grid, batch, depth=depth, order=order),
        )
