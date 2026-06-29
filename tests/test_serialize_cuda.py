"""
Parity + smoke test for the optional ``serialize_cuda`` backend.

The CUDA kernels (vendored at ``libs/point_serialization_cuda``) must produce
*exactly* the same serialization codes as the pure-PyTorch reference, otherwise
swapping them in would silently change the point ordering a model sees.

This loads the serialization module by file path so it runs without importing
the full ``pimm`` package (and its training-only deps). It is a no-op unless a
CUDA device is present AND the ``serialize_cuda`` extension has been built
(``pip install ./libs/serialize_cuda``; see ``libs/serialize_cuda/README.md``).

Run: python tests/test_serialize_cuda.py
"""

import importlib.util
import os
import sys
import types

import torch

_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pimm", "models", "utils", "serialization",
)


def _load_serialization():
    pkg = types.ModuleType("ser")
    pkg.__path__ = [_BASE]
    sys.modules["ser"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location("ser." + name, os.path.join(_BASE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ser." + name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("z_order")
    load("hilbert")
    backend = load("_cuda_backend")
    default = load("default")
    return default, backend


def main():
    default, backend = _load_serialization()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return 0
    if not backend.HAS_SERIALIZE_CUDA:
        print("SKIP: serialize_cuda extension not built "
              "(see libs/serialize_cuda)")
        return 0

    device = torch.device("cuda")
    depth = 14
    torch.manual_seed(0)
    # stay within 2**depth so codes fit and the reference path is exercised fully
    grid = torch.randint(0, 2 ** depth, (200_000, 3), dtype=torch.int32, device=device)
    batch = torch.zeros(grid.shape[0], dtype=torch.long, device=device)

    failures = 0

    def check(name, ok):
        nonlocal failures
        print(("PASS" if ok else "FAIL") + ": " + name)
        if not ok:
            failures += 1

    # --- Morton / z-order: CUDA kernel vs PyTorch LUT reference ---
    cuda_z = backend.morton_encode(grid)
    x, y, z = grid[:, 0].long(), grid[:, 1].long(), grid[:, 2].long()
    ref_z = sys.modules["ser.z_order"].xyz2key(x, y, z, b=None, depth=depth)
    check("morton_encode matches PyTorch z-order", torch.equal(cuda_z, ref_z))

    # --- Hilbert: CUDA kernel vs PyTorch Skilling reference ---
    cuda_h = backend.hilbert_encode(grid, depth)
    ref_h = sys.modules["ser.hilbert"].encode_int(grid, num_dims=3, num_bits=depth)
    check("hilbert_encode matches PyTorch Hilbert", torch.equal(cuda_h, ref_h))

    # --- End-to-end: encode() and encode_batch() route through the kernels ---
    for order in ("z", "hilbert", "z-trans", "hilbert-trans"):
        c = default.encode(grid, batch, depth=depth, order=order)
        check(f"encode(order={order}) shape", c.shape == (grid.shape[0],))

    orders = ("hilbert", "z", "hilbert-trans")
    cb = default.encode_batch(grid, batch, depth=depth, orders=orders)
    ok = all(
        torch.equal(cb[i], default.encode(grid, batch, depth=depth, order=o))
        for i, o in enumerate(orders)
    )
    check("encode_batch rows match per-order encode", ok)

    print("\n" + ("ALL PASSED" if failures == 0 else f"{failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
