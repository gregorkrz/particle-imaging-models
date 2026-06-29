"""Point-cloud serialization (z-order / Hilbert space-filling curves).

By default these run as pure-PyTorch ops. An optional CUDA backend
(:mod:`._cuda_backend`, built from ``libs/serialize_cuda``) transparently
replaces the Morton and Hilbert encoders for CUDA tensors when the
``serialize_cuda`` extension is installed, giving a faster serialization step on
the PTv3 hot path. It is fully opt-in: without the extension, or for CPU
tensors, the PyTorch path is used unchanged. Set
``PIMM_DISABLE_SERIALIZE_CUDA=1`` to force the PyTorch path.
"""

from .default import (
    encode,
    decode,
    z_order_encode,
    z_order_decode,
    hilbert_encode,
    hilbert_decode,
    encode_batch,
)
