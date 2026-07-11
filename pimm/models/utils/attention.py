"""Drop-in replacement for the flash-attn varlen API.

Mirrors flash_attn.flash_attn_varlen_func / flash_attn_varlen_qkvpacked_func
on top of torch.nn.attention.varlen_attn, which routes to torch's built-in
Flash Attention or cuDNN kernels depending on hardware.
"""

import torch
from torch.nn.attention.varlen import varlen_attn


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Attention over packed variable-length sequences.

    q: (total_q, heads, head_dim); k, v: (total_k, heads, head_dim).
    Returns (total_q, heads, head_dim).
    """
    if dropout_p:
        raise NotImplementedError(
            "torch.nn.attention.varlen_attn does not support dropout; "
            "set attn_drop to 0 or disable the flash attention path"
        )
    if causal:
        window_size = (window_size[0], 0)
    # torch's varlen kernels require head_dim % 8 == 0. Zero-padding the head
    # dimension is exact: padded dims contribute nothing to q.k scores, and the
    # padded value dims are sliced off. The softmax scale must be pinned to the
    # original head_dim before padding changes the default.
    head_dim = q.shape[-1]
    pad = -head_dim % 8
    if pad:
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        q, k, v = (
            torch.nn.functional.pad(tensor, (0, pad)) for tensor in (q, k, v)
        )
    out = varlen_attn(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        scale=softmax_scale, window_size=tuple(window_size),
    )
    if pad:
        out = out[..., :head_dim]
    return out


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Self-attention over packed qkv of shape (total, 3, heads, head_dim)."""
    # pad the packed tensor before unbinding: one allocation and one backward
    # op instead of three (see the head_dim note in flash_attn_varlen_func)
    head_dim = qkv.shape[-1]
    pad = -head_dim % 8
    if pad:
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        qkv = torch.nn.functional.pad(qkv, (0, pad))
    q, k, v = qkv.unbind(dim=1)
    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p, softmax_scale, causal, window_size,
    )
    if pad:
        out = out[..., :head_dim]
    return out
