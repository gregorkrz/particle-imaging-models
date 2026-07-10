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
    return varlen_attn(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        scale=softmax_scale, window_size=tuple(window_size),
    )


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
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p, softmax_scale, causal, window_size,
    )
