#!/usr/bin/env bash
# Single source of truth for pimm build/runtime pins.
#
# Sourced by install.sh, the other .github/docker/common/* scripts, and (by value) the
# Dockerfiles. Every value is overridable from the environment so a caller can
# build a variant without editing this file. If you bump TORCH/CUDA here, also
# bump the `FROM pytorch/pytorch:...` tag in .github/docker/Dockerfile* to match.

export TORCH_VERSION="${TORCH_VERSION:-2.5.0}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.0}"
export CUDA_VERSION="${CUDA_VERSION:-12.4}"
export CUDA_VERSION_NO_DOT="${CUDA_VERSION_NO_DOT:-124}"
export CUDNN_VERSION="${CUDNN_VERSION:-9}"
export FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.3}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Default GPU archs for building the CUDA extensions (A100 8.0, A6000/3090 8.6,
# L40/4090 8.9, H100 9.0). Override for your hardware; harmless to over-list.
# Find your GPU's arch here: https://developer.nvidia.com/cuda/gpus
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0}"
