#!/usr/bin/env bash
# system packages needed to build the CUDA extensions and run pimm
# extra package names may be passed as args, such as NERSC build dependencies
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update --no-install-recommends
apt-get install -y --no-install-recommends \
  git wget tmux vim zsh build-essential cmake ninja-build \
  libopenblas-dev "$@"
apt-get autoremove -y && apt-get clean -y
rm -rf /var/lib/apt/lists/*
