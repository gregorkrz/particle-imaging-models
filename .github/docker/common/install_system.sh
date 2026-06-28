#!/usr/bin/env bash
# System (apt) packages needed to build the CUDA extensions and run pimm.
# Docker-only (the local conda path gets its toolchain from conda). Extra package
# names may be passed as args (e.g. NERSC adds gfortran pkg-config).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update --no-install-recommends
apt-get install -y --no-install-recommends \
  git wget tmux vim zsh build-essential cmake ninja-build \
  libopenblas-dev libsparsehash-dev "$@"
apt-get autoremove -y && apt-get clean -y
rm -rf /var/lib/apt/lists/*
