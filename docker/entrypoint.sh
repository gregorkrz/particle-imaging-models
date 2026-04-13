#!/bin/bash
# Container entrypoint for pimm.
#
# When Apptainer/Singularity bind-mounts the user's home directory, their
# ~/.bashrc is sourced on every shell invocation. This commonly contains
# conda/mamba init blocks that try to run binaries not present in the
# container, causing hangs or errors.
#
# This entrypoint neutralizes those init blocks by shadowing conda/mamba
# with no-ops before any shell has a chance to source ~/.bashrc.

# Make conda/mamba init blocks harmless — they call `conda shell.bash hook`
# or `mamba shell.bash hook` which don't exist in this container.
conda() { :; }
mamba() { :; }
micromamba() { :; }
export -f conda mamba micromamba

# Also prevent user site-packages from leaking in
export PYTHONNOUSERSITE=1

exec "$@"
