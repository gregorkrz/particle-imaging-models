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

# The image ships no pimm source: when the current directory is a pimm
# checkout (bound by the user or by pimm submit), put it on PYTHONPATH so
# `import pimm` and the `pimm` shim resolve to it. Set PIMM_NO_CWD=1 to opt
# out.
if [ -z "${PIMM_NO_CWD:-}" ] \
   && [ -f "$PWD/pimm/__init__.py" ] \
   && [ -f "$PWD/launch/defaults.yaml" ]; then
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$@"
