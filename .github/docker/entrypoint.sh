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

# If the current directory is a pimm checkout, prefer it as the source so users
# don't have to bind their clone over /opt/pimm/src. `import pimm` then resolves
# to the cwd checkout (PYTHONPATH precedes the baked-in editable install); if the
# cwd is not a checkout, the image's /opt/pimm/src is used as before. Set
# PIMM_NO_CWD=1 to opt out.
if [ -z "${PIMM_NO_CWD:-}" ] \
   && [ -f "$PWD/pimm/__init__.py" ] \
   && [ -f "$PWD/launch/defaults.yaml" ]; then
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$@"
