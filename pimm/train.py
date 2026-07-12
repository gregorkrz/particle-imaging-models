"""Native training entrypoint for pimm configs.

This module is executed under ``torchrun`` for local and Slurm jobs. The public
``pimm launch`` and ``pimm submit`` commands render the torchrun invocation
directly; each process parses the same config, initializes distributed state
from the environment when available, builds the configured trainer, and runs
training.

Modified from the original Pointcept ``tools/train.py``.
"""

import sys
import os
import logging

# Drop the script dir (the `pimm/` package dir) from sys.path so pimm submodules
# don't shadow installed distributions (e.g. `datasets` -> HuggingFace, not
# `pimm.datasets`); make the repo root importable instead.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _script_dir]
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pimm.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pimm.engines.train import TRAINERS
from pimm.utils import comm


def main_worker(cfg):
    """Build and run the trainer after config normalization."""
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def main():
    """Parse CLI args, initialize distributed state, and start training."""
    logging.basicConfig(level=logging.INFO)
    
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    try:
        comm.setup_distributed()
        main_worker(cfg)
    finally:
        comm.cleanup_distributed()

if __name__ == "__main__":
    main()
