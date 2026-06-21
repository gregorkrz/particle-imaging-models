"""Checkpoint format, IO, and resume management for pimm training runtimes."""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

import pimm.utils.comm as comm
from pimm.datasets.stateful import (
    assert_exact_dataloader_state_available,
    dataloader_state_dict,
    load_dataloader_state_dict,
)
from pimm.engines._train_utils import (
    TrainState,
    apply_train_state_to_trainer,
    capture_distributed_rng_state,
    capture_rng_state,
    restore_distributed_rng_state,
)
from pimm.utils.comm import is_main_process, synchronize
from pimm.utils.path import (
    checkpoint_success_file as _dcp_success_file,
    is_complete_dcp_checkpoint,
    is_complete_split_checkpoint,
    latest_complete_checkpoint,
    resolve_model_weight_file,
    split_checkpoint_trainer_dir,
    split_checkpoint_weight_file,
)


def _distributed_object_state(local_state):
    """Gather one Python state object per rank into a checkpointable wrapper."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        states = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(states, local_state)
        return {
            "_pimm_distributed_state": True,
            "world_size": world_size,
            "states": states,
        }
    return local_state


def local_object_state(state, *, strict=True):
    """Return the current rank's state from a distributed object wrapper."""
    if not isinstance(state, dict) or not state.get("_pimm_distributed_state"):
        return state
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0
    saved_world_size = int(state.get("world_size", len(state.get("states", []))))
    if strict and saved_world_size != world_size:
        raise ValueError(
            f"State was saved with world_size={saved_world_size}, "
            f"but current world_size={world_size}."
        )
    states = state.get("states", [])
    if rank >= len(states):
        if strict:
            raise ValueError(f"No distributed state available for rank {rank}.")
        rank = 0
    return states[rank]


def build_checkpoint_payload(trainer, *, distributed_rng=False):
    """Build the structured checkpoint payload consumed by checkpoint loads."""
    train_state = TrainState.from_trainer(trainer)
    local_dataloader_state = dataloader_state_dict(trainer.train_loader)
    assert_exact_dataloader_state_available(
        local_dataloader_state,
        loader=trainer.train_loader,
        iter_in_epoch=train_state.iter_in_epoch,
    )
    train_state.dataloader_state = (
        _distributed_object_state(local_dataloader_state)
        if distributed_rng
        else local_dataloader_state
    )
    rng_state = (
        capture_distributed_rng_state()
        if distributed_rng
        else capture_rng_state()
    )
    train_state.rng_state = rng_state
    model_state = trainer.model.state_dict()
    optimizer_state = get_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        options=StateDictOptions(),
    )
    scheduler_state = trainer.scheduler.state_dict()
    scaler_state = (
        trainer.scaler.state_dict()
        if getattr(trainer, "scaler", None) is not None
        else None
    )
    world_size = comm.get_world_size()
    distributed_backend = (
        torch.distributed.get_backend()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else None
    )
    logger_state = {
        "backend": "wandb" if getattr(trainer.cfg, "use_wandb", False) else "tensorboard",
        "wandb": {
            "group": trainer.cfg.get("wandb_group", None),
            "run_name": trainer.cfg.get("wandb_run_name", None),
            "run_id": trainer.cfg.get("wandb_run_id", None),
            "job_type": trainer.cfg.get("wandb_job_type", None),
            "resume": trainer.cfg.get("wandb_resume", None),
            "step_offset": trainer.cfg.get("log_step_offset", 0),
            "checkpoint_global_step": train_state.global_step,
        },
    }
    return {
        "schema": "pimm.trainer_checkpoint",
        "version": 3,
        "checkpoint_version": 3,
        "model": {"state_dict": model_state},
        "optimizer": {
            "state_dict": optimizer_state,
            "class": trainer.optimizer.__class__.__name__,
            "format": "torch.distributed.checkpoint.state_dict",
        },
        "scheduler": {
            "state_dict": scheduler_state,
            "class": trainer.scheduler.__class__.__name__,
            "total_steps": getattr(trainer.scheduler, "total_steps", None),
        },
        "scaler": {
            "enabled": bool(getattr(trainer.cfg, "enable_amp", False)),
            "state_dict": scaler_state,
        },
        "dataloader": {
            "backend": trainer.train_loader.__class__.__name__,
            "state": train_state.dataloader_state,
            "world_size": world_size,
            "batch_size_per_rank": getattr(trainer.cfg, "batch_size_per_gpu", None),
            "num_workers": getattr(trainer.cfg, "num_worker_per_gpu", None),
            "drop_last": getattr(trainer.train_loader, "drop_last", None),
        },
        "rng": {
            "world_size": world_size,
            "state": rng_state,
        },
        "trainer": {
            "epoch": train_state.epoch,
            "iter_in_epoch": train_state.iter_in_epoch,
            "global_step": train_state.global_step,
            "samples_seen": train_state.samples_seen,
            "best_metric_value": train_state.best_metric_value,
        },
        "logger": logger_state,
        "distributed": {
            "world_size": world_size,
            "backend": distributed_backend,
            "rank_order": list(range(world_size)),
        },
    }


def empty_checkpoint_payload(trainer):
    """Build an empty typed payload for DCP load to fill in place."""
    payload = build_checkpoint_payload(trainer, distributed_rng=True)
    payload["trainer"]["best_metric_value"] = -float("inf")
    payload["trainer"].update(
        {"epoch": 0, "iter_in_epoch": 0, "global_step": 0, "samples_seen": 0}
    )
    return payload


def checkpoint_model_state_dict(checkpoint):
    """Extract model weights from structured or legacy checkpoint formats."""
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict) and "state_dict" in checkpoint["model"]:
            return checkpoint["model"]["state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def checkpoint_optimizer_state_dict(checkpoint):
    """Extract optimizer state from structured or legacy checkpoints."""
    optimizer = checkpoint.get("optimizer", None)
    if isinstance(optimizer, dict) and "state_dict" in optimizer:
        return optimizer["state_dict"]
    return optimizer


def checkpoint_scheduler_state_dict(checkpoint):
    """Extract scheduler state from structured or legacy checkpoints."""
    scheduler = checkpoint.get("scheduler", None)
    if isinstance(scheduler, dict) and "state_dict" in scheduler:
        return scheduler["state_dict"]
    return scheduler


def checkpoint_scaler_state_dict(checkpoint):
    """Extract AMP scaler state from structured or legacy checkpoints."""
    scaler = checkpoint.get("scaler", None)
    if isinstance(scaler, dict) and "state_dict" in scaler:
        return scaler["state_dict"]
    return scaler


def checkpoint_dataloader_state(checkpoint, train_state=None):
    """Extract dataloader resume state, preferring parsed TrainState."""
    if train_state is not None and train_state.dataloader_state is not None:
        return train_state.dataloader_state
    dataloader = checkpoint.get("dataloader", None)
    if isinstance(dataloader, dict) and "state" in dataloader:
        return dataloader["state"]
    return dataloader


def checkpoint_rng_state(checkpoint, train_state=None):
    """Extract RNG resume state, preferring parsed TrainState."""
    if train_state is not None and train_state.rng_state is not None:
        return train_state.rng_state
    rng = checkpoint.get("rng", None)
    if isinstance(rng, dict) and "state" in rng:
        return rng["state"]
    return checkpoint.get("rng_state", None)


def checkpoint_train_state(checkpoint):
    """Parse structured trainer state, returning None for legacy checkpoints."""
    if checkpoint.get("train_state", None) is not None:
        return TrainState.from_state_dict(checkpoint["train_state"])
    trainer_state = checkpoint.get("trainer", None)
    if not isinstance(trainer_state, dict):
        return None
    dataloader_state = checkpoint_dataloader_state(checkpoint)
    rng_state = checkpoint_rng_state(checkpoint)
    dataloader = checkpoint.get("dataloader", {})
    world_size = (
        int(dataloader.get("world_size", comm.get_world_size()))
        if isinstance(dataloader, dict)
        else comm.get_world_size()
    )
    return TrainState(
        schema_version=int(checkpoint.get("checkpoint_version", checkpoint.get("version", 0)) or 0),
        epoch=int(trainer_state.get("epoch", 0)),
        iter_in_epoch=int(trainer_state.get("iter_in_epoch", trainer_state.get("iteration", 0))),
        global_step=int(trainer_state.get("global_step", 0)),
        samples_seen=int(trainer_state.get("samples_seen", 0)),
        world_size=world_size,
        batch_size_per_rank=(
            dataloader.get("batch_size_per_rank") if isinstance(dataloader, dict) else None
        ),
        best_metric_value=trainer_state.get("best_metric_value"),
        rng_state=rng_state,
        dataloader_state=dataloader_state,
    )


def build_trainer_state_payload(checkpoint):
    """Return checkpoint state with model weights removed."""
    return {key: value for key, value in checkpoint.items() if key != "model"}


def empty_trainer_state_payload(trainer):
    """Build an empty typed payload for split-checkpoint trainer state loading."""
    return build_trainer_state_payload(empty_checkpoint_payload(trainer))


def atomic_torch_save(payload, filename):
    """Save a torch checkpoint via a temp file and one-level backup."""
    tmp = filename + ".tmp"
    prev = filename + ".prev"
    torch.save(payload, tmp)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    if os.path.exists(prev):
        os.remove(prev)
    if os.path.exists(filename):
        os.replace(filename, prev)
    os.replace(tmp, filename)


def save_model_weights_file(payload, filename):
    """Save just the model weights as a portable single-file checkpoint.

    Produces the same ``{"state_dict": ...}`` layout as the split-checkpoint
    weight file, so it loads identically via ``checkpoint_model_state_dict`` and
    a plain ``torch.load`` (used for ``model_best.pth`` and iter snapshots).
    """
    atomic_torch_save({"state_dict": checkpoint_model_state_dict(payload)}, filename)


def save_dcp_checkpoint(payload, checkpoint_dir):
    """Save a distributed checkpoint directory with atomic publish semantics."""
    import torch.distributed.checkpoint as dcp

    tmp_dir = checkpoint_dir + ".tmp"
    prev_dir = checkpoint_dir + ".prev"
    if is_main_process():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    synchronize()

    dcp.save(payload, checkpoint_id=tmp_dir)
    if is_main_process():
        with open(_dcp_success_file(tmp_dir), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        if os.path.exists(checkpoint_dir):
            os.replace(checkpoint_dir, prev_dir)
        os.replace(tmp_dir, checkpoint_dir)
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
    synchronize()


def save_split_checkpoint(payload, checkpoint_dir):
    """Save model weights plus DCP trainer state without duplicating tensors."""
    tmp_dir = checkpoint_dir + ".tmp"
    prev_dir = checkpoint_dir + ".prev"
    if is_main_process():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        atomic_torch_save(
            {"state_dict": checkpoint_model_state_dict(payload)},
            split_checkpoint_weight_file(tmp_dir),
        )
    synchronize()

    save_dcp_checkpoint(
        build_trainer_state_payload(payload),
        split_checkpoint_trainer_dir(tmp_dir),
    )
    if is_main_process():
        with open(_dcp_success_file(tmp_dir), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        if os.path.exists(checkpoint_dir):
            os.replace(checkpoint_dir, prev_dir)
        os.replace(tmp_dir, checkpoint_dir)
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
    synchronize()


def load_dcp_trainer_state(checkpoint_dir, trainer):
    """Load a complete trainer-state DCP into a typed placeholder payload."""
    import torch.distributed.checkpoint as dcp

    if not is_complete_dcp_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    payload = empty_trainer_state_payload(trainer)
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    dcp.load(payload, checkpoint_id=checkpoint_dir,
             planner=DefaultLoadPlanner(allow_partial_load=True))
    return payload


def load_dcp_checkpoint(checkpoint_dir, trainer):
    """Load a complete full DCP checkpoint into a typed placeholder payload."""
    import torch.distributed.checkpoint as dcp

    if not is_complete_dcp_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    payload = empty_checkpoint_payload(trainer)
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    dcp.load(payload, checkpoint_id=checkpoint_dir,
             planner=DefaultLoadPlanner(allow_partial_load=True))
    return payload


def load_split_checkpoint(checkpoint_dir, trainer, map_location):
    """Load a split checkpoint for exact resume."""
    if not is_complete_split_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete split checkpoint: {checkpoint_dir}")
    checkpoint = load_dcp_trainer_state(split_checkpoint_trainer_dir(checkpoint_dir), trainer)
    weight_checkpoint = torch.load(
        split_checkpoint_weight_file(checkpoint_dir),
        map_location=map_location,
        weights_only=False,
    )
    checkpoint["model"] = {"state_dict": checkpoint_model_state_dict(weight_checkpoint)}
    return checkpoint


def _cfg_get(cfg, key, default=None):
    """Read config values from dict-like or attribute-style config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


class CheckpointManager:
    """Own checkpoint format, save/load backends, and trainer resume semantics."""

    def __init__(self, trainer):
        self.trainer = trainer

    def _checkpoint_format(self, hook_backend=None):
        """Resolve the on-disk checkpoint format."""
        aliases = {"dcp": "standard", "torch": "legacy"}
        fmt = _cfg_get(self.trainer.cfg, "checkpoint_format", None)
        if fmt is None:
            fmt = hook_backend
        fmt = str(fmt or "standard").lower()
        fmt = aliases.get(fmt, fmt)
        if fmt not in ("standard", "legacy"):
            raise ValueError(
                "checkpoint_format must be 'standard' or 'legacy' "
                f"(or the deprecated 'dcp'/'torch'), got {fmt!r}"
            )
        return fmt

    def _write_checkpoint(
        self, payload, *, fmt, is_best, step_count, save_freq, save_iter_checkpoints
    ):
        """Write a built payload in the resolved format.

        Must be called on ALL ranks: the ``standard`` format performs a
        collective DCP save. Rank-0-only side artifacts (the best/iter weight
        files and the legacy single file) are guarded internally.
        """
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        best_file = os.path.join(model_dir, "model_best.pth")
        do_iter_snapshot = bool(
            save_iter_checkpoints and save_freq and step_count and step_count % save_freq == 0
        )
        if fmt == "standard":
            last_dir = os.path.join(model_dir, "last")
            if is_main_process():
                self.trainer.logger.info(
                    f"Saving checkpoint to: {last_dir} (weights.pth + trainer/ DCP)"
                )
            save_split_checkpoint(payload, last_dir)  # collective: all ranks
            if is_main_process():
                if is_best:
                    save_model_weights_file(payload, best_file)
                if do_iter_snapshot:
                    save_model_weights_file(
                        payload, os.path.join(model_dir, f"iter_{step_count}.pth")
                    )
            return
        # legacy: single monolithic file, written by rank 0 only
        if is_main_process():
            filename = os.path.join(model_dir, "model_last.pth")
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            atomic_torch_save(payload, filename)
            if is_best:
                shutil.copyfile(filename, best_file)
            if do_iter_snapshot:
                shutil.copyfile(
                    filename, os.path.join(model_dir, f"iter_{step_count}.pth")
                )

    def save_epoch_checkpoint(self, *, is_best=False, step_count=0, save_freq=None):
        """Save an epoch/metric-oriented checkpoint. Must run on all ranks."""
        fmt = self._checkpoint_format()
        payload = build_checkpoint_payload(self.trainer, distributed_rng=True)
        self._write_checkpoint(
            payload,
            fmt=fmt,
            is_best=is_best,
            step_count=step_count,
            save_freq=save_freq,
            save_iter_checkpoints=bool(save_freq),
        )

    def save_iteration_checkpoint(
        self,
        *,
        backend=None,
        is_best=False,
        step_count=0,
        save_freq=None,
        save_iter_checkpoints=False,
    ):
        """Save an iteration-oriented checkpoint. Must run on all ranks."""
        fmt = self._checkpoint_format(backend)
        payload = build_checkpoint_payload(self.trainer, distributed_rng=True)
        self._write_checkpoint(
            payload,
            fmt=fmt,
            is_best=is_best,
            step_count=step_count,
            save_freq=save_freq,
            save_iter_checkpoints=save_iter_checkpoints,
        )

    def load_weight_and_resume(self, *, keywords="", replacement=None, strict=False):
        """Load configured weights and restore training state when cfg.resume is true."""
        replacement = replacement if replacement is not None else keywords
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        weight_path = self.trainer.cfg.weight
        if weight_path and (os.path.isfile(weight_path) or os.path.isdir(weight_path)):
            self.trainer.logger.info(f"Loading weight at: {weight_path}")
            checkpoint = self._load_checkpoint(weight_path)
            self._load_model_weights(
                checkpoint,
                keywords=keywords,
                replacement=replacement,
                strict=strict,
            )
            if self.trainer.cfg.resume:
                self.resume_training_state(checkpoint)
            return

        message = f"No weight found at: {weight_path}"
        # A non-empty weight path that does not resolve is always an error: the
        # user asked to load weights, so silently training from random init would
        # hide a typo'd/moved checkpoint. Only the genuinely-unset case is a no-op
        # (unless resuming, which requires a checkpoint).
        if weight_path or self.trainer.cfg.resume:
            raise FileNotFoundError(message)
        self.trainer.logger.info(message)

    def _load_checkpoint(self, weight_path):
        """Load a direct, split, or directory checkpoint reference."""
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else "cpu"
        if os.path.isdir(weight_path):
            if is_complete_split_checkpoint(weight_path):
                if self.trainer.cfg.resume:
                    return load_split_checkpoint(weight_path, self.trainer, map_location)
                weight_file = resolve_model_weight_file(weight_path)
                return torch.load(weight_file, map_location=map_location, weights_only=False)
            if is_complete_dcp_checkpoint(weight_path):
                return load_dcp_checkpoint(weight_path, self.trainer)
            if self.trainer.cfg.resume:
                raise FileNotFoundError(f"Incomplete checkpoint directory: {weight_path}")
            weight_file = resolve_model_weight_file(weight_path)
            return torch.load(weight_file, map_location=map_location, weights_only=False)
        return torch.load(weight_path, map_location=map_location, weights_only=False)

    def _load_model_weights(self, checkpoint, *, keywords="", replacement="", strict=False):
        """Load checkpoint model weights with the existing keyword rewrite rules."""
        self.trainer.logger.info(
            f"Loading layer weights with keyword: {keywords}, "
            f"replace keyword with: {replacement}"
        )
        weight = OrderedDict()
        strip_module = lambda s: s[7:] if s.startswith("module.") else s # noqa: E731
        kw = strip_module(keywords)
        repl = strip_module(replacement)
        for key, value in checkpoint_model_state_dict(checkpoint).items():
            bare = strip_module(key)
            if kw and bare.startswith(kw):
                bare = repl + bare[len(kw):]
            key = bare if comm.get_world_size() == 1 else "module." + bare
            weight[key] = value
        missing, unexpected = self.trainer.model.load_state_dict(weight, strict=strict)
        n_model = len(self.trainer.model.state_dict())
        n_loaded = n_model - len(missing)
        self.trainer.logger.info(
            f"Loaded {n_loaded}/{n_model} model params "
            f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
        )
        if missing:
            self.trainer.logger.info(f"Missing keys: {missing}")
        if unexpected:
            self.trainer.logger.info(f"Unexpected keys: {unexpected}")
        # Guard against a remap that matched nothing: with strict=False this would
        # otherwise leave the whole model randomly initialized while training
        # proceeds, with only an INFO line to distinguish it from a real load.
        if n_model and n_loaded == 0:
            raise RuntimeError(
                f"Checkpoint load matched 0 of {n_model} model parameters "
                f"(keywords={keywords!r}, replacement={replacement!r}). "
                f"The model would train "
                f"from random init. Fix the keyword/replacement, or set strict=True "
                f"to find the mismatch."
            )

    def resume_training_state(self, checkpoint):
        """Restore structured or legacy optimizer, scheduler, RNG, and cursor state."""
        strict_state = self.trainer.cfg.get("resume_strict_state", True)
        iter_per_epoch = len(self.trainer.train_loader)

        train_state = checkpoint_train_state(checkpoint)
        if train_state is not None:
            dataloader_state = checkpoint_dataloader_state(checkpoint, train_state)
            train_state.dataloader_state = dataloader_state
            # Decide whether to drop the torchdata StatefulDataLoader cursor BEFORE
            # extracting it. That cursor asserts (lazily, on the first __iter__) on
            # ANY change to the world_size or num_workers it was saved with, and
            # local_object_state() below would itself raise on a world_size change
            # under strict resume. So when the world_size or num_workers changed
            # (or the resume is explicitly non-strict) we drop the cursor and
            # restart the resumed epoch from its first batch -- model / optimizer /
            # scheduler / global_step still restore below, so at most a sub-epoch of
            # data order is replayed. This makes resharding across GPU/worker counts
            # automatic without requiring resume_strict_state=False.
            skip_cursor, reason = False, ""
            if not strict_state:
                skip_cursor, reason = True, "resume_strict_state=False"
            elif isinstance(dataloader_state, dict) and dataloader_state.get(
                "_pimm_distributed_state"
            ):
                saved_ws = int(
                    dataloader_state.get(
                        "world_size", len(dataloader_state.get("states", []))
                    )
                )
                cur_ws = comm.get_world_size()
                if saved_ws != cur_ws:
                    skip_cursor, reason = True, f"world_size {saved_ws}->{cur_ws}"
            if not skip_cursor:
                saved_dl = checkpoint.get("dataloader", {})
                saved_workers = saved_dl.get("num_workers") if isinstance(saved_dl, dict) else None
                cur_workers = self.trainer.cfg.get("num_worker_per_gpu", None)
                if (
                    saved_workers is not None
                    and cur_workers is not None
                    and int(saved_workers) != int(cur_workers)
                ):
                    skip_cursor, reason = True, f"num_workers {saved_workers}->{cur_workers}"
            # Extract this rank's cursor. When we are going to drop it anyway, load
            # non-strictly so a world_size change does not raise here.
            local_dataloader_state = local_object_state(
                dataloader_state,
                strict=strict_state and not skip_cursor,
            )
            if skip_cursor and local_dataloader_state:
                self.trainer.logger.warning(
                    f"Skipping dataloader-cursor restore ({reason}); restarting the "
                    "resumed epoch from its first batch."
                )
                local_dataloader_state = None
            apply_train_state_to_trainer(self.trainer, train_state)
            if train_state.iter_in_epoch > 0:
                if not local_dataloader_state:
                    self.trainer.logger.warning(
                        "Checkpoint is mid-epoch but has no dataloader state; "
                        "resuming from the beginning of the saved epoch and "
                        "replaying already-completed batches."
                    )
                    self.trainer.start_iter = 0
                    self.trainer.global_step = self.trainer.start_epoch * iter_per_epoch
                else:
                    load_dataloader_state_dict(
                        self.trainer.train_loader,
                        local_dataloader_state,
                        strict=strict_state,
                    )
            rng_state = checkpoint_rng_state(checkpoint, train_state)
            restore_distributed_rng_state(rng_state, strict=strict_state)
            self.trainer.logger.info(
                "Resuming train from structured state: "
                f"epoch={self.trainer.start_epoch}, "
                f"iter={self.trainer.start_iter}, "
                f"global_step={self.trainer.global_step}"
            )
        else:
            self._resume_legacy_training_state(checkpoint, iter_per_epoch)

        checkpoint_trainer_state = checkpoint.get("trainer", {})
        if (
            isinstance(checkpoint_trainer_state, dict)
            and "best_metric_value" in checkpoint_trainer_state
        ):
            self.trainer.best_metric_value = checkpoint_trainer_state["best_metric_value"]
        else:
            self.trainer.best_metric_value = checkpoint.get(
                "best_metric_value", self.trainer.best_metric_value
            )

        optimizer_state = checkpoint_optimizer_state_dict(checkpoint)
        if optimizer_state is not None:
            self.load_optimizer_state(optimizer_state)
        else:
            self.trainer.logger.info("No optimizer state found in checkpoint.")
        scheduler_state = checkpoint_scheduler_state_dict(checkpoint)
        if scheduler_state is not None:
            self.trainer.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint_scaler_state_dict(checkpoint)
        if self.trainer.cfg.enable_amp and scaler_state is not None:
            self.trainer.scaler.load_state_dict(scaler_state)

    def _resume_legacy_training_state(self, checkpoint, iter_per_epoch):
        """Translate legacy epoch/iter fields into current trainer cursors."""
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_iter = int(checkpoint.get("iter", 0) or 0)
        self.trainer.logger.info(
            f"Resuming train at saved epoch: {checkpoint_epoch}, saved iteration: {checkpoint_iter}"
        )
        if 0 < checkpoint_iter < iter_per_epoch:
            self.trainer.logger.warning(
                "Legacy checkpoint is mid-epoch and has no dataloader state; "
                "resuming from the beginning of the saved epoch and replaying "
                "already-completed batches."
            )
            self.trainer.start_epoch = max(0, checkpoint_epoch - 1)
            self.trainer.start_iter = 0
        elif checkpoint_iter >= iter_per_epoch:
            self.trainer.start_epoch = checkpoint_epoch
            self.trainer.start_iter = 0
        else:
            self.trainer.start_epoch = checkpoint_epoch
            self.trainer.start_iter = 0
        self.trainer.global_step = (
            self.trainer.start_epoch * iter_per_epoch
            + self.trainer.start_iter
        )
        self.trainer.logger.info(
            "Resuming train at epoch index: "
            f"{self.trainer.start_epoch}, iteration: {self.trainer.start_iter}"
        )

    def load_optimizer_state(self, optimizer_state):
        """Load canonical optimizer state and fail if moments are not restored."""
        set_optimizer_state_dict(
            self.trainer.model,
            self.trainer.optimizer,
            optimizer_state,
            options=StateDictOptions(),
        )
        if optimizer_state.get("state") and not self.trainer.optimizer.state_dict().get("state"):
            raise RuntimeError(
                "Optimizer checkpoint contained state tensors, but optimizer resume "
                "left no optimizer state. Exact resume would restart optimizer moments."
            )
