"""Optional hook: push model checkpoints to the Hugging Face Hub during/after training.

This hook is entirely opt-in: it only runs when explicitly added to a config's
``hooks`` list, e.g.::

    hooks = [
        ...,
        dict(type="CheckpointSaver"),
        dict(type="PushToHub", repo_id="youngsm/sonata-pilarnet-L", private=True),
    ]

By default it uploads the raw ``model_best`` checkpoint at the end of training
(``weights_only=True``), so the result loads back byte-identically via
``weight=hf://<repo>/model_best.pth``. Set ``weights_only=False`` to push a
consolidated ``pimm export`` artifact instead.

Cross-cluster sync (pretrain on one site, monitor/fine-tune on another) is
supported via periodic pushes:

- ``on_best=True``   -> push ``model_best`` whenever the best metric improves
- ``every_n_epochs`` -> push ``model_last`` at the end of every N epochs

Periodic uploads run in a background thread (``background=True``) so they never
block the training step. Place ``PushToHub`` *after* the checkpoint saver in the
``hooks`` list so the files it reads are already written for the current step.

Caveat: uploading large checkpoints repeatedly to the same path accumulates blobs
in the repo's LFS history. For frequent periodic pushes, prefer a dedicated repo
and/or a long ``every_n_epochs`` cadence.

Failures are logged but never crash the run.
"""

from __future__ import annotations

import os

from pimm.utils.comm import is_main_process
from pimm.utils.path import (
    latest_complete_checkpoint,
    resolve_model_weight_file,
    split_checkpoint_weight_file,
)
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class PushToHub(HookBase):
    """Push model checkpoints to the Hugging Face Hub during and/or after training.

    Entirely opt-in. By default it uploads the raw ``model_best`` checkpoint once
    at the end of training (``weights_only=True``), so it loads back
    byte-identically via ``weight=hf://<repo>/model_best.pth``; set
    ``weights_only=False`` to push a consolidated ``pimm export`` artifact
    instead. For cross-cluster sync it can also push periodically: with
    ``on_best=True`` it pushes ``model_best`` in ``after_step`` whenever the best
    metric improves, and with ``every_n_epochs`` set it pushes the rolling
    checkpoint in ``after_epoch`` every N epochs. The final end-of-training push
    happens in ``after_train``, which also drains any in-flight background
    uploads. Registered as ``PushToHub``.

    Args:
        repo_id (str): Target Hub repo, e.g. ``"youngsm/sonata-pilarnet-L"``.
        checkpoint (str): Which checkpoint to push at end of training, e.g.
            ``"model_best"`` or ``"last"``/``"model_last"`` (the latter resolve
            to the newest rolling checkpoint). Defaults to ``"model_best"``.
        weights_only (bool): If ``True``, upload the raw checkpoint file; if
            ``False``, build and push a consolidated ``pimm export`` artifact.
            Defaults to ``True``.
        private (bool): Create the repo as private if it does not exist. Defaults
            to ``True``.
        token (str, optional): Hugging Face token; falls back to the ambient
            credential when ``None``. Defaults to ``None``.
        revision (str, optional): Target branch/revision for uploads. Defaults
            to ``None``.
        name (str, optional): Destination filename in the repo for raw uploads;
            defaults to the source file's basename when ``None``. Defaults to
            ``None``.
        on_train_end (bool): Push ``checkpoint`` in ``after_train``. Defaults to
            ``True``.
        on_best (bool): Push ``model_best`` in ``after_step`` whenever
            ``best_metric_value`` improves. Defaults to ``False``.
        every_n_epochs (int, optional): Push the rolling checkpoint in
            ``after_epoch`` every N epochs. ``None`` disables. Defaults to
            ``None``.
        background (bool): Run periodic raw uploads in a background thread so
            they never block the training step (uploads to the same path are
            serialized). The final ``after_train`` push is always blocking.
            Defaults to ``True``.

    Note:
        Uploads run on rank 0 only, and failures are logged but never crash the
        run. Place ``PushToHub`` **after** the checkpoint saver in the ``hooks``
        list so the files it reads are already written. Repeatedly uploading
        large checkpoints to the same path accumulates LFS blobs in the repo's
        history, so prefer a dedicated repo and/or a long ``every_n_epochs``
        cadence for frequent pushes.

    Example:
        Add to ``cfg.hooks`` after the checkpoint saver; by default it uploads
        ``model_best`` to the Hub once at the end of training:

        .. code-block:: python

            hooks = [
                dict(type="CheckpointSaver"),
                dict(type="PushToHub", repo_id="youngsm/sonata-pilarnet-L",
                     private=True),
            ]
            # → in after_train (rank 0, blocking) uploads
            #   <save_path>/model/model_best.pth to
            #   hf://youngsm/sonata-pilarnet-L/model_best.pth; with on_best=True it
            #   also pushes model_best in after_step whenever best_metric_value
            #   improves, and every_n_epochs=N pushes the rolling ckpt in after_epoch
    """

    def __init__(
        self,
        repo_id,
        checkpoint="model_best",
        weights_only=True,
        private=True,
        token=None,
        revision=None,
        name=None,
        on_train_end=True,
        on_best=False,
        every_n_epochs=None,
        background=True,
    ):
        """Configure the target repo, what to push, and when."""
        self.repo_id = repo_id
        self.checkpoint = checkpoint
        self.weights_only = weights_only
        self.private = private
        self.token = token
        self.revision = revision
        self.name = name
        self.on_train_end = on_train_end
        self.on_best = on_best
        self.every_n_epochs = every_n_epochs
        self.background = background
        self._api = None
        self._repo_ready = False
        self._seen_best = -float("inf")
        self._futures = {}  # path_in_repo -> in-flight future (one per path)

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def _ensure_repo(self):
        if not self._repo_ready:
            self._get_api().create_repo(
                repo_id=self.repo_id, repo_type="model",
                private=self.private, exist_ok=True,
            )
            self._repo_ready = True

    def _resolve_checkpoint_file(self, name):
        """Resolve a checkpoint name to a single local weight file.

        ``"last"``/``"model_last"`` resolve to the newest rolling checkpoint via
        ``latest_complete_checkpoint`` -- in the default ``standard``/DCP format
        that is the ``model/last/`` directory (no ``model_last.pth`` exists), so a
        plain name probe would never find it.
        """
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        if name in ("last", "model_last"):
            ckpt = latest_complete_checkpoint(model_dir)
            if ckpt is None:
                raise FileNotFoundError(f"No rolling checkpoint under {model_dir}")
            return resolve_model_weight_file(str(ckpt))
        candidates = [name]
        if not os.path.splitext(name)[1]:
            candidates += [f"{name}.pth"]
        for cand in candidates:
            path = cand if os.path.isabs(cand) else os.path.join(model_dir, cand)
            if os.path.isfile(path):
                return path
            split = split_checkpoint_weight_file(path)
            if os.path.isfile(split):
                return split
            if os.path.isdir(path):
                return resolve_model_weight_file(path)
        raise FileNotFoundError(
            f"PushToHub could not resolve checkpoint {name!r} under {model_dir}"
        )

    def _push_raw(self, weight_file, *, background):
        """Upload a single raw checkpoint file; returns the hf:// URI."""
        self._ensure_repo()
        path_in_repo = self.name or os.path.basename(weight_file)
        if background:
            # Serialize uploads to the SAME path so two in-flight commits cannot
            # race/clobber each other; keeps one future per path (bounded).
            prev = self._futures.get(path_in_repo)
            if prev is not None and not prev.done():
                prev.result()
        future = self._get_api().upload_file(
            path_or_fileobj=weight_file,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            run_as_future=background,
        )
        if background:
            self._futures[path_in_repo] = future
        return f"hf://{self.repo_id}/{path_in_repo}"

    def _push_export(self, weight_file):
        """Upload a consolidated pimm-export artifact; returns the hf:// URI."""
        import tempfile

        from pimm.export import push_to_hub, save_pretrained

        config_path = os.path.join(self.trainer.cfg.save_path, "config.py")
        with tempfile.TemporaryDirectory() as tmp:
            save_pretrained(weight_file, tmp, config_path=config_path)
            push_to_hub(
                tmp, self.repo_id, private=self.private,
                token=self.token, revision=self.revision,
            )
        return f"hf://{self.repo_id}"

    def _push(self, name, *, background, weights_only=None):
        """Resolve ``name`` and push it; best-effort, never fatal. Returns success."""
        if not is_main_process():
            return False
        weights_only = self.weights_only if weights_only is None else weights_only
        try:
            weight_file = self._resolve_checkpoint_file(name)
            if weights_only:
                uri = self._push_raw(weight_file, background=background)
            else:
                uri = self._push_export(weight_file)  # always synchronous
            verb = "queued" if (background and weights_only) else "pushed"
            self.trainer.logger.info(f"PushToHub: {verb} {weight_file} -> {self.repo_id}")
            self.trainer.logger.info(f"PushToHub: load with weight={uri}")
            return True
        except Exception as exc:  # pragma: no cover - best-effort upload
            self.trainer.logger.warning(f"PushToHub: failed to push {name}: {exc}")
            return False

    def after_step(self):
        """Push model_best when the best metric improves (cadence sync)."""
        if not self.on_best:
            return
        best = float(self.trainer.best_metric_value)
        if best > self._seen_best:
            # Only advance _seen_best on a successful push, so a transient failure
            # (e.g. model_best.pth not written yet) is retried on a later step.
            if self._push("model_best", background=self.background, weights_only=True):
                self._seen_best = best

    def after_epoch(self):
        """Push the rolling checkpoint every N epochs when configured."""
        if not self.every_n_epochs:
            return
        if (self.trainer.epoch + 1) % self.every_n_epochs == 0:
            self._push("last", background=self.background, weights_only=True)

    def after_train(self):
        """Push the configured checkpoint when training finishes (blocking)."""
        if self.on_train_end:
            self._push(self.checkpoint, background=False)
        # Drain any in-flight background uploads so the job does not exit early.
        for future in self._futures.values():
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - best-effort upload
                if is_main_process():
                    self.trainer.logger.warning(f"PushToHub: background upload failed: {exc}")
