"""
Online Linear Probe - trains a lightweight linear classifier alongside
the backbone during pretraining. Features are detached so probe gradients
never affect the backbone.

Requires:
  - segment_motif in the training data pipeline (added to view_keys + Collect)
  - segment_motif propagated through GridPooling (head_indices based)
  - Model stores _probe_feat, _probe_segment_motif, _probe_batch in model itself
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase


@HOOKS.register_module()
class OnlineLinearProbe(HookBase):
    """Train a detached point-level linear probe during pretraining steps.

    On every training step (``after_step``), reads the detached features the
    model stashed on itself (``_probe_feat`` and ``_probe_segment_motif``) and
    trains a small ``LayerNorm`` + ``Linear`` classifier with its own AdamW
    optimizer. Because the features are detached by the model, probe gradients
    never reach the backbone, giving a cheap online read-out of representation
    quality. The probe is lazily created on the first step once the feature
    dimension is known. Every ``log_frequency`` steps it logs running accuracy,
    loss, and per-class plus macro precision/recall/F1 to the writer, then
    resets the running counters. This hook does NOT set the checkpoint-selection
    ``comm_info`` keys. Registered as ``OnlineLinearProbe`` (use as ``type`` in a
    ``hooks=[...]`` entry).

    Args:
        num_classes (int): Number of probe output classes. Defaults to ``5``.
        lr (float): AdamW learning rate for the probe. Defaults to ``1e-3``.
        weight_decay (float): AdamW weight decay for the probe. Defaults to
            ``1e-6``.
        log_frequency (int): Log and reset running metrics every N probe steps.
            Defaults to ``10``.
        prefix (str): Writer metric namespace. Defaults to ``"online_probe"``.
        class_names (list | None): Names used in per-class metric tags. Defaults
            to ``None``.
        weight (Sequence[float] | None): Optional per-class cross-entropy class
            weights. Defaults to ``None``.

    Note:
        Requires the model to expose ``segment_motif`` through the pipeline and
        to store ``_probe_feat`` / ``_probe_segment_motif`` (detached) on itself;
        the hook silently no-ops on steps where these are missing. It trains on
        the live training batch each step (no separate validation pass) and does
        not drive checkpoint selection.

    Example:
        Add to ``cfg.hooks`` for SSL pretraining; on every step it trains a
        detached point-level probe on the live training batch:

        .. code-block:: python

            hooks = [
                dict(type="OnlineLinearProbe", num_classes=5,
                     log_frequency=50, prefix="online_probe"),
            ]
            # → trains a LayerNorm+Linear probe (its own AdamW) on the model's
            #   detached _probe_feat each step; every 50 steps writes
            #   online_probe/{accuracy,loss,macro_precision,macro_recall,macro_f1}
            #   (+ per-class precision/recall/f1) to the writer; does NOT drive
            #   checkpoint selection
    """

    def __init__(
        self,
        num_classes=5,
        lr=1e-3,
        weight_decay=1e-6,
        log_frequency=10,
        prefix="online_probe",
        class_names=None,
        weight=None,
    ):
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.class_names = class_names
        self.weight = weight
        self.step_count = 0
        self.probe = None
        self.probe_optimizer = None
        # Running metrics
        self._correct = 0
        self._total = 0
        self._class_correct = None
        self._class_total = None

    def before_train(self):
        # Defer probe creation to first after_step when we know feat_dim
        pass

    def _init_probe(self, feat_dim, device, dtype=None):
        """Create the probe and metric buffers on the feature device."""
        self.probe = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, self.num_classes),
        ).to(device=device, dtype=dtype)
        self.probe_optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.weight is not None:
            self._loss_weight = torch.tensor(
                self.weight, dtype=dtype, device=device
            )
        else:
            self._loss_weight = None
        self._tp = torch.zeros(self.num_classes, device=device)
        self._fp = torch.zeros(self.num_classes, device=device)
        self._fn = torch.zeros(self.num_classes, device=device)

    def _get_model(self):
        model = self.trainer.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def _get_global_step(self):
        current_epoch = self.trainer.comm_info["epoch"] + 1
        current_iter = self.trainer.comm_info["iter"]
        return (current_epoch - 1) * len(self.trainer.train_loader) + current_iter + 1

    def after_step(self):
        """Train the probe from detached features exposed by the model."""
        model = self._get_model()

        # Check if model stored probe data on itself
        feat = getattr(model, "_probe_feat", None)
        labels = getattr(model, "_probe_segment_motif", None)
        if feat is None or labels is None:
            return

        labels = labels.squeeze(-1).long()

        # Initialize probe on first call
        if self.probe is None:
            self._init_probe(feat.shape[-1], feat.device, feat.dtype)
            self.trainer.logger.info(
                f"OnlineLinearProbe initialized: feat_dim={feat.shape[-1]}, "
                f"dtype={feat.dtype}, num_points={feat.shape[0]}"
            )

        # Forward through probe (features already detached by model)
        logits = self.probe(feat)
        loss = F.cross_entropy(
            logits, labels, weight=self._loss_weight
        )

        # Backward + step (only probe parameters)
        self.probe_optimizer.zero_grad()
        loss.backward()
        self.probe_optimizer.step()

        # Accumulate TP/FP/FN per class
        preds = logits.argmax(dim=-1)
        self._correct += (preds == labels).sum().item()
        self._total += labels.shape[0]
        for c in range(self.num_classes):
            pred_c = preds == c
            label_c = labels == c
            self._tp[c] += (pred_c & label_c).sum()
            self._fp[c] += (pred_c & ~label_c).sum()
            self._fn[c] += (~pred_c & label_c).sum()

        self.step_count += 1

        # Log periodically
        if (
            self.step_count % self.log_frequency == 0
            and self.trainer.writer is not None
            and self._total > 0
        ):
            global_step = self._get_global_step()

            accuracy = self._correct / self._total
            self.trainer.writer.add_scalar(
                f"{self.prefix}/accuracy", accuracy, global_step
            )
            self.trainer.writer.add_scalar(
                f"{self.prefix}/loss", loss.item(), global_step
            )

            # Per-class precision, recall, F1
            precisions, recalls, f1s = [], [], []
            for c in range(self.num_classes):
                tp = self._tp[c].item()
                fp = self._fp[c].item()
                fn = self._fn[c].item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

                name = (
                    self.class_names[c]
                    if self.class_names and c < len(self.class_names)
                    else str(c)
                )
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/recall_{name}", recall, global_step
                )
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/precision_{name}", precision, global_step
                )
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/f1_{name}", f1, global_step
                )

            self.trainer.writer.add_scalar(
                f"{self.prefix}/macro_recall",
                sum(recalls) / len(recalls),
                global_step,
            )
            self.trainer.writer.add_scalar(
                f"{self.prefix}/macro_precision",
                sum(precisions) / len(precisions),
                global_step,
            )
            self.trainer.writer.add_scalar(
                f"{self.prefix}/macro_f1",
                sum(f1s) / len(f1s),
                global_step,
            )

            # Reset running metrics
            self._correct = 0
            self._total = 0
            self._tp.zero_()
            self._fp.zero_()
            self._fn.zero_()
