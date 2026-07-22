"""Joint detector-v5 (px/py/pz) LINEAR PROBE -- train ONLY the physics heads.

Copy of detector-v5-pt-v3m2-ft-joint-pxpypz-fft that FREEZES the whole
pretrained feature extractor -- the PTv3 backbone encoder AND the query decoder
(its transformer blocks, learned query bank, point projection) AND the
segmentation heads (mask / PID classifier / stuff) -- and trains ONLY the
per-query physics regression heads (decoder.head_by_key): momentum,
momentum_vec = px/py/pz, vertex, is_primary.

This differs from the fft baseline in three ways:

  1. Freeze the trunk (config-only):
     - backbone.freeze_encoder=True -> requires_grad=False on the PTv3 encoder.
     - The optimizer's default group (group 0: everything NOT matched by the
       head keyword below -- decoder blocks, query bank, cls/stuff heads, point
       projection) runs at lr=0, so AdamW makes no update / no decoupled weight
       decay to it. Only decoder.head_by_key trains, at base_lr.
     NOTE: the frozen-by-lr=0 modules keep requires_grad=True, so they still show
     as "trainable" in ParameterCounter and compute (discarded) grads. A hard
     freeze would need a code change; lr=0 is the config-only equivalent.

  2. No segmentation objective ("not segmentation anymore"): the mask (focal +
     dice), PID-class, and aux loss weights are zeroed in both label criteria, so
     the training loss is purely the physics-head regression losses. The matcher
     COSTS (cost_mask/cost_dice/cost_class) are kept, so query<->truth matching --
     which the head losses need to define their per-instance targets -- still
     works off the frozen mask/class predictions. (The stuff-head loss is added
     by the model outside the criterion and cannot be zeroed here; it is harmless
     because the stuff head is frozen and it does not enter the selection metric.)

  3. Best model by LOWEST validation head loss (not det F1): the evaluator logs
     each head's validation loss + a total (val/head_loss_total) and publishes
     that total as the lower-is-better model_best metric (select_metric=
     val_head_loss). Segmentation metrics are still logged but no longer select.

Everything else -- data, transforms, warm-start checkpoint -- is inherited from
the fft base. This is the linear-probe sibling of the decoder-only (-dec) recipe,
which freezes only the encoder.

To also train the PID / stuff heads (freeze only the trunk, not the seg heads),
add "decoder.cls_pred_by_label" / "decoder.stuff_head_by_label" to HEAD_KEYWORDS
and drop the criterion loss-weight zeroing below.
"""

_base_ = ["./detector-v5-pt-v3m2-ft-joint-pxpypz-fft.py"]

# (1a) Hard-freeze the PTv3 encoder, and (2) zero the segmentation loss weights
# for both labels so only the physics-head regression losses train. Deep-merges
# into the base model: the matcher costs, head configs, and everything else are
# kept; only these fields change.
_ZERO_SEG_LOSS = dict(
    loss_weight_focal=0.0,
    loss_weight_dice=0.0,
    cls_weight_matched=0.0,
    cls_weight_noobj=0.0,
    aux_loss_weight=0.0,
)
model = dict(
    backbone=dict(freeze_encoder=True),
    label_configs=dict(
        particle=dict(criterion=_ZERO_SEG_LOSS),
        interaction=dict(criterion=_ZERO_SEG_LOSS),
    ),
)

# Heads-only probe: a higher LR than the full-finetune run is fine since only a
# few million head params train. Tune as needed.
base_lr = 1e-3
base_wd = 0.01

# (1b) Only the physics regression heads (decoder.head_by_key -> momentum,
# momentum_vec, vertex, is_primary across both labels) are trained; everything
# else stays in the lr=0 default group and is thus frozen.
HEAD_KEYWORDS = ["decoder.head_by_key"]
param_dicts = [dict(keyword=keyword, lr=base_lr) for keyword in HEAD_KEYWORDS]

# Group 0 (the frozen trunk) is driven by the optimizer's base lr -> 0; the head
# groups above train at base_lr. OneCycleLR max_lr lists one value per optimizer
# group (group 0 first, then the head groups).
optimizer = dict(type="AdamW", lr=0.0, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0] + [group["lr"] for group in param_dicts],
    pct_start=0.025,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# (3) Select model_best on the lowest total validation head loss, and log the
# per-head validation losses. Also tag the W&B run.
hooks_override = {
    "WandbNamer": {"extra": "joint-pxpypz-lin"},
    "InstanceSegmentationEvaluator": {
        "select_metric": "val_head_loss",
        "log_val_loss": True,
    },
}
