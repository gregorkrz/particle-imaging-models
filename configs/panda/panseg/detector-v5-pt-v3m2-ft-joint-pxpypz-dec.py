"""Joint detector-v5 (px/py/pz) DECODER-ONLY fine-tune -- freeze ONLY the encoder.

Middle ground between the full fine-tune (-fft, trains everything) and the
linear probe (-lin, freezes encoder + decoder + seg heads, trains only the
physics regression heads):

  - FREEZE the pretrained PTv3 encoder (backbone.freeze_encoder=True).
  - TRAIN the query decoder (transformer blocks, learned query bank, point
    projection) AND all heads -- both the segmentation/PID heads and the
    per-query physics regression heads (momentum, momentum_vec = px/py/pz,
    vertex, is_primary).

Unlike -lin, the segmentation losses are KEPT ON (inherited unchanged from the
fft base). This matters: the query<->truth matcher costs (cost_mask/cost_dice/
cost_class) define the per-instance targets the physics-head losses depend on,
so the mask/class predictions must stay valid. Training the decoder with the seg
losses on keeps the masks healthy; zeroing them (as -lin does) would let the
decoder drift and corrupt the matching. Letting the decoder train also frees the
per-query features to reorganize for momentum/vertex regression instead of being
locked into a seg-only representation.

Model selection stays on the base detection metric (det F1) since segmentation
is still supervised. Data, transforms (incl. RandomRotate/RandomFlip), and the
warm-start checkpoint are inherited from the fft base.
"""

_base_ = ["./detector-v5-pt-v3m2-ft-joint-pxpypz-fft.py"]

# (1) Freeze ONLY the PTv3 encoder. Deep-merges into the base model; everything
# else (matcher costs, head configs, label criteria incl. their non-zero
# segmentation loss weights) is inherited unchanged.
model = dict(backbone=dict(freeze_encoder=True))

# (2) Rebuild the optimizer param groups: with the encoder frozen, schedule the
# decoder blocks only (highest LR at the last block), matching the -dec recipe.
# The default optimizer group (base_lr) covers the query bank, point projection,
# and all heads; the frozen encoder params carry requires_grad=False and are
# skipped by the group builder. Base config locals are not visible to child
# configs, so these are restated -- keep them in sync with the fft base.
lr_decay = 0.97
base_lr = 2e-4
base_wd = 0.01
dec_depth = 3  # = base model["depth"]

param_dicts = []
for b in range(dec_depth):
    exp = dec_depth - b - 1
    param_dicts.append(
        dict(
            keyword=f"decoder.blocks.{b}.",
            lr=base_lr * (lr_decay**exp),
        )
    )

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.025,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Distinguish the run in W&B from the full-finetune and linear-probe siblings.
hooks_override = {
    "WandbNamer": {"extra": "joint-pxpypz-dec"},
}
