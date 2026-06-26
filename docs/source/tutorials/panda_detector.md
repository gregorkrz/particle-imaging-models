# Panda panoptic detector

**Goal.** Move from *semantic* segmentation (label every point) to *panoptic*
segmentation (find each particle **instance**, its **mask**, and its **class**).
The [Panda Detector](https://arxiv.org/abs/2512.01324) is a Mask2Former-style
model adapted for low-energy deposits: a set of learned queries attend over
backbone features and each emits a mask + class (+ optionally momentum/vertex).

This is the advanced tutorial. Do {doc}`byo_dataset_semseg` first — it
establishes the dataset, transform, and launch patterns this one builds on.

```text
points ─▶ PT-v3m2 encoder (frozen, Sonata-pretrained) ─▶ point features
                                                              │
                          N learned queries ──cross-attn──────┤
                                  │                            │
                   per-query: pred_masks, pred_logits (PID), pred_momentum
                                  │
                  FastInstanceSegmentationLoss (Hungarian match)
                                  │
                     InstanceSegmentationEvaluator (PQ / ARI)
```

## What's different from semantic segmentation

```{list-table}
:header-rows: 1
:widths: 30 35 35

* -
  - Semantic (prev tutorial)
  - Panoptic (this tutorial)
* - Output
  - per-point class logits
  - per-**instance** mask + class
* - Labels needed
  - `segment` (per-point class)
  - `segment` **and** `instance` (per-point instance id)
* - Loss
  - CE + Lovász
  - `FastInstanceSegmentationLoss` (mask + dice + class, Hungarian-matched)
* - Evaluator
  - `SemSegEvaluator` (mIoU)
  - `InstanceSegmentationEvaluator` (PQ-style, needs **val batch size 1**)
* - Data revision
  - any
  - PILArNet **v2/v3** (carries PID + instance + momentum)
```

## 1. Data: you need instance labels

The detector supervises per-instance masks, so your dataset must emit a per-point
`instance` id alongside the per-point class `segment`. With PILArNet-M v2/v3 this
is a `Copy` in the transform; the dataset already provides `instance_particle`
and `segment_pid`:

```python
dict(type="Copy", keys_dict={"instance_particle": "instance",
                             "segment_pid": "segment"}),
...
dict(type="Collect",
     keys=("coord", "grid_coord", "segment", "instance"),
     feat_keys=("coord", "energy")),
```

:::{tip}
**Bringing your own data?** Your reader must produce a per-point `instance` array
(integers, one per particle, with a background/ignore convention) in addition to
the class label. Add any new point-aligned keys to `index_valid_keys` so they
survive subsampling transforms. See {doc}`../datasets/bring_your_own`.
:::

## 2. Start from the reference config

Copy `configs/panda/panseg/detector-v1m1-pt-v3m2-ft-pid-dec.py`. It is the
"decoder-only" fine-tune: a **frozen, Sonata-pretrained** PTv3 encoder with a
trainable detection decoder. The headline pieces:

```python
_base_ = ["../../_base_/default_runtime.py"]

batch_size = 48
val_batch_size = 1            # InstanceSegmentationEvaluator requires bs=1
clip_grad = 1.0
enable_amp = True
amp_dtype = "bfloat16"
epoch = 20

model = dict(
    type="detector-v1m1",
    num_classes=6,                       # photon, electron, muon, pion, proton, led
    query_type="learned",
    num_queries=32,
    use_stuff_head=True,
    stuff_classes=[5],                   # "led" (low-energy deposit) is stuff
    supervise_attn_mask=True,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,                   # xyz + energy
        enc_mode=True,                   # encoder-only
        freeze_encoder=True,             # <-- frozen; only the decoder trains
        enable_flash=True,
        # ... (same enc/dec dims as the semseg backbone)
    ),
    full_in_channels=1232,
    hidden_channels=256, num_heads=16, depth=3,
    criteria=[
        dict(
            type="FastInstanceSegmentationLoss",
            cost_mask=1.0, cost_dice=1.0, cost_class=1.0,
            loss_weight_focal=2.0, loss_weight_dice=5.0,
            cls_weight_matched=2.0, cls_weight_noobj=0.5,
            momentum_loss_weight=1.0,
            focal_alpha=0.25, focal_gamma=2.0,
            num_points=100_000,
            truth_label="instance",
        ),
    ],
)
```

A few things worth understanding:

- **`num_queries`** caps how many instances the model can emit per event. Set it
  comfortably above your busiest event's particle count.
- **Stuff vs things.** `stuff_classes=[5]` marks the low-energy-deposit class as
  "stuff" (segmented as a region, not counted as instances), the rest are
  "things" (counted instances) — standard panoptic terminology.
- **The loss is set-based.** `FastInstanceSegmentationLoss` does Hungarian
  matching between predicted and truth instances, then applies focal (class) +
  dice + mask costs; `truth_label="instance"` selects the supervision target.

### Layer-wise decoder LR

Because the encoder is frozen, only the decoder needs a schedule. The config
gives later decoder blocks a higher LR via `param_dicts`:

```python
base_lr, lr_decay = 2e-4, 0.97
param_dicts = [
    dict(keyword=f"decoder.blocks.{b}.", lr=base_lr * (lr_decay ** (dec_depth - b - 1)))
    for b in range(dec_depth)
]
scheduler = dict(type="OneCycleLR",
                 max_lr=[base_lr] + [g["lr"] for g in param_dicts],
                 pct_start=0.025, anneal_strategy="cos",
                 div_factor=10.0, final_div_factor=1000.0)
```

## 3. Warm-start the encoder from Sonata

The whole point of the `-ft-...-dec` variant is to reuse a self-supervised
backbone. The {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` hook loads pretrained weights and remaps the
keys from the SSL checkpoint's `student.backbone.*` namespace to the detector's
`backbone.*`:

```python
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.backbone",
),
```

Then point `--train.weight` at the pretrained checkpoint:

```bash
pimm submit --site s3df \
  --train.config panda/panseg/detector-v1m1-pt-v3m2-ft-pid-dec \
  --train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth
```

:::{important}
A remap matching **zero** parameters raises — so a silent random-init can't
happen. After launch, confirm the load reported no missing backbone keys, and
judge success by the decoder losses (`loss_cls`, `dice`) trending down, not just
the absence of errors. See {doc}`../hpc/resuming`.
:::

### The three fine-tuning variants

The config family encodes a common ablation; pick by filename suffix:

```{list-table}
:header-rows: 1
:widths: 18 40 42

* - Suffix
  - Encoder
  - Use when
* - `-dec`
  - **frozen**, decoder-only training
  - cheapest; tests how good the SSL features already are
* - `-fft`
  - **full fine-tune** (encoder unfrozen)
  - best accuracy when you have the compute
* - `-scratch`
  - random init, no warm-start
  - the from-scratch baseline (a clean control with no SSL leakage)
```

## 4. Hooks for detection

The detector run uses a richer hook stack than semantic segmentation. The
important additions (evaluator **before** saver, as always):

```python
hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len", "amp_dtype", "seed"), extra="dec"),
    dict(type="WeightDecayExclusion", exclude_bias_from_wd=True, exclude_norm_from_wd=True),
    dict(type="CheckpointLoader", keywords="module.student.backbone", replacement="module.backbone"),
    dict(type="ParameterCounter", show_details=True),       # see frozen vs trainable split
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InstanceSegmentationEvaluator",
         every_n_steps=1000, stuff_threshold=0.5, mask_threshold=0.5,
         stuff_classes=[5], iou_thresh=0.5,
         class_names=["photon", "electron", "muon", "pion", "proton"]),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="AttentionMaskAnnealingHook", log_frequency=100, prefix="anneal"),
    dict(type="FinalEvaluator", test_last=True),
]

test = dict(type="InstanceSegTester", stuff_classes=[5],
            class_names=["photon", "electron", "muon", "pion", "proton"])
```

{py:class}`~pimm.engines.hooks.eval.instance_segmentation.InstanceSegmentationEvaluator` reports detection/class statistics, ARI, and
(when present) momentum-regression metrics. It **requires validation batch size
1** — hence `val_batch_size = 1`. See {doc}`../evaluation/index` and
{doc}`../hooks/diagnostics` (for {py:class}`~pimm.engines.hooks.diagnostics.ParameterCounter`, {py:class}`~pimm.engines.hooks.diagnostics.AttentionMaskAnnealingHook`).

## 5. Quick check, then train

```bash
# smoke
pimm launch --train.config panda/panseg/detector-v1m1-pt-v3m2-ft-pid-dec \
  --run.name det-smoke \
  -- epoch=1 data.train.max_len=64 data.val.max_len=16 \
     batch_size=4 num_worker=0 use_wandb=False

# real run, 4 GPUs
pimm launch --train.config panda/panseg/detector-v1m1-pt-v3m2-ft-pid-dec \
  --resources.nproc-per-node 4 \
  --train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth
```

## 6. On HPC with requeue chaining

Detector runs are longer; on a walltime-limited queue, chain them so an attempt
that times out resumes from the latest complete checkpoint:

```bash
pimm submit --site s3df --recipe launch/runs/ft_sphenix_panoptic_pid.yaml \
  --train.config panda/panseg/detector-v1m1-pt-v3m2-ft-pid-dec \
  --chain.jobs 4 --resources.time 02:00:00 \
  --run.name det-pid-chain
```

Attempt 1 starts (warm-started); attempts 2+ resume automatically. See
{doc}`../hpc/chaining`. Always `--dry-run` first to confirm the rendered
resources, account, and partition.

## 7. Inference

Load the trained detector and feed it data in the **same** format (instance
labels aren't needed at inference, but the coord/energy transform must match):

```python
import pimm, torch
model = pimm.from_pretrained("exports/panda-detector", device="cuda")
with torch.no_grad():
    out = model(batch, return_point=True)   # detectors expect return_point=True
masks  = out["pred_masks"]       # per-query masks
logits = out["pred_logits"]      # per-query class logits
```

See {doc}`../models/index` and {doc}`../models/dataset_format`.

## Recap

You've gone from per-point labels to per-instance masks + PID by:

1. supplying `instance` labels in the dataset,
2. swapping in the `detector-v1m1` model with `FastInstanceSegmentationLoss`,
3. warm-starting a frozen Sonata encoder via a `CheckpointLoader` remap,
4. evaluating with `InstanceSegmentationEvaluator` at val batch size 1, and
5. scaling out with requeue chaining.

## See also

- {doc}`byo_dataset_semseg` — the foundation this builds on.
- {doc}`../reference/model_zoo` — other detector variants (`detector-v1m2`,
  `detector-v3*`, `detector-v4`).
- {doc}`../evaluation/index` — panoptic metrics in depth.
