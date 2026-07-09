# Panda panoptic detector

**Goal.** Move from *semantic* segmentation (label every point) to *panoptic*
segmentation (find each particle **instance**, its **mask**, and its **class**).
The [Panda Detector](https://arxiv.org/abs/2512.01324) is a Mask2Former-style
model adapted for low-energy deposits: a set of learned queries attend over
backbone features and each emits a mask + class (+ optionally momentum/vertex).

This assumes you've read {doc}`byo_dataset_semseg` first or understand how
datasets, transforms, and launch patterns in pimm work.

```text
points ─▶ PT-v3m2 encoder (frozen, Sonata-pretrained) ─▶ point features
                                                              │
                          N learned queries ──cross-attn──────┤
                                  │                            │
                   per-query: pred_masks, pred_logits (PID)
                                  │
                  FastUnifiedInstanceLoss (Hungarian match)
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
  - `FastUnifiedInstanceLoss` (mask + dice + class, Hungarian-matched)
* - Evaluator
  - `SemSegEvaluator` (mIoU)
  - `InstanceSegmentationEvaluator` (PQ-style, needs **val batch size 1**)
* - Data revision
  - any
  - PILArNet **v2/v3** (per-point PID + instance ids)
```

## 1. Data: you need instance labels

The detector supervises per-instance masks, so your dataset must provide a
per-point `instance` id alongside the per-point class. `detector-v4` reads them
from the keys named in its `label_configs` (here `instance_particle` and
`segment_pid`), which PILArNet-M v2 already provides - so you just `Collect` them
(no `Copy` to rename):

```python
dict(type="Collect",
     keys=("coord", "grid_coord", "segment_pid", "instance_particle"),
     feat_keys=("coord", "energy")),
```

:::{tip}
**Bringing your own data?** Your reader must produce a per-point `instance` array
(integers, one per particle, with a background/ignore convention) in addition to
the class label. Add any new point-aligned keys to `index_valid_keys` so they
survive subsampling transforms. See {doc}`../research_ecosystem/contributing_a_dataset`.
:::

## 2. Start from the reference config

Copy `configs/panda/panseg/detector-v4-pt-v3m2-ft-pid-dec.py`. It is the
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
    type="detector-v4",
    labels=("particle",),
    # which batch keys hold the instance ids and the per-point class:
    label_configs=dict(
        particle=dict(instance_key="instance_particle", segment_key="segment_pid"),
    ),
    num_classes=6,                       # photon, electron, muon, pion, proton, led
    query_type="learned",
    num_queries=32,
    use_stuff_head=True,
    stuff_classes=[5],                   # "led" (low-energy deposit) is stuff
    supervise_attn_mask=True,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,                   # xyz + energy
        enc_mode=True,
        freeze_encoder=True,             # <-- frozen; only the decoder trains
        enable_flash=True,
        # ... (same backbone dims as the semseg config)
    ),
    full_in_channels=1232,
    mlp_point_proj=True,
    hidden_channels=256, num_heads=16, depth=3,
    criteria=[
        dict(
            type="FastUnifiedInstanceLoss",
            cost_mask=1.0, cost_dice=1.0, cost_class=1.0,
            loss_weight_focal=2.0, loss_weight_dice=5.0,
            cls_weight_matched=2.0, cls_weight_noobj=0.5,
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
  "things" (counted instances) - standard panoptic terminology.
- **The loss is set-based.** `FastUnifiedInstanceLoss` does Hungarian
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

## 3. Fine-tune the encoder from Sonata

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
pimm submit --site mycluster \
  --train.config panda/panseg/detector-v4-pt-v3m2-ft-pid-dec \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

:::{note}
`hf://<your-org>/sonata-pilarnet-L/model_best.pth` is a placeholder for **your**
Sonata SSL checkpoint - its `student.backbone.*` keys are what the
`CheckpointLoader` remap below requires. Produce one with a `configs/panda/pretrain/`
recipe, or run the `-scratch` variant. Released task detectors for *inference*
(loaded with `from_pretrained`) are on the Hub - see {doc}`../research_ecosystem/using_trained_models`.
:::

:::{important}
A remap matching **zero** parameters raises - so a silent random-init can't
happen. After launch, confirm the load reported no missing backbone keys, and
judge success by the decoder losses (`loss_cls`, `dice`) trending down, not just
the absence of errors. See {doc}`../checkpoints/saving_and_loading`.
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
  - random init, no pretraining
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
1** - hence `val_batch_size = 1`. See {doc}`../evaluation/index` and
{doc}`../hooks/diagnostics` (for {py:class}`~pimm.engines.hooks.diagnostics.ParameterCounter`, {py:class}`~pimm.engines.hooks.diagnostics.AttentionMaskAnnealingHook`).

## 5. Quick check, then train

```bash
# quick check
pimm launch --train.config panda/panseg/detector-v4-pt-v3m2-ft-pid-dec \
  --run.name det-quickcheck \
  -- epoch=1 data.train.max_len=64 data.val.max_len=16 \
     batch_size=4 num_worker=0 use_wandb=False

# real run, 4 GPUs
pimm launch --train.config panda/panseg/detector-v4-pt-v3m2-ft-pid-dec \
  --resources.nproc-per-node 4 \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

## 6. On HPC with requeue chaining

Detector runs are longer; on a walltime-limited queue, chain them so an attempt
that times out resumes from the latest complete checkpoint:

```bash
pimm submit --site mycluster --recipe launch/runs/my_finetune.yaml \
  --train.config panda/panseg/detector-v4-pt-v3m2-ft-pid-dec \
  --chain.jobs 4 --resources.time 02:00:00 \
  --run.name det-pid-chain
```

Attempt 1 starts (fine-tuned); attempts 2+ resume automatically. See
{doc}`../hpc/chaining`. Always `--dry-run` first to confirm the rendered
resources, account, and partition.

## 7. Inference

Load the trained detector and turn raw events into **per-point** instances, PIDs,
and scores. Instance labels aren't needed at inference, but the coord/energy
transform must match training - and the detector needs `postprocess()` to turn
per-query masks into per-point predictions. (`forward` alone returns raw query
tensors, which is the step most people stop at by mistake.)

```python
import torch
import pimm
from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn
from pimm.models.utils.misc import offset2bincount

# Load the released particle detector (or your own `pimm export` directory).
model = pimm.from_pretrained("deeplearnphysics/panda-particle", device="cuda")

# Same transform as the config's test split, but WITHOUT the label steps:
# no `Copy`, and no segment/instance in `Collect`.
pipeline = Compose([
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train",
         return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=("coord", "energy")),
])

# one raw event: coord (N, 3) float32, energy (N, 1) float32
batch = collate_fn([pipeline({"coord": coord, "energy": energy})])
batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

with torch.no_grad():
    out = model(batch, return_point=True)        # return_point=True is required
point = out["point"]

# Turn per-query masks/logits into per-point predictions - the same call the
# InstanceSegmentationEvaluator makes internally.
preds = model.postprocess({
    "pred_masks":   out["pred_masks"],
    "pred_logits":  out["pred_logits"],
    "stuff_probs":  point.outputs.get("stuff_probs"),
    "point_counts": offset2bincount(point.offset),
}, stuff_threshold=0.5, mask_threshold=0.5)

# All per-point, row-aligned to batch["coord"]:
instance_id = preds["instance_labels"]   # int; -1 = stuff / uncovered
pid_class   = preds["class_labels"]      # 0..5 → photon, electron, muon, pion, proton, led
score       = preds["confidences"]       # per-point confidence
```

:::{note}
`deeplearnphysics/panda-particle` is the released particle detector (a
`detector-v4` export); swap in your own `pimm export` directory to run a model you
trained. `postprocess()` takes more knobs (`conf_threshold`, NMS, `min_points`, …)
that otherwise default to the model's `postprocess_cfg`.
:::

See {doc}`../research_ecosystem/using_trained_models` and {doc}`../datasets/transforms`.

## Recap

You've gone from per-point labels to per-instance masks + PID by:

1. supplying `instance` labels in the dataset,
2. swapping in the `detector-v4` model with `FastUnifiedInstanceLoss`,
3. fine-tuning a frozen Sonata encoder via a `CheckpointLoader` remap,
4. evaluating with `InstanceSegmentationEvaluator` at val batch size 1, and
5. scaling out with requeue chaining.

## See also

- {doc}`byo_dataset_semseg` - the foundation this builds on.
- {doc}`../research_ecosystem/using_trained_models` - the published `detector-v4` checkpoints.
- {doc}`../evaluation/index` - panoptic metrics in depth.
