# Using & fine-tuning models

Once a model is trained and {doc}`exported <../checkpoints/exporting>`, you load it
for inference or fine-tuning with **one function**: {py:func}`~pimm.from_pretrained`.

To publish your own trained weights, see {doc}`publishing_a_model`.

- **Load a model** - local dir, Hub repo, `hf://` URI, or raw checkpoint (see below).
- [Feed it the right data](../datasets/transforms.md#reproducing-the-pipeline-at-inference) - reproduce the exact transform + packed-batch format.
- {doc}`Contributing a model <contributing_a_model>` - register a new architecture on the pimm substrate.

:::{seealso}
**Sharing** is the other half of this loop and lives in the Checkpoints section:
{doc}`export portable weights <../checkpoints/exporting>` with `pimm export` /
{py:func}`~pimm.export.save_pretrained`, and {doc}`push them to the Hub
<../checkpoints/huggingface>` (manually or with the `PushToHub` hook). This page
assumes you already have an export or checkpoint in hand.
:::

## Published checkpoints

The DeepLearnPhysics org publishes ready-to-load checkpoints on the Hugging Face
Hub as **consolidated exports** (`model.safetensors` + `config.json`), so the
architecture travels with the weights - {py:func}`~pimm.from_pretrained` rebuilds
the model with no config needed.

```{list-table}
:header-rows: 1
:widths: 38 40 22

* - Repo
  - Task
  - `model.type`
* - [`deeplearnphysics/panda-base`](https://huggingface.co/deeplearnphysics/panda-base)
  - Pretrained PT-v3m2 encoder - fine-tune downstream tasks from this
  - `PT-v3m2`
* - [`deeplearnphysics/panda-semantic`](https://huggingface.co/deeplearnphysics/panda-semantic)
  - Semantic segmentation - 5 classes (shower, track, michel, delta, led)
  - `DefaultSegmentorV2`
* - [`deeplearnphysics/panda-particle`](https://huggingface.co/deeplearnphysics/panda-particle)
  - Panoptic **particle ID** - 6 classes (photon, electron, muon, pion, proton, led)
  - `detector-v4`
* - [`deeplearnphysics/panda-interaction`](https://huggingface.co/deeplearnphysics/panda-interaction)
  - Panoptic **interaction / vertex** grouping - 2 classes
  - `detector-v4`
* - [`deeplearnphysics/polar-mae-base`](https://huggingface.co/deeplearnphysics/polar-mae-base)
  - Pretrained PoLAr-MAE (ViT) encoder - fine-tune downstream tasks from this
  - `PoLAr-MAE`
* - [`deeplearnphysics/polar-mae-semantic`](https://huggingface.co/deeplearnphysics/polar-mae-semantic)
  - Semantic segmentation (4 classes) on a PoLAr-MAE encoder
  - `PoLArMAE-SemSeg`
```

Load any of these for inference with a single call (build the input and read the
output as shown [below](#running-inference); the Panda detectors also need
`postprocess()`, shown in {doc}`../tutorials/panda_detector`):

```python
import pimm
model = pimm.from_pretrained("deeplearnphysics/panda-semantic", device="cuda")
```

The `*-base` encoders are the starting point for fine-tuning - copy a config, load
the encoder via a `CheckpointLoader` remap, and train (see
{doc}`../tutorials/byo_dataset_semseg` and {doc}`../tutorials/panda_detector`).

:::{note}
A bare `hf://<repo>` resolves to the repo's `model.safetensors`; the explicit form
is `hf://<repo>/<file>`. A fine-tune config's `CheckpointLoader` rewrites a
checkpoint's keys onto the model's `backbone.*`, so the remap rule must match the
checkpoint's key layout - confirm the load reports no missing backbone keys.
:::

## Load with `from_pretrained`

```python
import pimm

model = pimm.from_pretrained("org-or-user/my-model", device="cuda")  # Hub repo
model = pimm.from_pretrained("exports/my-model")                     # local export dir
model = pimm.from_pretrained("hf://org-or-user/my-model@v2")         # hf:// + revision
```

The argument may be a local exported directory, a Hugging Face Hub repo id, an
`hf://` URI (same scheme as a training `weight=`), or a local checkpoint file.
The returned model is in **eval mode**.

For an exported directory or repo, `from_pretrained` rebuilds the architecture
from the export's config and loads the weights.
Config precedence is: explicit `model_config` > the export's
`config.json` > `config_path` / run dir.

:::{tip}
**Config drift is tolerated.** Constructor kwargs that the current code no longer
accepts are dropped (with a warning) and construction is retried - so older
exports keep loading for free as the model code evolves.
:::

### From a raw checkpoint

A raw training checkpoint has no embedded architecture, so supply one via
`model_config` or `config_path` (or place it in an experiment layout where the
config can be inferred):

```python
model = pimm.from_pretrained(
    "exp/my-run/model/model_last.pth",
    config_path="exp/my-run/config.py",
    strict=False,
    device="cuda",
)
```

Extra keyword args are merged into the model config before construction:

```python
model = pimm.from_pretrained("exports/my-model", num_classes=7)
```

`return_metadata=True` returns `(model, metadata)`; `strict` is forwarded to
`model.load_state_dict`. Hub loads download to `PIMM_HF_CACHE` (else HF's
`HF_HOME`) and require the optional `huggingface_hub` package.

## Running inference

Build an input the way the dataloader would (see [reproducing the pipeline at
inference](../datasets/transforms.md#reproducing-the-pipeline-at-inference) - this is
the part people get wrong), move it to the device, and call the model. There is
no `make_packed_batch` helper: reproduce the model's **val** transform with
{py:class}`~pimm.datasets.transform.base.Compose` and collate the result with
`collate_fn`. Models accept the packed batch dict and return an
output dict:

```python
import torch
from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn

model = pimm.from_pretrained("exports/my-semseg-model", device="cuda")

# Same pipeline the config's val/test split uses (numbers here are PILArNet-M's).
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
    out = model(batch)

# Conventional output keys (present depends on the model/task):
logits = out.get("seg_logits", out.get("sem_logits"))   # semantic segmentation
pred = logits.argmax(1)                                  # per-point class id
# out["cls_logits"]                                       # classification
# out["pred_masks"], out["pred_logits"]                   # detector / panoptic (see the Panda tutorial)
```

The output-key conventions (the same ones evaluators consume):

```{list-table}
:header-rows: 1
:widths: 30 70

* - Key
  - Meaning
* - `seg_logits` / `sem_logits`
  - per-point semantic logits (testers look for `seg_logits` first)
* - `cls_logits`
  - classification logits
* - `point`
  - output `Point` (when called with `return_point=True`)
* - `pred_logits`, `pred_masks`, `pred_momentum`
  - detector / instance-segmentation outputs
* - `loss`, `total_loss`
  - present in *training* mode; not needed for inference
```

:::{note}
Some models expose extra methods - `predict`, `encode`, `forward_features`,
`postprocess`, `update_anneal_step` - duck-typed by specific scripts. These are
not a global rule; check the model family you're loading.
:::

## Fine-tune from a checkpoint

To start a *new training run* from pretrained weights, you don't use
`from_pretrained` - you point the training config's `weight=` at the checkpoint
(no `--train.resume`, so only weights load):

```bash
pimm submit --site mycluster \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

:::{note}
`hf://<your-org>/sonata-pilarnet-L/model_best.pth` is a placeholder for **your**
Sonata SSL checkpoint (push one with the {doc}`../checkpoints/huggingface` hook) -
its `student.backbone.*` keys are what the fine-tune configs' `CheckpointLoader`
remap requires. Published task checkpoints for *inference* (loaded with
`from_pretrained`) are in [Published checkpoints](#published-checkpoints) above.
:::

When the checkpoint's keys don't line up with the fine-tune model, remap them
with the {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` hook (`keywords` → `replacement`). Full mechanics -
including the "a remap matching zero params raises" guard - are in
{doc}`../checkpoints/saving_and_loading`.

For a parameter-efficient alternative that trains only injected LoRA weights with the backbone frozen, see {doc}`../tutorials/panda_detector_peft`.

### Partial / programmatic loading

To load only a submodule (e.g. just the backbone) into an already-built model, or
to remap keys by hand, use the lower-level {py:func}`~pimm.export.load_pretrained`
helper and the `pimm.export` state-dict utilities. The full set, with examples,
is in {doc}`../checkpoints/saving_and_loading`.

## Forward and loss requirements (for reference)

If you're loading a model to keep training it, remember what the trainer requires: the
outer model's `forward(input_dict)` must return a dict with a scalar `loss`
during training. Internal modules/backbones can return `Point`, tensors, or
tuples. See {doc}`../getting_started/concepts`.
