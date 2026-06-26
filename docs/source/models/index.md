# Using trained models

Once a model is trained and {doc}`exported <../checkpoints/export>`, you load it
for inference or fine-tuning with **one function**: {py:func}`~pimm.from_pretrained`. This
page covers loading from every source, reproducing the exact input the model
expects, and warm-starting a fine-tune.

- **Load a model** — local dir, Hub repo, `hf://` URI, or raw checkpoint (see below).
- {doc}`Feed it data <dataset_format>` — reproduce the exact transform + packed-batch format.

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

For an exported directory or repo, `from_pretrained` probes for the weights
file, reads the model config from `training_config.json` (`["model"]`), imports
`pimm.models`, builds the architecture, loads the state dict, and returns the
model. Config precedence is: explicit `model_config` > the export's
`training_config.json` > `config_path` / run dir.

:::{tip}
**Config drift is tolerated.** Constructor kwargs that the current code no longer
accepts are dropped (with a warning) and construction is retried — so older
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

Build an input the way the dataloader would (see {doc}`dataset_format` — this is
the part people get wrong), move it to the device, and call the model. Models
accept the packed batch dict and return an output dict:

```python
import torch

model = pimm.from_pretrained("exports/my-semseg-model", device="cuda")

batch = make_packed_batch(event)          # {"coord","feat","offset",...} — see dataset_format
batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

with torch.no_grad():
    out = model(batch)

# Conventional output keys (present depends on the model/task):
logits = out.get("seg_logits", out.get("sem_logits"))   # semantic segmentation
# out["cls_logits"]                                       # classification
# out["pred_masks"], out["pred_logits"]                   # detector / panoptic
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
Some models expose extra methods — `predict`, `encode`, `forward_features`,
`postprocess`, `update_anneal_step` — duck-typed by specific scripts. These are
not a global contract; check the model family you're loading.
:::

## Warm-start a fine-tune

To start a *new training run* from pretrained weights, you don't use
`from_pretrained` — you point the training config's `weight=` at the checkpoint
(no `--train.resume`, so only weights load):

```bash
pimm submit --site s3df \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth
```

When the checkpoint's keys don't line up with the fine-tune model, remap them
with the {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` hook (`keywords` → `replacement`). Full mechanics —
including the "a remap matching zero params raises" guard — are in
{doc}`../hpc/resuming` and {doc}`../checkpoints/hooks`.

### Partial / programmatic warm-start

To load only a submodule (e.g. just the backbone) into an already-built model,
use the lower-level helper:

```python
from pimm.models.builder import build_model
from pimm.export import load_pretrained

model = build_model(cfg.model)
load_pretrained(
    model.backbone,
    "exp/pretrain/model/model_last.pth",
    prefix="student.backbone.",   # keep keys with this prefix, then strip it
    remove_prefix=True,
    strict=False,
)
```

See {doc}`../checkpoints/export` for the full helper set
({py:func}`~pimm.export.clean_state_dict`, {py:func}`~pimm.export.filter_state_dict_by_prefix`, {py:func}`~pimm.export.remap_state_dict_keys`,
{py:func}`~pimm.export.load_state_dict_from_checkpoint`).

## The forward/loss contract (for reference)

If you're loading a model to keep training it, remember the trainer contract: the
outer model's `forward(input_dict)` must return a dict with a scalar `loss`
during training. Internal modules/backbones can return `Point`, tensors, or
tuples. See {doc}`../getting_started/concepts`.

## Next

- {doc}`dataset_format` — build the exact input a model expects.
- {doc}`../checkpoints/export` — produce the export you're loading.
- {doc}`../reference/model_zoo` — what models exist and their `type` names.

```{toctree}
:hidden:

dataset_format
```
