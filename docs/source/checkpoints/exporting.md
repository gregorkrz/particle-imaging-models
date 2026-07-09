# Exporting

A training checkpoint carries everything needed to *resume* - optimizer, RNG,
dataloader state, and so on ({doc}`index`). To **share** a model - for
fine-tuning, uploading to the Hub, or shipping elsewhere for inference - you only
need the weights. `pimm.export` produces these portable weights and can optionally
publish them to Hugging Face.

Inside a training run, saving and uploading happen automatically via hooks (see
{doc}`saving_and_loading` and {doc}`huggingface`). This page is the **manual**
path: turning a checkpoint into a portable artifact yourself.

## What an export contains

An export is the model weights plus a small `config.json`. The `config.json`
records the hyperparameters `pimm` needs to instantiate the architecture (number
of layers, hidden dimension, etc.). An exported directory looks like:

```text
artifacts/my-model/
  model.safetensors      # or model.bin with safe_serialization=False
  config.json            # the resolved training config (makes from_pretrained free)
  README.md              # only when a model_card is supplied
```

The architecture code itself is always supplied by your local install of `pimm`.

## Exporting from the CLI

```bash
pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model
```

This reads the checkpoint named `last` under `<run-dir>/model`, pairs it with the
run's config, and writes a pretrained directory to `./artifacts/my-model`. It
handles split checkpoint directories (e.g. `model/last`) and legacy `.pth`
weights files, and can take a direct checkpoint path instead of `--run-dir` +
name. See `pimm export --help` for safe-serialization and Hub-upload options.

## Exporting in Python

{py:func}`~pimm.save_pretrained` does what `pimm export` does, from a Python
runtime (e.g. a notebook):

```python
from pimm.export import save_pretrained

save_pretrained(
    model,                                   # trained nn.Module, state-dict, or checkpoint path
    "exports/my-model",
    training_config=cfg,                     # saved to config.json
    safe_serialization=True,                 # safetensors
)
```

The first argument may be a `torch.nn.Module`, a raw state-dict, a checkpoint
mapping (with `state_dict`/`model`), or a checkpoint path. The config is recorded
by priority: explicit `training_config` > `cfg` > `config_path` > the run dir
around a checkpoint path - so you don't need to pass `training_config` if it's
already next to your checkpoint.

## Verify the round trip

Before publishing, reload the export with {py:func}`~pimm.from_pretrained` to
confirm it rebuilds the architecture and loads cleanly:

```python
import pimm

model, meta = pimm.from_pretrained("exports/my-model", return_metadata=True)
print(meta["model_config"])   # config used to rebuild the architecture
```

A good export reloads with `strict=True` (the default) and no errors. If you only
want *part* of the saved weights - for example just the student backbone from a
Sonata export - see {doc}`saving_and_loading`, which covers submodel loading and
key remapping.
