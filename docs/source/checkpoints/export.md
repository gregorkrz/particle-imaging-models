# Manual export

Training checkpoints carry optimizer/RNG/dataloader state for resume. For
*sharing* a model — fine-tuning elsewhere, uploading to the Hub, or shipping for
inference — you want a **portable, self-contained artifact**: the model weights
plus the config needed to rebuild them. That's what export produces.

## `pimm export` (CLI)

```bash
pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model
```

This reads the checkpoint named `last` under `<run-dir>/model`, pairs it with the
run's resolved config, and writes a pretrained directory to `./artifacts/my-model`.
It handles split checkpoint directories (`model/last`) and legacy `.pth` files,
and can take a direct checkpoint path instead of `--run-dir` + name. See
`pimm export --help` for safe-serialization and Hub-upload options.

## What an export contains

An export is the model weights plus a small `config.json` — no optimizer, RNG, or
dataloader state, and no `model_config.json` or `pimm_config.py`:

```text
artifacts/my-model/
  model.safetensors      # or model.bin with safe_serialization=False
  config.json            # the resolved training config (makes from_pretrained free)
  README.md              # only when a model_card is supplied
```

`config.json` is the resolved training config; its `["model"]` section
is what lets `pimm.from_pretrained("artifacts/my-model")` rebuild the
architecture before loading weights. The architecture is always supplied on the
*loading* side — a fine-tune config, or
`model_config`/`config_path`/`config.json` for {py:func}`~pimm.from_pretrained`.

## `save_pretrained` (Python)

```python
from pimm.export import save_pretrained

save_pretrained(
    model,                                   # nn.Module, state-dict, or checkpoint path
    "exports/my-model",
    model_config=dict(type="my-model-v1", hidden_dim=256),
    training_config=cfg,                     # provenance → config.json
    safe_serialization=True,                 # safetensors
)
```

The first argument may be a `torch.nn.Module`, a raw state-dict, a checkpoint
mapping (with `state_dict`/`model`), or a checkpoint path. The config is recorded
by priority: explicit `training_config` > `cfg` > `config_path` > the run dir
around a checkpoint path. (For a checkpoint inside `.../model/<file>`, the run
root is resolved logically, so `MODEL_DIR`-symlinked `model/` dirs work.)

## Export checklist

Before you publish a model, verify the round trip:

```python
import pimm
model, meta = pimm.from_pretrained("exports/my-model", return_metadata=True)
```

- `build_model(model_config)` constructs the same architecture.
- State-dict keys match under `strict=True` (or you've documented why you need
  `strict=False`).
- You included `cfg` / `training_config` / `config_path` so provenance is
  preserved.
- `safe_serialization=True` when `safetensors` is available.

## Low-level helpers

For partial loads (loading only a submodule, or remapping keys), the
helpers in `pimm.export` are more surgical than a full reload:

```python
from pimm.export import load_pretrained

load_pretrained(
    model.backbone,
    "exp/pretrain/model/model_last.pth",
    prefix="student.backbone.",   # keep + strip this prefix
    remove_prefix=True,
    strict=False,
)
```

| Helper | Purpose |
|--------|---------|
| `clean_state_dict` | strip `module.` / `_orig_mod.` prefixes |
| `load_state_dict_from_checkpoint` | load `.safetensors` or torch checkpoints → state dict |
| `filter_state_dict_by_prefix` | keep keys with a prefix (optionally stripping it) |
| `remap_state_dict_keys` | exact-then-prefix key remapping |
| `load_pretrained` | load into an existing model with filtering/remapping |

:::{note}
`pimm/models/utils/checkpoint_loader.py` is a compatibility shim re-exporting
`pimm.export.checkpoint`. New code should import from `pimm.export`.
:::

## Next

- {doc}`huggingface` — push exports (or raw checkpoints) to the Hub.
- {doc}`../models/index` — loading exports for inference and fine-tuning.
