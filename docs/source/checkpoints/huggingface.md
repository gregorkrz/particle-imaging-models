# Hugging Face integration

pimm can **push checkpoints to the Hub during or after training** and **load
weights straight from the Hub** for fine-tuning or inference. This is the
recommended way to move models across clusters that don't share a filesystem
(e.g. pretrain on S3DF, fine-tune on NERSC).

## Auto-push during training: the `PushToHub` hook

Add {py:class}`~pimm.engines.hooks.export.PushToHub` to `cfg.hooks`, *after* the checkpoint saver so the files it
reads are already written:

```python
hooks = [
    ...,
    dict(type="CheckpointSaver"),
    dict(type="PushToHub", repo_id="<your-org>/sonata-pilarnet-L", private=True),
]
```

By default it uploads the raw `model_best` checkpoint at the end of training
(`weights_only=True`), so it loads back **byte-identically** via
`weight=hf://<repo>/model_best.pth`. Key options:

```{list-table}
:header-rows: 1
:widths: 26 18 56

* - Option
  - Default
  - Effect
* - `repo_id`
  - —
  - target Hub repo (created if missing)
* - `checkpoint`
  - `"model_best"`
  - which checkpoint to push (`"last"`/`"model_last"` resolve the newest rolling
    checkpoint — including the `model/last/` DCP directory)
* - `weights_only`
  - `True`
  - push the raw checkpoint; `False` pushes a consolidated `pimm export` artifact
* - `on_train_end`
  - `True`
  - push once when training finishes
* - `on_best`
  - `False`
  - push `model_best` whenever the metric improves
* - `every_n_epochs`
  - `None`
  - push `model_last` every N epochs (cross-cluster monitoring)
* - `background`
  - `True`
  - periodic uploads run in a thread so they never block the step
* - `private`, `token`, `revision`, `name`
  - —
  - repo visibility, auth token, branch/tag, file name
```

:::{warning}
Repeatedly uploading large checkpoints to the same path accumulates blobs in the
repo's LFS history. For frequent periodic pushes, use a dedicated repo and/or a
long `every_n_epochs` cadence. Upload failures are logged but never crash the
run.
:::

## Manual push: `push_to_hub`

```python
from pimm.export import push_to_hub

push_to_hub(
    model,                                  # module, checkpoint, or export dir
    "org-or-user/my-pimm-model",
    training_config=cfg,                    # forwarded to save_pretrained → config.json
    private=True,
)
```

If the first argument is already an export directory (contains a weights file),
pimm uploads it as-is. Otherwise it runs {py:func}`~pimm.save_pretrained` into a temp/given
directory (forwarding any `save_pretrained` kwargs such as `training_config`) and
uploads the allowed files (weights, `config.json`, `README.md`). Pass `private=`
and an optional `revision`. Requires `huggingface_hub` and Hub authentication.

`pimm export` can also push directly — see `pimm export --help` for the
`--push-to-hub` options.

## Load from the Hub

Two distinct entry points, depending on what you have on the Hub:

::::{tab-set}

:::{tab-item} Fine-tune a training run
A raw checkpoint pushed with `weights_only=True` loads via an `hf://` URI in your
config's `weight=` (same scheme as a local path):

```bash
pimm submit --site nersc \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

`<your-org>/...` is a placeholder for your own pushed Sonata checkpoint. Published
task checkpoints (loaded with `from_pretrained` for inference) are on the Hub —
see {doc}`../research_ecosystem/using_trained_models`.
:::

:::{tab-item} Build a ready-to-use model
A consolidated export loads via {py:func}`~pimm.from_pretrained`:

```python
import pimm
model = pimm.from_pretrained("org-or-user/my-model", device="cuda")
model = pimm.from_pretrained("hf://org-or-user/my-model@v2")  # with a revision
```
:::

::::

Download location for Hub loads is `PIMM_HF_CACHE` (else HF's `HF_HOME`). Hub
loading requires the optional `huggingface_hub` package. For the full
`from_pretrained` contract (config precedence, drift tolerance) see
{doc}`../research_ecosystem/using_trained_models`; for checkpoint-side loading (the `CheckpointLoader` hook,
key remapping, submodel/partial loads) see {doc}`saving_and_loading`.

## Cross-cluster recipe

```text
S3DF (pretrain)                          NERSC (fine-tune / monitor)
  hooks += PushToHub(                      pimm submit --site nersc \
    repo_id="me/sonata-L",                   --train.weight hf://me/sonata-L/model_best.pth \
    on_best=True,                            --train.config .../semseg-...
    every_n_epochs=5)               ──Hub──▶
```

No shared filesystem required — the Hub is the transport. This pattern has been
verified bitwise on a Sonata-L checkpoint.

## Next

- {doc}`../research_ecosystem/using_trained_models` — `from_pretrained` and loading data the right way.
- {doc}`exporting` — building the export artifact you push.
