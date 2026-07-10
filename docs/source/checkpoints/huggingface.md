# Hugging Face integration

pimm can **push checkpoints to the Hub during or after training** and **load
weights straight from the Hub** for fine-tuning or inference. This is the
recommended way to move models across clusters that don't share a filesystem
(e.g. pretrain on one cluster, fine-tune on another).

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
  - -
  - target Hub repo (created if missing)
* - `checkpoint`
  - `"model_best"`
  - which checkpoint to push (`"last"`/`"model_last"` resolve the newest rolling
    checkpoint - including the `model/last/` DCP directory)
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
  - -
  - repo visibility, auth token, branch/tag, file name
```

:::{warning}
Repeatedly uploading large checkpoints to the same path accumulates blobs in the
repo's LFS history. For frequent periodic pushes, use a dedicated repo and/or a
long `every_n_epochs` interval. Upload failures are logged but never crash the
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

`pimm export` can also push directly - see `pimm export --help` for the
`--push-to-hub` options.

## Load what you pushed

Loading from the Hub uses the same mechanism as loading from a local path: point a
training config's `weight=` at an `hf://` URI to fine-tune from a pushed checkpoint.

```bash
pimm submit --site mycluster \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

`<your-org>/...` is a placeholder for your own pushed Sonata checkpoint - its
`student.backbone.*` keys are what the fine-tune configs' `CheckpointLoader`
remap requires.

The full loading behavior (config precedence, `hf://` and `from_pretrained` usage,
download cache, key remapping) is documented in
{doc}`../research_ecosystem/using_trained_models`.

Composing an auto-push run on one cluster with an `hf://` fine-tune on another
gives a cross-cluster workflow with no shared filesystem: the Hub is the transport.

```text
Cluster A (pretrain)                     Cluster B (fine-tune / monitor)
  hooks += PushToHub(                      pimm submit --site mycluster \
    repo_id="me/sonata-L",                   --train.weight hf://me/sonata-L/model_best.pth \
    on_best=True,                            --train.config .../semseg-...
    every_n_epochs=5)               ──Hub──▶
```
