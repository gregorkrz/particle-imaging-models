# Publishing your model

Once you have a trained model that others should be able to use, publishing it takes three steps: export a portable artifact, push it to the Hub, and let others load it with one call.
For the full detail behind each step, follow the links down to the Checkpoints section.

:::{seealso}
The other half of this loop - **loading** a published model for inference or fine-tuning - lives in {doc}`using_trained_models`.
:::

## 1. Export a consolidated checkpoint

A training checkpoint carries everything needed to *resume* (optimizer, RNG, dataloader state).
To share a model you only need the weights, so first export a portable artifact:

```bash
pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model
```

This reads the `last` checkpoint under `<run-dir>/model`, pairs it with the run's config, and writes an export directory containing:

```text
artifacts/my-model/
  model.safetensors      # the weights (model.bin with --no-safe-serialization)
  config.json            # the resolved training config, so from_pretrained is free
  README.md              # only when a model card is supplied via --model-card
```

The architecture code itself always comes from the reader's local `pimm` install; the export just carries weights plus the config needed to rebuild the model.
See {doc}`../checkpoints/exporting` for the Python equivalent ({py:func}`~pimm.save_pretrained`) and for verifying the round trip before you publish.

## 2. Push to the Hub

Pass `--push-to-hub` to upload the export to a Hugging Face repo in the same command (repos are private by default; add `--public` for a public one):

```bash
pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model \
  --push-to-hub <your-org>/my-model
```

From Python, {py:func}`~pimm.export.push_to_hub` does the same thing, and inside a training run the {py:class}`~pimm.engines.hooks.export.PushToHub` hook uploads automatically at the end of training (or on every best metric):

```python
hooks = [
    ...,
    dict(type="CheckpointSaver"),
    dict(type="PushToHub", repo_id="<your-org>/my-model", private=True),
]
```

See {doc}`../checkpoints/huggingface` for the full `PushToHub` option table, manual `push_to_hub`, and the cross-cluster recipe.

## 3. How others load it

A consolidated export loads with a single call:

```python
import pimm
model = pimm.from_pretrained("<your-org>/my-model", device="cuda")
```

Because `config.json` travels with the weights, {py:func}`~pimm.from_pretrained` rebuilds the architecture with no extra config.
For loading from every source (local dir, Hub repo, `hf://` URI, raw checkpoint) and feeding the model the right data, see {doc}`using_trained_models`.
