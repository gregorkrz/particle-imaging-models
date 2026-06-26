# Reference

Quick-lookup material: the command-line interface and the catalog of registered
models. For the conceptual material, start at {doc}`../getting_started/concepts`;
for how configs are written, see {doc}`../configuration/index`.

- {doc}`CLI reference <cli>` — `pimm launch`, `pimm submit`, and `pimm export`;
  common flags; config-path normalization; post-`--` training overrides;
  and the direct `scripts/train.sh` / `scripts/test.sh` flags.
- {doc}`Model zoo <model_zoo>` — backbones, pretraining methods, and
  segmentation/detection heads, each with its registry `type` name, task, and
  paper link, plus the `vXmY` versioning convention.
- {doc}`Environment variables <environment>` — data roots, checkpoint/logging,
  Hugging Face cache, distributed, and build variables, in one table.

## Other references

- {doc}`../checkpoints/index` — the on-disk checkpoint formats and the
  `pimm export` artifact layout.
- {doc}`../datasets/index` — the packed point-cloud contract and dataset readers.
- {doc}`../hooks/index` — the lifecycle hook system and built-in hooks.

```{toctree}
:hidden:

cli
model_zoo
environment
```
