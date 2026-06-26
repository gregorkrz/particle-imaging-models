# Raw report 3 — "Bring my own dataset AND model on pimm" UX audit

> **Persona:** Experienced ML researcher (PyTorch, DDP/FSDP, custom datasets). Doesn't care about Panda — wants to use **pimm as a substrate/framework** (registries, configs, packed-tensor contract, distributed training, exact-resume checkpointing, hooks, eval harness, HPC launch) to train a **novel model architecture on their own dataset**.
> **Method:** Browsed the **live rendered site** only; no access to `docs/` source. Tested whether someone could build on pimm from the docs without reading source.
> **Date:** 2026-06-25 · **Resume this agent via SendMessage:** `ad1cc4546d7c83fc9`

---

# pimm Docs Audit — Persona: ML researcher using pimm as a framework substrate

## 1. Journey log (ordered URLs visited)

1. `/stable/` (landing)
2. `/stable/getting_started/concepts.html`
3. `/stable/datasets/index.html`
4. `/stable/datasets/packed_format.html`
5. `/stable/datasets/bring_your_own.html`
6. `/stable/datasets/transforms.html`
7. `/stable/tutorials/byo_dataset_semseg.html`
8. `/stable/configuration/index.html`
9. `/stable/models/dataset_format.html`
10. `/stable/api/index.html`
11. `/stable/api/registry/models.html`
12. `/stable/api/registry/datasets.html`
13. `/stable/api/registry/transforms.html`
14. `/stable/api/registry/hooks.html`
15. `/stable/api/generated/pimm.models.modules.PointModel.html`
16. `/stable/api/registry/generated/pimm.models.default.DefaultSegmentorV2.html`
17. `/stable/api/generated/pimm.models.utils.structure.Point.html`
18. `/stable/api/registry/generated/pimm.datasets.defaults.DefaultDataset.html`
19. `/stable/api/registry/generated/pimm.datasets.transform.base.Collect.html`
20. `/stable/models/index.html`
21. `/stable/hooks/index.html`
22. `/stable/hooks/writing_hooks.html`
23. `/stable/evaluation/index.html`
24. `/stable/api/registry/trainers.html`
25. `/stable/api/registry/losses.html`
26. `/stable/api/loading.html`
27. `/stable/api/core.html`
28. `/stable/distributed/index.html`
29. `/stable/checkpoints/index.html`
30. `/stable/reference/cli.html`
31. `/stable/api/registry/generated/pimm.engines.train.Trainer.html`
32. `/stable/api/generated/pimm.from_pretrained.html`
33. `/stable/tutorials/index.html`
34. `/stable/api/registry/generated/pimm.models.default.DefaultSegmentor.html`
35. `/stable/tutorials/panda_detector.html`
36. `/stable/getting_started/quickstart.html`

(Base: `https://deeplearnphysics.org/particle-imaging-models/stable/`)

## 2. Verdict

This persona can stand up a **custom dataset** end-to-end from the docs, can **write a config** from scratch, and can **add a custom hook** — these three are genuinely well-served. But the persona **cannot reliably stand up a custom MODEL from the docs alone.** The registration *mechanism* is documented everywhere; the model *interface* is not. There is no "bring your own model" page (only a dataset equivalent), no custom-model tutorial (both tutorials reuse built-in models like `DefaultSegmentorV2`/`detector-v1m1`), the nominal base class autodoc is an empty placeholder, and the one thing that matters most — what `forward()` must return and how losses/criteria are wired into it — exists only as prose on `concepts.html` and nowhere in the API. You'd be forced to reverse-engineer it from `DefaultSegmentor`'s `__init__(backbone, criteria)` signature and the `[source]` links.

**Single biggest blocker:** The exact custom-model contract (forward inputs → required return dict → how `criteria` produces `loss`) is undocumented as an authorable spec. `api/.../PointModel.html` literally says *"placeholder, PointModel can be customized as a pimm hook,"* and `DefaultSegmentor.forward()`'s docstring is the stock PyTorch *"Define the computation performed at every call. Should be overridden by all subclasses."*

Ratings:
- **(a) Register a custom dataset — Yes.** `datasets/bring_your_own.html` gives a complete, copy-pasteable reader+dataset+config+verification flow; the `DefaultDataset` autodoc has a real docstring and return contract; the `__init__.py`-import-for-resume rule is stated.
- **(b) Register a custom model — No (Partially at best).** Registration decorator + import rule are clear; the forward I/O contract, loss wiring, and a base class to subclass are not. No example model body, no tutorial, placeholder base class.
- **(c) Write a config from scratch — Yes.** `configuration/index.html` is the strongest page: full anatomy, `_base_` inheritance/merge/`_delete_`, default hook lifecycle, runtime-defaults groups, worked layer-wise-LR example, CLI `--options` grammar.
- **(d) Add a custom hook/evaluator — Yes for hooks / Partially for evaluators.** `hooks/writing_hooks.html` gives `HookBase`, lifecycle methods, `self.trainer` attributes, and a full example. Evaluators have no base class or example — `evaluation/index.html` only gives the selection-metric contract and punts to "Writing a hook."

## 3. The model-interface answer (reconstructed from docs only)

**What I could determine:**
- **Registration:** `@MODELS.register_module("MyModel")` on the class, and the class file **must** be imported in `pimm/models/__init__.py` (not via config `__import__`) "so they survive a resume" (`concepts.html`). Selected via `model = dict(type="MyModel", ...)`.
- **It's an `nn.Module`:** `DefaultSegmentor`/`DefaultSegmentorV2` autodoc both show `Bases: Module`. (`PointModel` is `Bases: PointModule, HookBase` but is a documented placeholder, so it is *not* the segmentation base.)
- **Forward signature:** `forward(input_dict)` (V1) / `forward(input_dict, return_point=False)` (V2). Input is the packed batch dict (coord/feat/offset/segment/…).
- **Return value:** From `concepts.html` prose only: `output_dict = model(input_dict)`; `loss = output_dict["loss"]` → backward. Contract stated as *"return a dict with a scalar `loss`."* Optional keys consumed downstream: `seg_logits`/`sem_logits` (semantic evaluators), `cls_logits` (classification), `point` (panoptic, gated by `return_point=True`), `pred_logits`/`pred_masks`/`pred_momentum` (detector), and `total_loss` (*"preferred over raw `loss`"* for logging hooks).
- **Loss config:** Losses attach via `criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)]`, assembled by `build_criteria` (`api/registry/losses.html`). `DefaultSegmentor`'s signature is `(backbone=None, criteria=None)`.

**What was undefined in the docs:**
- **How the model actually turns `criteria` into `loss`.** No example forward body calls `self.criteria(...)`; the model→loss wiring is a black box. Must read `[source]`.
- **`loss` vs `total_loss`:** trainer consumes `output["loss"]` but logging "prefers" `total_loss` — relationship/when-to-emit-both unspecified.
- **Required base class:** Is plain `nn.Module` enough, or must you subclass something to get criteria handling / `return_point` semantics? Undocumented (placeholder `PointModel`, generic `forward` docstrings).
- **Per-task required output keys:** Which keys are mandatory for `SemSegEvaluator` vs `InstanceSegmentationEvaluator` vs training-only is implied by a scattered table, never stated as a contract.
- **dtype/device entering forward** and how `num_classes`/`in_channels` flow into a custom head.

Net: registration is fully specified; the *interface a class must implement* is reconstructable only by combining `concepts.html` prose + a concrete model's `__init__` signature + reading source. Not implementable "from docs alone" with confidence.

## 4. The packed-batch answer (reconstructed from docs only)

Consistent across `concepts.html`, `packed_format.html`, `models/dataset_format.html`, `datasets/transforms.html`:

```
{
  "coord":   Tensor[N_total, D],   # D = 2 or 3; float32 (per dataset __getitem__)
  "feat":    Tensor[N_total, C],   # built by Collect(feat_keys=...); concat along dim=1
  "offset":  Tensor[batch_size],   # cumulative end indices; offset[-1] == N_total
  "segment": Tensor[N_total],      # int64 per-point labels (supervised)
  "grid_coord": Tensor[N_total, 3],# present for sparse backbones (GridSample return_grid_coord=True)
  "name":    list[str],            # per-event ids
  # + "instance" (N,) for instance/panoptic
}
```
- **Convention:** "Concatenate, not stack." No batch dimension; per-event slices come from `offset` (cumulative point counts). `Point` container can derive `batch` from `offset` if absent.
- **Collate:** tensors concat dim 0; strings→list; sequence lengths→cumulative offsets; mapping keys collated by key with **`_`-prefixed keys dropped**. Variants: `collate_fn` (base), `point_collate_fn` (`mix_prob` mixing), `inseg_collate_fn` (per-sample query-dict flattening for instance seg).
- **`Collect`** builds the model-facing contract: copies `keys`, builds `offset` from `offset_keys_dict` (default `dict(offset="coord")`), and concatenates `feat_keys` → `feat`. `feat_keys=("coord","energy")` ⇒ 4-channel `[x,y,z,E]`; channel width **must** equal backbone `in_channels`.

**Gaps:** `coord`/`feat`/`offset` dtypes are never stated in the batch table itself (only inferred from the dataset `__getitem__` example: coord float32, segment int64); `offset` dtype unstated; device contract ("trainer moves batch to device") not shown in the batch schema; the *full* required-vs-optional key set differs slightly page-to-page; no single authoritative schema table covering 2D vs 3D, supervised vs instance/panoptic, and dtypes together.

## 5. API reference assessment (per sampled page)

**Index/registry pages — THIN (index-only, no signatures):**
- `api/index.html` — **Useful as a hub.** Real categorization (6 registries + core API), good descriptions.
- `api/registry/models.html` — **Thin + partly unhelpful.** 60 names; descriptions are often the generic *"Base class for all neural network modules"* (e.g., for `MinkUNet14`, `DefaultSegmentor`). No signatures.
- `api/registry/datasets.html`, `transforms.html`, `hooks.html`, `losses.html`, `trainers.html` — **Thin** index tables with one-line summaries; no signatures, no base classes. (Datasets/transforms summaries are at least descriptive; trainers are bare.)
- `api/loading.html`, `api/core.html` — **Thin.** Function lists with one-liners ("Build models.") + links; no inline signatures.

**Generated autodoc pages — INCONSISTENT:**
- `Point` — **Populated & useful.** Real docstring, full attribute list (coord, grid_coord, offset, batch, feat, serialized_*, sparse_*), methods (`serialization`, `sparsify`, `octreelization`).
- `DefaultDataset` — **Populated & useful.** Real `__init__` signature, docstring, explicit return contract ("always carries `coord`, `segment`, `instance`, `name`, `split`"), methods, `VALID_ASSETS`.
- `Collect` — **Populated & useful.** Signature, parameter docs, behavior, runnable example.
- `DefaultSegmentor` / `DefaultSegmentorV2` — **Partial/misleading.** Real `__init__` and `forward(input_dict[, return_point])` signatures, but `forward` docstring is the stock PyTorch placeholder — **the return contract, the #1 thing this persona needs, is absent.**
- `PointModel` — **Empty stub.** *"placeholder, PointModel can be customized as a pimm hook."* `Bases: PointModule, HookBase`. No methods.
- `Trainer` (DefaultTrainer) — **Thin.** Lists `build_model`/`build_optimizer`/`run_step`/`train` with one-line each; no `run_step` forward-contract, no hook-facing attributes (`comm_info`, `storage`, `writer`).
- `from_pretrained` — **Partial.** Full real signature (16 args) but no per-argument docs.

Takeaway: dataset/transform/structure autodoc is genuinely good; **model and trainer autodoc is where it collapses** — exactly the persona's crux. Every generated page exposes a `[source]` link, so the docs effectively force source-reading for the model contract, defeating "build from docs alone."

## 6. Gaps & friction (prioritized)

**[P0] No "Bring your own model" doc and no custom-model tutorial** — `models/index.html` is consumer-only (*"Build an input the way the dataloader would … and call the model"*); `tutorials/index.html` has only BYO-dataset and Panda, both using built-in models; there is a `datasets/bring_your_own.html` but **no** `models/bring_your_own.html`. *Blocks the persona's entire reason for using pimm.* **Fix:** add a model-authoring guide mirroring the dataset one (class skeleton, decorator, `__init__.py` import, forward I/O, criteria wiring) + a custom-model tutorial.

**[P0] Model `forward()` return contract not in the API; only `concepts.html` prose** — `DefaultSegmentor(V2).forward` docstrings are generic PyTorch; `PointModel` is a "placeholder." *A developer cannot implement against a prose table on one page that the API contradicts/omits.* **Fix:** write real docstrings for `forward` (inputs = packed batch keys; returns = `{"loss": scalar, "seg_logits": ...}`), and make `PointModel` (or a documented `BaseModel`) the real, specified base.

**[P0] Loss/criteria → `loss` wiring undocumented for custom models** — `criteria=[dict(type=...)]` and `build_criteria` are named, but no example shows a model consuming criteria to produce `output["loss"]`; `loss` vs `total_loss` is ambiguous. *Without this you can register a model that the trainer can't get a loss from.* **Fix:** show the canonical forward body that applies `self.criteria` and returns the loss dict; define `loss`/`total_loss`.

**[P1] Registry API pages are bare indexes; model descriptions are generic** — repeated *"Base class for all neural network modules"* is noise; no inline signatures force a click-through to pages that are themselves stubs for models. **Fix:** inline `__init__` signatures (or at least real one-liners) and fix the generic descriptions.

**[P1] Custom evaluator has no base class or example** — `evaluation/index.html` gives only `trainer.comm_info["current_metric_value"]/["current_metric_name"]` and redirects to hooks; the read-side (pull predictions/targets from `comm_info["model_output_dict"]`/`["input_dict"]`, `all_gather`, reduce) is never exampled. **Fix:** add a minimal evaluator-hook example.

**[P1] Two generated-doc URL namespaces** — core classes live under `api/generated/...` (e.g., `PointModel`) while registry classes live under `api/registry/generated/...` (e.g., `DefaultSegmentorV2`). *Guessing a URL 404s; mentally jarring.* **Fix:** unify, or cross-link prominently.

**[P1] Packed-batch spec is scattered with no dtype-complete authoritative table** — four pages give slightly different key sets; dtypes only appear in a dataset example. **Fix:** one canonical batch-schema table (key, shape, dtype, required/optional, 2D/3D, task).

**[P2] Custom LOSS authoring undocumented** — `losses` page lists built-ins only; no `@LOSSES.register_module()` example, no `forward(pred, target)` signature/base class.

**[P2] Custom TRANSFORM authoring is half-specified** — `transforms.html` states the `@TRANSFORMS.register_module()` + `__call__(data_dict)->data_dict` convention but gives no full example and (unlike datasets/hooks) omits the import-in-`__init__.py`-for-resume warning.

**[P2] HF export round-trip for a CUSTOM model type unspecified** — `from_pretrained` has `model_type`/`model_cls`/`model_config` args, but docs don't state whether a custom registered `type` auto-exports and is reconstructable (presumably via `model_config.json`, but never spelled out).

**[P2] Trainer attributes are documented on `hooks/index.html`, not in the `Trainer` autodoc** — split-brain; `comm_info` keys (`input_dict`, `model_output_dict`) — the de facto model↔hook interface — never appear on evaluation or any API page.

## 7. "I needed this but couldn't find it"

- A minimal, copy-pasteable **custom model on custom data** example (model class body + config + launch). None exists.
- The **exact forward return spec per task** (train-only vs `SemSegEvaluator` vs `InstanceSegmentationEvaluator`).
- **How a model applies `criteria`** to produce `loss` (and `loss` vs `total_loss`).
- One **canonical packed-batch schema** with dtypes + required/optional + 2D/3D + instance/panoptic keys + device.
- **Register a custom loss** and **custom evaluator** with examples.
- **`from_pretrained` round-trip for a custom `type`** (does my registered model auto-export and reload?).
- Confirmation of **feat/coord/offset dtypes** and device contract at the `forward` boundary.

## 8. Broken / confusing / rendering issues

- **Dual generated-doc namespaces** (`api/generated/` vs `api/registry/generated/`) — confirmed real; e.g., `PointModel` is at `api/generated/pimm.models.modules.PointModel.html` but `DefaultSegmentorV2` is at `api/registry/generated/pimm.models.default.DefaultSegmentorV2.html`. URL-guessing fails.
- `api/registry/models.html` — generic placeholder descriptions (*"Base class for all neural network modules"*) used as the human-facing summary for many models — confusing/borderline-misleading.
- `Trainer` autodoc vs `hooks/index.html` — disagree on where trainer attributes are documented; neither cross-links.
- **Needs manual verification (likely a fetch artifact, not confirmed broken):** the "Related links" on `datasets/bring_your_own.html` surfaced as `https://deeplearnphysics.org/pimm/datasets/...` (wrong base path) in my read. Other pages rendered the same links correctly as relative (`packed_format.html`), and the canonical pages resolve, so this is probably a rendering/normalization artifact rather than a live broken link — worth a quick manual check of that page's cross-ref base URL.

## 9. Top 5 prioritized recommendations

1. **Write a "Bring your own model" guide + custom-model tutorial.** Mirror `datasets/bring_your_own.html`: full `nn.Module` subclass with `@MODELS.register_module()`, `forward(input_dict)` consuming `coord/feat/offset`, applying `build_criteria(criteria)`, returning `{"loss":…, "seg_logits":…}`, plus the `pimm/models/__init__.py` import-for-resume note and a verification snippet. This single addition unblocks the entire persona.
2. **Put the model contract in the API.** Replace the generic `forward` docstrings on `DefaultSegmentor(V2)` with real input/return docs, and turn `PointModel` (or a new `BaseModel`) from "placeholder" into the documented base with the canonical forward signature and loss-wiring.
3. **Publish one canonical batch-schema reference** (key, shape, dtype, required/optional, 2D/3D, supervised/instance/panoptic, device), consolidating `concepts`/`packed_format`/`dataset_format`/`transforms`.
4. **Make registry API pages usable** — inline real `__init__` signatures (or accurate one-liners), kill the repeated "Base class for all neural network modules" descriptions, and unify the `api/generated` vs `api/registry/generated` URL scheme.
5. **Add custom-evaluator and custom-loss authoring docs** — evaluator example that reads `comm_info["model_output_dict"]`/`["input_dict"]`, `all_gather`s, and writes `current_metric_value`/`current_metric_name`; loss example with `@LOSSES.register_module()` and a `forward(pred, target)` signature.
