# Raw report 2 — "Apply my dataset to Panda" UX audit

> **Persona:** Researcher on a LArTPC experiment who already has labeled 3D point-cloud events (x,y,z + energy, per-point semantic + per-instance particle labels). Wants to **use the shipped Panda detector** — start from a pretrained checkpoint, run inference, then fine-tune on own labels. Does not want to invent a new model.
> **Method:** Browsed the **live rendered site** only; no access to `docs/` source. Cross-checked HF repos via the public API.
> **Date:** 2026-06-25 · **Resume this agent via SendMessage:** `a9ab4ebfed20d3bc3`

---

# Panda Detector Docs — Adversarial UX Audit (persona: LArTPC researcher with own labeled point clouds)

## 1. Journey log (ordered URLs visited)

1. `https://deeplearnphysics.org/particle-imaging-models/stable/` (landing)
2. `.../tutorials/index.html`
3. `.../tutorials/panda_detector.html` (headline artifact)
4. `.../models/dataset_format.html`
5. `.../reference/model_zoo.html` (fetched twice to verify "no checkpoints")
6. `.../models/index.html`
7. `.../datasets/packed_format.html`
8. `.../datasets/bring_your_own.html`
9. `.../reference/cli.html`
10. `.../datasets/pilarnet.html`
11. `.../datasets/transforms.html`
12. `.../checkpoints/huggingface.html`
13. `.../checkpoints/export.html`
14. `.../evaluation/index.html`
15. `.../configuration/index.html`
16. `.../api/registry/generated/pimm.models.panda_detector.detector_v1m1.Detector.html`
17. `.../api/registry/generated/pimm.datasets.transform.detector.PDGToSemantic.html`
18. `.../api/loading.html`
19. `.../datasets/builtin_datasets.html`
20. `.../tutorials/byo_dataset_semseg.html`
- External verification: `https://huggingface.co/DeepLearnPhysics`, `https://huggingface.co/api/models/DeepLearnPhysics/Panda` (public), `https://huggingface.co/youngsm/sonata-pilarnet-L` + `/api/models/...` (HTTP 401), one WebSearch.

## 2. Verdict

Following the docs alone, this persona **cannot** get a pretrained Panda producing interpretable predictions on their own data, and can only **partially** fine-tune. The training mechanics are well-documented (packed format, BYO reader, config inheritance, hooks, warm-start remap), but the entire "use the shipped pretrained model" arc collapses at the last mile: the pretrained Panda checkpoints exist and are public on HuggingFace, yet **no doc page ever mentions or links them**; the one warm-start checkpoint the tutorials hardcode is **inaccessible (HTTP 401)**; and the only inference snippet stops at raw query tensors without the `postprocess()` step needed to get per-point instances/PIDs/scores.

**Single biggest blocker:** The pretrained artifacts are undiscoverable and/or unobtainable from the docs. `tutorials/panda_detector.html` and `tutorials/byo_dataset_semseg.html` both hardcode `--train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth`, which is not public (HF API returns 401 for both the page and the API). The actually-public weights — `DeepLearnPhysics/Panda` (files `panda_base.pth`, `panda_particle.pth`, `panda_interaction.pth`, `panda_semantic.pth`, MIT, updated 2025-11-30) — are never referenced anywhere in the docs, ship as bare `.pth` with an **empty README**, and have no documented config/PID mapping, so `pimm.from_pretrained` cannot rebuild them.

- **(a) Inference on own data: No.** Model loading + packed-batch construction are documented, but (1) you're never told the Panda weights exist or how to load bare `.pth` (repo has no `training_config.json`, which `from_pretrained` requires per `checkpoints/export.html`); (2) the only inference example (`tutorials/panda_detector.html` §7) returns `pred_masks`/`pred_logits` with no call to `Detector.postprocess(...)` and no spec of postprocess output, so you can't turn outputs into instances/PIDs/scores.
- **(b) Fine-tune/train on own data: Partially.** There is a genuine, fairly complete recipe (custom reader → config copy → `CheckpointLoader` remap → detector loss/evaluator). But the documented warm-start checkpoint is broken (401); whether per-point `momentum (N,3)` labels are mandatory is never stated; the PID label scheme is buried in an autodoc page; and registration requires editing installed package source. A from-scratch (`-scratch`) run is achievable; the documented warm-start is not.

## 3. The data-format answer (reconstructed from the site only)

**What I could determine.** Panda consumes the standard packed batch. The raw per-event dict your reader/dataset must emit (assembled from `datasets/bring_your_own.html`, `models/dataset_format.html`, `datasets/pilarnet.html`, `tutorials/panda_detector.html`):

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `coord` | (N,3) | float32 | raw detector coords; PILArNet center≈[384,384,384], extent 768 |
| `energy` | (N,1) | float32 | per-point deposit; threshold **0.13** must equal `LogTransform min_val` |
| `segment` | (N,) | int64 | per-point **PID class** (detector renames PILArNet `segment_pid`→`segment` via `Copy`) |
| `instance` | (N,) | int64 | per-point instance/particle ID (renamed from `instance_particle`); "background/ignore convention" mentioned |
| `momentum` | (N,3) | float | PILArNet v2/v3; referenced by loss `momentum_loss_weight=1.0` |
| `name`,`split` | scalar | str | metadata |

PID scheme (`pid_6cls`, only on `PDGToSemantic` autodoc): `photon=0, electron=1, muon=2, pion=3, proton=4, other=5`; detector uses `num_classes=6`, `stuff_classes=[5]`. Transform pipeline (must match training): `NormalizeCoord(center,scale)` → `LogTransform(min_val=0.13,max_val=20.0)` → `GridSample(grid_size=0.001, hash_type="fnv", return_grid_coord=True)` → `Copy(segment_pid→segment, instance_particle→instance)` → `ToTensor` → `Collect(keys=("coord","grid_coord","segment","instance"), feat_keys=("coord","energy"))`. `feat_keys=("coord","energy")` ⇒ `feat` is (N,4) = [x,y,z,energy] ⇒ `in_channels=4`. Packed model input: `coord (N,3)`, `grid_coord (N,3)`, `feat (N,4)`, `offset (B,)` cumulative, `segment`, `instance`, `name` list (`datasets/packed_format.html`). Voxel size `grid_size=0.001` (PILArNet, normalized units) vs `0.01` (BYO example) — clearly dataset-specific.

**What is left undefined on the site:**
- The actual **PDG-integer → class** dictionary (only class-name → index is shown for `pid_6cls`; the literal `{22:0, 11:1, ...}` is not given — you'd guess or use `custom_map`).
- Whether `momentum (N,3)` is **mandatory** for `detector-v1m1` and how to disable the momentum head if you (like this persona) have only energy. The reference config keeps `momentum_loss_weight=1.0`; nothing says how to drop it.
- The **instance ignore/background value** (is it `-1`, `0`?) — only described prose-wise as "a background/ignore convention."
- The **postprocess() output contract** (keys/shapes of final masks, PIDs, scores).
- Single canonical "detector data contract" page — the answer is scattered across ~4 pages + 1 autodoc.

## 4. Gaps & friction (prioritized)

**[P0] No discoverable/loadable pretrained Panda.** `reference/model_zoo.html` — intro: *"A catalog of the models registered in pimm…"* — contains **zero** checkpoints, hf:// URLs, or `.pth` names (verified twice); it only says weights live "elsewhere." No doc page links `DeepLearnPhysics/Panda`, which is public and real. *Blocks:* the persona's primary goal ("start from a pretrained Panda, run inference"). *Fix:* Add a real zoo table: repo `DeepLearnPhysics/Panda`, the four `.pth` files with what each is (semantic/particle/interaction/base), the matching `configs/panda/...` for each, num_classes/PID scheme, and a copy-paste load snippet. Re-export them as proper `from_pretrained` artifacts (add `training_config.json` + `model.safetensors`) and write the empty README.

**[P0] Hardcoded warm-start checkpoint is inaccessible.** `tutorials/panda_detector.html` and `tutorials/byo_dataset_semseg.html`: *"--train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth"*. The page and HF API both return **HTTP 401** (private/nonexistent), while `DeepLearnPhysics/Panda` returns clean public JSON — so this is not a WebFetch artifact. `checkpoints/huggingface.html` separately calls these "example repo IDs," but the tutorials present them as literal runnable commands. *Blocks:* the documented warm-start (Step 3, the central idea of the tutorial). *Fix:* publish the Sonata encoder under the `DeepLearnPhysics` org (or point to the real public repo) and use that ID consistently; if it must stay an example, label it as a placeholder.

**[P0] Inference example is incomplete — no postprocess, no output spec.** `tutorials/panda_detector.html` §7 ends at `masks = out["pred_masks"]; logits = out["pred_logits"]`. The Detector autodoc reveals a `postprocess(forward_output, stuff_threshold=…, mask_threshold=…, conf_threshold=…, nms_*, min_points, background_class_label, fill_uncovered)` method that the tutorial never calls, and its returned keys/shapes are undocumented. *Blocks:* "run inference and look at the output (masks, PIDs, scores)." *Fix:* show the full path `out = model(batch, return_point=True); preds = model.postprocess(out)` and document `preds` keys (per-point instance id, class, score, momentum).

**[P1] PID label scheme is buried.** The class→integer map (`pid_6cls`) appears only on `api/.../PDGToSemantic.html`. `datasets/transforms.html` explicitly omits it ("does not contain explicit PDG-to-semantic class mapping tables"); `datasets/pilarnet.html` describes `segment_pid` as just "PID class labels (v2/v3 only)" with no integers. Naming is also inconsistent: tutorial calls class 5 "led", the transform calls it "other". *Forces guessing* when mapping the persona's own PID labels. *Fix:* put the `pid_6cls` table on the dataset-format and Panda-tutorial pages; reconcile "led" vs "other".

**[P1] Panda tutorial is not standalone.** The config block contains `# ... (same enc/dec dims as the semseg backbone)` — the full PTv3 backbone (`enc_depths`, `enc_channels`, `dec_*`) lives **only** in `tutorials/byo_dataset_semseg.html`. So copying the Panda config alone yields a non-runnable model. *Fix:* include the full backbone block (or an explicit `_base_` import) in the Panda config.

**[P1] Momentum requirement unclear for BYO data.** The reference `detector-v1m1` loss has `momentum_loss_weight=1.0` and the architecture emits `pred_momentum`, but PILArNet supplies `momentum`; a BYO user with only energy is never told whether momentum labels are required or how to disable that head. *Fix:* state it, and show the energy-only variant.

**[P1] BYO requires editing installed package source.** `datasets/bring_your_own.html`: *"Add import to pimm/datasets/__init__.py … Config-level __import__ is insufficient for resume."* For a pip/container install this is awkward and undocumented for editable installs. *Fix:* document an editable-install/plugin path, or a supported `custom_imports` that survives resume.

**[P2] `api/loading.html` and `datasets/builtin_datasets.html` are near-empty stubs** that only defer to autodoc/`pilarnet.html`. `PILArNetH5Dataset` constructor args (`revision`, `remove_low_energy_scatters`, `min_points`) never appear on a narrative page even though `models/dataset_format.html` flags `remove_low_energy_scatters` as a correctness gotcha. *Fix:* document the constructor and a ready detector config.

**[P2] No dedicated inference CLI.** `reference/cli.html`: *"No dedicated inference command — only train.sh and test.sh."* Reasonable, but the persona expects a `pimm predict`. *Fix:* either add one or cross-link the Python inference recipe prominently from the CLI page.

## 5. "I needed this but couldn't find it"

- A direct link + load instructions for the real pretrained Panda (`DeepLearnPhysics/Panda`) — and what its 4 `.pth` files are.
- A working, accessible Sonata warm-start checkpoint ID (the hardcoded one is 401).
- A complete inference example including `postprocess()` and a spec of the prediction output (per-point instance/PID/score/momentum).
- The `pid_6cls` integer scheme stated on the data-format/tutorial pages, plus the instance ignore value.
- A clear statement of whether `momentum` labels are required, with an energy-only config diff.
- One consolidated "Panda data contract" page (today it's spread across 4+ pages).

## 6. Broken / confusing links & rendering

- **Broken/inaccessible (P0):** `hf://youngsm/sonata-pilarnet-L/model_best.pth` — returns HTTP 401 from both `https://huggingface.co/youngsm/sonata-pilarnet-L` and the HF API. Referenced from `tutorials/panda_detector.html`, `tutorials/byo_dataset_semseg.html`, `models/index.html`, `checkpoints/huggingface.html`.
- **Confusing:** `DeepLearnPhysics/Panda` is reachable only by clicking the generic global "Hugging Face" header link → org page → repo with an **empty README** (verified empty). No in-docs path leads there intentionally.
- **Thin pages:** `api/loading.html`, `datasets/builtin_datasets.html` render as one-line redirects to other pages.
- (Note: the `example.com` hrefs that appeared in the Detector autodoc fetch are a fetch-summarizer artifact, not a real site defect — not counted.)

## 7. Top 5 prioritized recommendations

1. **Publish a real Model Zoo entry for `DeepLearnPhysics/Panda`** (P0): re-export as `from_pretrained`-loadable artifacts (`model.safetensors` + `training_config.json`), write the README, and add a docs table mapping each `.pth` → config → task → PID scheme with a copy-paste load+infer snippet.
2. **Fix the warm-start checkpoint** (P0): replace `youngsm/sonata-pilarnet-L` with a public, accessible repo ID everywhere, or clearly mark it a placeholder.
3. **Complete the inference recipe** (P0): add `model.postprocess(...)` to `tutorials/panda_detector.html` §7 and document the prediction-output schema (masks→per-point instance ids, PIDs, scores, momentum).
4. **Make the Panda tutorial standalone + state the data contract** (P1): inline the full PTv3 backbone config, add the `pid_6cls` table and the instance ignore convention, and clarify the `momentum` requirement with an energy-only variant.
5. **Document a non-source-editing BYO path** (P1): support reader registration that survives resume without editing `pimm/datasets/__init__.py`, and flesh out `datasets/builtin_datasets.html` with the `PILArNetH5Dataset` constructor + a ready detector config to copy.
