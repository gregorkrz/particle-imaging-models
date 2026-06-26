# pimm documentation audit — June 2026

Working notes for a documentation overhaul of the **live site**
(<https://deeplearnphysics.org/particle-imaging-models/stable/>). This folder is
deliberately **outside `docs/source/`**, so it is not part of the Sphinx build.

## How this audit was run

Four sub-agents each browsed the **rendered live site only** (no access to the
`docs/` source), adopting a distinct real user persona and walking that user's
journey end-to-end — following sidebar links, cross-references, and external
links as a user would. Together they fetched ~100 pages and verified the
load-bearing claims (dataset size, HF repos, link integrity) first-hand.

| # | Persona / lens | Maps to the question | Raw report |
|---|----------------|----------------------|------------|
| 1 | Cold newcomer (neutrino PhD, knows PyTorch) | *Understandable & followable?* | [`raw/1-newcomer-onboarding.md`](raw/1-newcomer-onboarding.md) |
| 2 | Has a dataset, wants to run **Panda** | *Easy to apply your dataset to Panda?* | [`raw/2-panda-my-dataset.md`](raw/2-panda-my-dataset.md) |
| 3 | Brings own dataset **+ own model**, pimm as substrate | *Easy to build on pimm?* | [`raw/3-byo-model-substrate.md`](raw/3-byo-model-substrate.md) |
| 4 | Findability / navigation / link integrity | *Easy to get the info you want?* | [`raw/4-findability-and-links.md`](raw/4-findability-and-links.md) |

Sub-agents are resumable via SendMessage (IDs in each raw file's header) if we
want to push back on a finding or ask for more detail.

---

## TL;DR — one line per persona

- **Newcomer:** Comprehension OK, **first run blocked** — every "first command" secretly needs a 168 GB download + an NVIDIA GPU, and the honest page (`first_run`) is 3rd in the sequence. No prerequisites box, no data-free smoke test.
- **Panda-with-my-data:** **Inference = No, fine-tune = Partial.** The "use a pretrained model" path is broken end-to-end (dead checkpoint URL + linkless model zoo + unloadable weights), and the inference example stops before `postprocess()`.
- **BYO model on pimm:** Dataset/config/hook stories are **good**; the **custom-model story is missing** — no "bring your own model" page, no custom-model tutorial, and the `forward()`→`loss` contract lives only as prose (the API base class is a "placeholder" stub).
- **Findability:** Structurally **sound** (0 broken internal links, API pages populated, clean rendering). Main weakness: **downloadable models are undiscoverable**, plus scattered reference info and missing UX chrome (version switcher, breadcrumbs).

---

## 🔴 Verified broken (do these first) — ✅ ALL RESOLVED 2026-06-25

These were confirmed first-hand, not inferred. **All three are now fixed**, and the
underlying checkpoint infra was published by the maintainer mid-session. See the
**Progress log** below for what changed.

1. **Dead warm-start checkpoint, hardcoded in both tutorials.**
   `hf://youngsm/sonata-pilarnet-L/model_best.pth` returns **HTTP 401** to anonymous users (personal repo — private or gone). Appears as a literal copy-paste command in `tutorials/panda_detector.md`, `tutorials/byo_dataset_semseg.md`, `models/index.md`, `checkpoints/huggingface.md`. It is also the only concrete weight URI in the whole site. → *You can verify instantly; it's your repo.*

2. **Model zoo links nothing, and the real weights can't be loaded.**
   `reference/model_zoo.md` lists ~40 architectures with **no repo IDs / download links / `from_pretrained` snippets**. The real public weights at `huggingface.co/DeepLearnPhysics/Panda` (4 `.pth`, ~1.55 GB) are undiscoverable from the docs and ship with a **109-byte README and no `config.json`**, so `from_pretrained` can't reconstruct them.

3. **The "60-second sanity check" secretly needs 168 GB.**
   `installation.md` calls the verify step "tiny **synthetic** limits" and `quickstart.md` implies `max_len` avoids needing data — both false. Only `first_run.md` admits PILArNet-M is required first. New users hit a "no .h5 files" failure with no on-page explanation.

---

## Progress log — 2026-06-25 (session 1)

**All P0s landed** (docs edits), and the checkpoint infra was published by the maintainer mid-session. Every claim was verified against `pimm/`, `configs/`, and the live HF repos before writing.

- **Bonus P0 the live-site audit couldn't see — a config that doesn't exist.** `semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft` was used as *the* example across `first_run` (whole page), `quickstart`, `model_zoo`, `cli`, `configuration`, `hpc/resuming`, `models/index`, `byo_dataset_semseg`, `huggingface` (19 refs in 9 files) — **no such file exists**. Replaced everywhere with the real `…-ft-5cls-fft`.
- **Real checkpoints published** at `hf://deeplearnphysics/panda-{base,semantic,particle,interaction}` — all consolidated exports (`model.safetensors` + `training_config.json`), so `from_pretrained` loads them directly. Wired into the model zoo (real table + load snippets) and the Panda inference recipe. Resolves audit #1/#2/#3 and original **P0-3** (maintainer-side).
- **Terminology:** "warm-start" → "fine-tune" site-wide (user preference), preserving `warmup`/`warm_up`.

**Source-verified corrections to the raw reports:**
- `from_pretrained` reads **`training_config.json`**, not `config.json` (the audit's filename was wrong); it does not fetch a bare `.pth` from the Hub.
- The published detectors are **`detector-v4`**, not the `detector-v1m1` the tutorial trains.
- The fine-tune configs' `CheckpointLoader` remaps `student.backbone → backbone` (SSL layout), so the tutorials' fine-tune examples correctly target *a Sonata SSL checkpoint* (kept as a `<your-org>` placeholder); the published `panda-base` is a bare `PT-v3m2` encoder used for inference / as a starting point.

**Detector v4 migration (done this session):**
- Created `configs/panda/panseg/detector-v4-pt-v3m2-ft-{pid,vtx}-{dec,fft,scratch}.py` (6 configs) by porting the v1m2 (pid) / v1m1 (vtx) recipes to `detector-v4` — same behavior, no new heads (no momentum/charge/vertex/filter). The instance/segment keys are set explicitly via `label_configs` (no default `segment_particle→segment_pid` fallback), so the configs are self-documenting. Validated: resolve, and `param_dicts`/`max_lr` group lengths match.
- The Panda tutorial + model zoo now use `detector-v4`; **all detector-`v1` mentions removed from the docs** (v1 deprecated). The v1 config *files* are left on disk — delete them on request.
- Resolved earlier open items: inference loads the released `detector-v4` `panda-particle`; fine-tune examples target a Sonata SSL checkpoint (matches the `student.backbone` remap), while `panda-base` (a `PT-v3m2` encoder export) is for inference / as a starting point.

---

## Findings by question

### Is it understandable and followable? (newcomer)
Comprehension is roughly right but under-grounded: no concrete input→output ("3D hits in → per-point shower/track/Michel out"), and **"built on Pointcept" is never explained or linked**. The real wall is **missing prerequisites + bad sequencing** — no statement of Linux + NVIDIA GPU (CUDA 12.4) + ~168 GB disk + slow conda build; the "recommended" container path assumes apptainer with no `docker run` fallback; config names are cryptic with no in-product discovery (`pimm` has no `--list`).

### Is it easy to get the info you want? (findability)
Reference lookups mostly succeed (CLI, resume, HF upload, eval = obvious). Gaps: pretrained downloads (**not found**), and **fragmented** answers for env vars (3 pages), the transform catalog (only 9 of 48 in prose), and "add your own HPC site" (no schema). Structure: **reading order is inverted** (ops sections before Datasets/Models), no version switcher (`/latest/` 404s), no breadcrumbs, no "edit this page".

### Is it easy to apply your dataset to Panda? (dataset → existing model)
Worst-served relative to importance. Beyond the 3 broken items: the **inference recipe is incomplete** (no `postprocess()`, output schema undocumented; no `pimm predict`), the **data contract is scattered across 4+ pages**, the **PID scheme (`pid_6cls`) is buried** in one autodoc page (with "led" vs "other" naming drift), the **Panda tutorial isn't standalone** (backbone config only exists in the *other* tutorial), and whether **`momentum` labels are required** is never stated.

### Is it easy to build on pimm with your own model? (substrate)
Dataset = Yes, Config = Yes, Hook = Yes, **Model = No**. No `models/bring_your_own.md`, no custom-model tutorial, and the crux — `forward()` inputs/return + how `criteria` becomes `loss` — exists only as prose on `concepts.md`; the API contradicts it (stock PyTorch `forward` docstring) and `PointModel` is a documented "placeholder." Also: no canonical packed-batch schema with **dtypes / required-vs-optional**, no custom-loss/custom-evaluator examples, and two confusing autodoc URL namespaces.

---

## Prioritized fix tracker

Check items off as we land them. Tags: **[docs]** = pure doc edit · **[code]** = source/docstring change · **[infra]** = HF repo / dataset / CI work. "Source(s)" = which raw report(s) flagged it.

### P0 — broken or blocks the core job

- [x] **P0-1** Dead `youngsm/sonata-pilarnet-L` checkpoint — **done**: all refs genericized to a `<your-org>` Sonata-SSL placeholder (correct for the `student.backbone` remap); real published checkpoints now referenced for inference. **[docs]**
- [x] **P0-2** Model zoo actionable — **done**: real "Pretrained checkpoints" table (`deeplearnphysics/panda-*`) + `from_pretrained` snippets + fine-tune-from-encoder pointer; fixed the bogus example config path. **[docs]** `reference/model_zoo.md`
- [x] **P0-3** Re-export Panda weights as `from_pretrained`-loadable — **done by maintainer**: published as consolidated exports (`model.safetensors` + `training_config.json`). **[infra]**
- [x] **P0-4** Honest first-run wording — **done**: "synthetic" wording removed; install/quickstart/first_run state the GPU + PILArNet-M (v1) requirement; first_run v2→v1 fixed. *(A true data-free smoke test still needs a code change — flagged, not built.)* **[docs]**
- [x] **P0-5** "Bring your own model" guide — **done**: new `models/bring_your_own.md` (grounded in `default.py`), wired into nav. **[docs]**
- [x] **P0-6** Inference recipe — **done**: `panda_detector.md` §7 now shows build-batch → `forward(return_point=True)` → `postprocess()` → per-point read-out (loads the real `panda-particle`); `models/index.md` snippet fixed. **[docs]**
- [x] **P0-7 (new)** Bogus config name `…-5cls-enc-upcast-fft` (19 refs, 9 files) → real `…-5cls-fft`. **[docs]**

### P1 — significant friction

- [ ] **P1-1** Add a **Prerequisites / Before you start** box (Linux, NVIDIA GPU + CUDA 12.4, disk budget incl. 168 GB, conda build time); add a `docker run --gpus` recipe + one line on getting apptainer. **[docs]** — `getting_started/installation.md` — *src: 1*
- [ ] **P1-2** Publish **one canonical packed-batch schema** (key, shape, **dtype**, required/optional, 2D vs 3D, supervised/instance/panoptic, device), consolidating the 4 scattered descriptions. **[docs]** — `datasets/packed_format.md` (canonical) + cross-link from `concepts`, `models/dataset_format`, `transforms` — *src: 2,3,4*
- [ ] **P1-3** Put the **model contract in the API**: real `forward()` input/return docstrings on `DefaultSegmentor(V2)`; turn `PointModel` (or a new `BaseModel`) from "placeholder" into the documented base. **[code]** — `pimm/models/...` docstrings — *src: 3*
- [ ] **P1-4** Surface the **`pid_6cls` table** on the data-format + Panda-tutorial pages; reconcile "led" vs "other"; state the instance ignore/background value. **[docs]** — `models/dataset_format.md`, `tutorials/panda_detector.md`, `datasets/pilarnet.md` — *src: 2*
- [ ] **P1-5** Make the **Panda tutorial standalone** — inline the full PTv3 backbone config (or an explicit `_base_` import). **[docs]** — `tutorials/panda_detector.md` — *src: 2*
- [ ] **P1-6** State whether **`momentum` labels are required** for `detector-v1m1`; show an energy-only variant. **[docs]** — `tutorials/panda_detector.md`, `models/dataset_format.md` — *src: 2*
- [ ] **P1-7** **Reorder the sidebar** so Datasets + Models precede the ops sections (distributed/config/hooks/checkpoints/eval/hpc). **[docs]** — `index.md` toctree + `conf.py` — *src: 4*
- [ ] **P1-8** Consolidate **environment variables** into one reference table; cross-link from the scattered pages. **[docs]** — new/expanded `reference/` page — *src: 4*
- [ ] **P1-9** Add a **categorized transforms catalog** (all 48) instead of the ~9 named in prose. **[docs]** — `datasets/transforms.md` — *src: 4*
- [ ] **P1-10** Add an **"add your own HPC site"** guide (the `launch/sites/*.yaml` schema + worked example). **[docs]** — `hpc/sites.md` — *src: 4*
- [ ] **P1-11** Add a **version switcher** (and/or publish `/latest/`) given the "APIs may change" banner. **[infra/docs]** — `conf.py` / deploy — *src: 4*
- [ ] **P1-12** Make configs discoverable: a **config catalog** + "how to read a config name" key (and/or a `pimm configs` lister). **[docs(+code)]** — `reference/cli.md` / `reference/model_zoo.md` — *src: 1*
- [ ] **P1-13** Document a **non-source-editing BYO path** (reader/model registration that survives resume without editing `pimm/.../__init__.py`). **[docs(+code)]** — `datasets/bring_your_own.md`, new model guide — *src: 2,3*
- [ ] **P1-14** Add a **custom-evaluator** example (read `comm_info["model_output_dict"]`/`["input_dict"]`, all_gather, write `current_metric_*`). **[docs]** — `evaluation/index.md` — *src: 3*

### P2 — polish

- [ ] **P2-1** Pick **one canonical "first command"** and reuse verbatim everywhere; reconcile **"three vs four ideas."** **[docs]** — getting_started/* — *src: 1*
- [ ] **P2-2** Define **"Pointcept"** (with link) on first use; expand the **glossary** (LArTPC, panoptic, SSL, DDP, FSDP, requeue, world-size). **[docs]** — `index.md`, `getting_started/concepts.md` — *src: 1*
- [ ] **P2-3** Reframe the **landing page** as a pitch (concrete before/after figure + "is this for you?"), not a sitemap. **[docs]** — `index.md` — *src: 1*
- [ ] **P2-4** Publish a **tiny sample dataset shard** + `download_pilarnet.py --sample`; wire into the smoke test. **[infra]** — `scripts/`, `datasets/pilarnet.md` — *src: 1*
- [ ] **P2-5** Unify the **`api/generated/` vs `api/registry/generated/`** URL namespaces (or cross-link). **[infra/docs]** — `conf.py` / `gen_api.py` — *src: 3*
- [ ] **P2-6** Fix **registry API page descriptions** (kill repeated "Base class for all neural network modules"); add inline `__init__` signatures or accurate one-liners. **[code/docs]** — `api/registry/*` generation — *src: 3*
- [ ] **P2-7** Add **custom-loss** and **custom-transform** authoring examples (decorator + `forward`/`__call__` signature; import-for-resume note on transforms). **[docs]** — `datasets/transforms.md`, new losses guide — *src: 3*
- [ ] **P2-8** Remove **duplicate H1s** (`getting_started/index`, `models/dataset_format`, `hpc/monitoring`); fix inline math rendering on `dataset_format`. **[docs]** — *src: 4*
- [ ] **P2-9** Add **breadcrumbs** and an **"Edit this page"** link. **[docs]** — `conf.py` theme options — *src: 4*
- [ ] **P2-10** Resolve **command-style asymmetry** (`pimm launch` vs `scripts/test.sh`; consider `pimm test`) and the **dataset-count mismatch** (registry 5 vs prose more). **[docs(+code)]** — `reference/cli.md`, `datasets/builtin_datasets.md` — *src: 4*
- [ ] **P2-11** Add an inline **outbound link to the HF org** on the HuggingFace page; flesh out `datasets/builtin_datasets.md` (`PILArNetH5Dataset` constructor + ready detector config) and `api/loading.html` stub. **[docs]** — *src: 2,4*

---

## Cross-agent reconciliations (so we trust the list)

- **Panda HF repo "empty" vs "has weights":** report 2 said empty README; report 4 verified the repo **does** ship 4 `.pth` weights (~1.55 GB) but the README is ~109 bytes with no `config.json`. Both agree: **weights exist, metadata/usable-export does not** (P0-3).
- **Breadcrumbs:** one early read thought breadcrumbs existed; report 4 confirmed **none anywhere** (it mistook the sidebar tree).
- **`youngsm/sonata-pilarnet-L` 401:** report 2 verified via the HF API (401 for both page and API) while `DeepLearnPhysics/Panda` returned public JSON — so it's a real access problem, not a fetch artifact (P0-1).
- **`bring_your_own` "wrong base path" link:** report 3 flagged a possibly-wrong cross-ref base URL but rated it *likely a fetch artifact*; report 4's link audit found **0 broken internal links**. Treat as **not broken** but worth a 10-second manual glance.

## Strengths to preserve (don't "fix" these)

- **Link integrity is excellent** — 0 broken internal links across 30+ checked; all external links (repo, Pointcept, HF org, 6 arXiv papers) live.
- **API leaf pages render real signatures** with `[source]` links; no raw autodoc/MyST directives leaking anywhere.
- **Three standout pages:** `getting_started/first_run.md` (real commands + expected output + no-GPU/no-data notes), `getting_started/concepts.md` (mental model + glossary), `configuration/index.md` (the richest, most complete page on the site).
- Substantial single-page sections (distributed, configuration, evaluation) are fine as-is — not stubs.

The throughline: **the "use a pretrained model" path is broken end-to-end**, and the **two most important authoring stories** ("Panda on my data", "my model on pimm") each **stop one step short** of being runnable from the docs alone.
