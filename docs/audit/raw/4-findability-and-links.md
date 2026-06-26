# Raw report 4 — Findability, navigation, link integrity & completeness audit

> **Persona / lens:** Information-findability and structural-soundness auditor — "is it easy to get the info you want, and is the site sound?" Ran 10 realistic "find X" lookups, navigation/structure checks, link-integrity verification (by fetching), API-reference health, and rendering/consistency.
> **Method:** Browsed the **live rendered site** only; no access to `docs/` source. 40+ content pages + ~40 verification fetches (incl. HF repos, arXiv, GitHub).
> **Date:** 2026-06-25 · **Resume this agent via SendMessage:** `a2a93038a244344f8`
>
> **Note:** This agent independently **overturned one claim** from report 2 — the HF `DeepLearnPhysics/Panda` repo is **not** empty: it ships four `.pth` weight files (~1.55 GB). (Both agents agree the README is ~109 bytes and there's no `config.json`.)

---

# pimm Documentation Site Audit — `deeplearnphysics.org/particle-imaging-models/stable/`

## 1. Coverage log (URLs fetched)

**Content pages (40 unique, all HTTP 200 unless noted):**
- `index.html`
- `getting_started/{index, installation, quickstart, first_run, concepts}.html`
- `distributed/index.html`, `configuration/index.html`, `evaluation/index.html`
- `hooks/{index, logging, diagnostics, writing_hooks}.html`
- `checkpoints/{index, hooks, export, huggingface, resume_world_size}.html`
- `hpc/{index, sites, chaining, monitoring, resuming}.html`
- `datasets/{index, packed_format, transforms, pilarnet, builtin_datasets, bring_your_own}.html`
- `models/{index, dataset_format}.html`
- `tutorials/{index, byo_dataset_semseg, panda_detector}.html`
- `reference/{index, cli, model_zoo}.html`
- `api/{index, loading, core}.html`, `api/registry/{models, datasets, transforms, hooks, losses, trainers}.html`

**API leaf (generated/*) pages spot-checked for signature rendering:** `pimm.from_pretrained`, `pimm.save_pretrained`, `pimm.push_to_hub`, `pimm.export.load_pretrained`, `pimm.utils.checkpoints.CheckpointManager`, `pimm.utils.config.Config`, `pimm.models.utils.structure.Point`, `pimm.datasets.builder.build_dataset`, `...GridSample`, `...SemSegEvaluator`, `...PushToHub`, `pimm.models.litept.litept.LitePT`, `pimm.models.losses.misc.BinaryFocalLoss`.

**Infra/utility URLs:** `search.html` (OK), `searchindex.js` (OK), `genindex.html` (OK), `py-modindex.html` (**404**), `/latest/index.html` (**404**), `/particle-imaging-models/index.html` (stub→stable), `/particle-imaging-models/` (stub→stable), `/stable/` (serves index), `this_page_does_not_exist.html` (**404**).

**External:** `github.com/DeepLearnPhysics/particle-imaging-models` (OK), GitHub new-issue form (OK), `github.com/Pointcept/Pointcept` (OK), `huggingface.co/DeepLearnPhysics` (OK), `huggingface.co/DeepLearnPhysics/Panda/tree/main` (OK, has weights), `huggingface.co/datasets/DeepLearnPhysics/PILArNet-M` (OK), arXiv 2312.10035, 2210.05666, 2012.09164, 2503.16429, 2512.01324, 2502.02558 (all OK).

That is **40+ content pages plus ~40 verification fetches** — well beyond the requested floor.

---

## 2. Findability scorecard (Section A lookups)

| # | Question | Found? | Best URL | Discoverability | Note |
|---|----------|--------|----------|-----------------|------|
| 1 | Env vars pimm reads | Yes | `getting_started/installation.html` | **Dig** | Canonical table here (`PILARNET_DATA_ROOT_V1/2/3`, `MODEL_DIR`, `WANDB_API_KEY`) but distributed vars (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`…) on `distributed/index.html` and data-root fallback on `datasets/pilarnet.html`. No consolidated page. |
| 2 | CLI commands & flags | Yes | `reference/cli.html` | **Obvious** | Real reference: `pimm launch`/`submit`/`export` with flags, plus `scripts/train.sh`/`test.sh`. Linked widely. |
| 3 | Pretrained models + where to download | **Partial** | `reference/model_zoo.html` | **Not found (for downloads)** | Catalog of ~40 architectures but **zero download links / repo IDs / `from_pretrained` snippets**; page body doesn't even link the HF org. Weights actually exist at `huggingface.co/DeepLearnPhysics/Panda` — undiscoverable from the zoo. |
| 4 | Resume after kill/preemption | Yes | `hpc/resuming.html` | **Obvious** | "Resume is **exact**…restores model, optimizer, scheduler, AMP scaler, trainer progress, the stateful dataloader position, and RNG — per rank." Also in `quickstart.html`. |
| 5 | Loss functions available | Yes | `api/registry/losses.html` | **Obvious-ish** | 14 losses enumerated (FocalLoss, DiceLoss, LovaszLoss, …). Only in API section, not surfaced from a guide page. |
| 6 | Upload checkpoint to HF | Yes | `checkpoints/huggingface.html` | **Obvious** | First H2 is the `PushToHub` hook; also `push_to_hub()`. Dedicated sidebar item. |
| 7 | HPC sites + add my own | **Partial** | `hpc/sites.html` | **Dig / Not found (for "add mine")** | 3 sites named (`local`, `s3df`, `nersc`); shows `launch/sites/*.yaml` layout but **no instructions/schema for authoring a new site**. |
| 8 | Transforms for augmentation | **Partial** | `datasets/transforms.html` + `api/registry/transforms.html` | **Dig** | Prose names ~9; the full 48-transform catalog (ElasticDistortion, SphereCrop, RandomJitter…) exists only as API-registry links, no prose enumeration. |
| 9 | Run final evaluation/testing | Yes | `evaluation/index.html` | **Obvious** | `sh scripts/test.sh -c <config> -n <name> -w model_best`; also `reference/cli.html` and the BYO tutorial step 7. |
| 10 | Exact packed-tensor format | **Partial** | `datasets/packed_format.html` | **Found but incomplete** | Logical contract concrete (`coord`/`feat`/`offset`/`segment` shapes; "offset is the cumulative sum of the per-event point counts"). **Missing dtypes, byte layout, required-vs-optional fields.** |

---

## 3. Navigation & structure findings

- **"Thin lone index" suspicion — mostly FALSE.** `distributed/index.html` (~1,200 words, 10 sections incl. DDP/FSDP2 table), `configuration/index.html` (~2,800 words — the richest page on the site), and `evaluation/index.html` (~1,200–1,500 words) are substantial single-page sections, not stubs. They simply have no child pages in the sidebar (unlike Hooks/Checkpoints/HPC/Datasets). Verdict: intentional single-page sections, acceptable.
- **Reading-order problem (P1).** Sidebar/pager order runs: getting_started → **distributed → configuration → hooks → checkpoints → evaluation → hpc** → datasets → models → tutorials → reference → api. A newcomer reading top-to-bottom hits advanced ops (distributed training, configuration internals, hooks, HPC) **before** ever reaching Datasets and Models. The natural flow (data → model → train → eval → scale) is inverted. `getting_started/concepts.html` partly mitigates this.
- **Orphans / guess-only pages:** None of substance. All content pages are reachable via sidebar + pager. `py-modindex.html` is the only generated page that 404s, and it isn't linked from content (the linked index, `genindex.html`, works).
- **No breadcrumbs** anywhere (verified first-hand on `getting_started/index.html` and `reference/model_zoo.html`: "No breadcrumb trail… The page begins directly with the 'Model zoo' heading"). Hierarchy is conveyed only by the left sidebar. (One subagent's "breadcrumbs present" reading was a false positive — it mistook the sidebar tree for breadcrumbs.)
- **Prev/next pager:** present and correct site-wide (e.g., model_zoo prev=CLI reference, next=API reference).
- **No version switcher** despite the `/stable/` path. `/latest/` returns **404**; bare root `/particle-imaging-models/` serves a stub that forwards to `/stable/`. Single published version, no UI to switch.
- **No "Edit on GitHub"/"View source"** affordance; the only contribute path is a footer "Open issue" link (pre-filled GitHub issue template per page) + a "Repository" link.

---

## 4. Broken / suspect links (verified)

Internal link integrity is **strong**: across a sample of 30+ distinct internal targets (cross-section `{doc}` links and `api/registry/*` links) verified across the six section audits, **zero 404s and zero wrong-anchors**. Same-page anchors (e.g., `losses.html#classification`, `resuming.html#mid-epoch-resume`) were cross-checked against destination section IDs and all matched.

| Source / context | Link target | Type | Status |
|---|---|---|---|
| Sample of ~30 internal cross-links (losses, transforms, loading, core, dataset_format, packed_format, huggingface, resuming, evaluation, configuration, distributed, cli, model_zoo, registry pages, generated leaf pages) | various `…/stable/...html` | internal | **OK** (all 200, correct page) |
| `api/index.html` → 6 registry pages + loading + core | `registry/*.html`, `loading.html`, `core.html` | internal | **OK** (all resolve, unique content) |
| Header nav (all pages) | `huggingface.co/DeepLearnPhysics` | external | **OK** (org; 1 model + 1 dataset) |
| `index.html` | `github.com/DeepLearnPhysics/particle-imaging-models` | external | **OK** |
| `index.html` ("built on Pointcept") | `github.com/Pointcept/Pointcept` | external | **OK** (note: Pointcept link is on `index.html`, *not* on datasets/models/concepts pages) |
| `index.html` + `reference/model_zoo.html` | 6 arXiv papers (2312.10035, 2210.05666, 2012.09164, 2503.16429, 2512.01324, 2502.02558) | external | **OK** (all live) |
| `datasets/pilarnet.html` | `huggingface.co/datasets/DeepLearnPhysics/PILArNet-M` | external | **OK** (168 GB, real) |
| (corrected) `huggingface.co/DeepLearnPhysics/Panda/tree/main` | 4× `.pth` (panda_base/interaction/particle/semantic, ~1.55 GB) | external | **OK — weights present** (README only 109 B, no config.json) |
| `genindex.html`, `search.html`, `searchindex.js` | — | internal | **OK** |
| `py-modindex.html` | — | internal | **404** (no module index generated; not linked from content) |
| `/latest/index.html` | — | internal | **404** (only `/stable/` exists) |
| `this_page_does_not_exist.html` | — | internal | **404** (proper status; no custom 404 page evidenced) |

**Suspect-but-not-broken:** the model card at `huggingface.co/DeepLearnPhysics/Panda` is essentially undocumented (109-byte README, no `config.json`) — weights download fine but there is no usage doc on the Hub, and the docs' model zoo never links to it.

---

## 5. API reference health

**Healthy top-to-bottom.** No empty/broken/stub pages, no unrendered autodoc directives.

| Page | Verdict | Evidence |
|---|---|---|
| `api/index.html` | Populated hub | Links to all 6 registries + loading + core; every link verified OK. |
| `api/loading.html` | Populated (index) | ~10 functions (`from_pretrained`, `save_pretrained`, `push_to_hub`, `remap_state_dict_keys`, …) with summaries; signatures one click deeper. |
| `api/core.html` | Populated (index) | ~23 entries (`build_model`, `build_dataset`, `Point`, `Registry`, `CheckpointManager`, `TrainState`, distributed helpers…). |
| `api/registry/models.html` | Populated | 60 classes/61 names (PTv2/v3 families, MinkUNet14–101, Sonata, PoLArMAE, VoltMAE, Detector, PointGroup, LoRAAdapter…). |
| `api/registry/transforms.html` | Populated | 48 classes across 6 categories. |
| `api/registry/hooks.html` | Populated | ~26 hooks across 8 categories. |
| `api/registry/losses.html` | Populated | 14 losses (Classification/Segmentation/Regression/Instance). |
| `api/registry/datasets.html` | Populated (small) | 5 (`ConcatDataset`, `DefaultDataset`, `JAXTPCDataset`, `PILArNetH5Dataset`, `LUCiDDataset`). |
| `api/registry/trainers.html` | Populated (small) | 5 (`DefaultTrainer`, `MultiDatasetTrainer`, `ImageClassTrainer`, `InsegTrainer`, `GRPOTrainer`). |
| **`generated/*` leaf pages** | **Populated (verified)** | First-hand checks: `from_pretrained` shows full signature + param docs + `[source]` (→ `_modules/pimm/export/api.html#from_pretrained`); `LitePT` shows `class LitePT(in_channels=4, order=('z','z-trans',…), …)` + `forward()` + `[source]`; `BinaryFocalLoss` shows `class BinaryFocalLoss(gamma=2.0, alpha=0.5, logits=True, …)` + params + `[source]`. No raw `:param:`/`.. autoclass::`. |

Caveat: registry/loading/core pages are autosummary **indexes** (names + one-line summaries); real signatures live on `generated/*` leaves — which I confirmed render correctly. This is normal Sphinx structure, not a defect.

---

## 6. Rendering & consistency issues

- **No literal MyST/Sphinx directives** rendered as raw text anywhere across 40+ pages. The one suspected raw ` ```{note} ``` ` on `reference/model_zoo.html` was re-checked and is a **correctly rendered admonition** ("This is a catalog, not a benchmark table…") — false positive.
- **Duplicate H1 headings (P2):** confirmed first-hand on `getting_started/index.html` ("Getting started" appears twice at top of main content). Also reported on `models/dataset_format.html` (H1 == first H2 "Feeding a model the right data") and `hpc/monitoring.html` (H1 "Job monitoring" repeated as H2).
- **Inline math as plain text** on `models/dataset_format.html`: `768 * sqrt(3) / 2 ≈ 665.1` rendered as Unicode, not LaTeX. Cosmetic.
- **Duplicate logo SVG** (header + sidebar) on every page — theme behavior, harmless.
- **Search:** box present ("Search Ctrl+K"); `search.html` + `searchindex.js` + `genindex.html` all exist. Caveat: client-side JS search can't be exercised via fetch, but the infrastructure is in place.
- **Consistency frictions:**
  - **Command style:** training is documented as `pimm launch` (and `scripts/train.sh`), but final testing is only `scripts/test.sh` — there is **no `pimm test` subcommand** in `reference/cli.html`. Asymmetric.
  - **Dataset count mismatch:** `api/registry/datasets.html` lists 5 datasets, while `datasets/builtin_datasets.html` prose names more (LUCiDEventSSLDataset, UBooNEH5Dataset, plus non-auto-imported ones). Minor contradiction.
  - **HF page missing inline outbound link:** `checkpoints/huggingface.html` — the page literally about Hub integration — has no body link to `huggingface.co`; only the global header nav links the org.
  - Site-wide banner "pimm is research software under active development — APIs and configs may change between versions" is the closest thing to a stability disclaimer (not a defect).

---

## 7. Gaps & friction (prioritized)

**P0 — blocks finding info**
- **[P0-1]** `reference/model_zoo.html` — Catalogs ~40 architectures but provides **no download links, no HF repo IDs, no `from_pretrained` examples**, and no body link to the HF org. Real weights exist (`huggingface.co/DeepLearnPhysics/Panda`, 4× `.pth`) but are undiscoverable from the zoo. Directly fails lookup Q3. *Fix:* add a "Weights" column with HF repo IDs + a copy-paste `from_pretrained("DeepLearnPhysics/Panda")` snippet per available checkpoint.

**P1 — significant friction**
- **[P1-1]** `hpc/sites.html` — No instructions/schema for adding your own cluster; only 3 built-in sites shown. *Fix:* document the `launch/sites/*.yaml` field reference + a worked "add your site" example.
- **[P1-2]** Env-var docs fragmented across `installation`, `distributed`, `datasets/pilarnet`. *Fix:* one consolidated "Environment variables" reference table, cross-linked.
- **[P1-3]** `datasets/transforms.html` — No prose catalog of the 48 augmentation transforms (only ~9 named); full set hidden in API registry. *Fix:* add a categorized augmentation catalog table linking to API leaves.
- **[P1-4]** Reading order — Datasets/Models sit after Distributed/Configuration/Hooks/Checkpoints/Evaluation/HPC. *Fix:* reorder sidebar so Datasets + Models precede the ops sections.
- **[P1-5]** No version switcher and `/latest/` 404 for "active development" software. *Fix:* enable a version dropdown (e.g., `sphinx-version-warning`/`mike`-style) or at least publish `/latest/`.

**P2 — polish**
- **[P2-1]** Duplicate H1 on `getting_started/index.html`, `models/dataset_format.html`, `hpc/monitoring.html`.
- **[P2-2]** No breadcrumbs anywhere — add a breadcrumb trail for orientation in deep `api/registry/generated/*` pages.
- **[P2-3]** No "Edit this page"/"View source" link (only "Open issue").
- **[P2-4]** `datasets/packed_format.html` omits dtypes/byte layout/required-vs-optional fields (Q10 partial).
- **[P2-5]** `checkpoints/huggingface.html` lacks an inline outbound link to the Hub/org.
- **[P2-6]** Command-style asymmetry (`pimm launch` vs `scripts/test.sh`; no `pimm test`).
- **[P2-7]** Dataset list inconsistency between `api/registry/datasets.html` (5) and `datasets/builtin_datasets.html` (more).
- **[P2-8]** Panda HF model card is near-empty (109 B, no `config.json`) — add a usage card.
- **[P2-9]** `py-modindex.html` 404 (cosmetic) and no custom 404 page (bare GitHub Pages 404).

---

## 8. Top 5 recommendations

1. **Make the model zoo actionable (P0-1).** Add per-model HF repo IDs, a download/weights column, and `from_pretrained(...)` snippets so Q3 ("what models exist and where do I get them") is answered on-page. The weights already exist on the Hub — just connect them.
2. **Add an "add your own HPC site" guide (P1-1)** with the `sites/*.yaml` schema and a worked example, so the cluster-onboarding path isn't a dead end.
3. **Consolidate environment variables and the transforms catalog (P1-2, P1-3)** into single reference tables, eliminating the "scattered across 3 pages" / "only in API" friction for Q1 and Q8.
4. **Fix the reading order and add a version switcher (P1-4, P1-5):** move Datasets/Models ahead of the ops sections, and expose a version dropdown given the "APIs may change" disclaimer.
5. **Cosmetic cleanup (P2-1, P2-2, P2-3):** remove the duplicate H1s, add breadcrumbs (helps the deep API tree), and add an "Edit this page" link.

**Overall:** structurally sound and content-rich. Link integrity is excellent (no broken internal links found; all external links — repo, Pointcept, HF org, 6 arXiv papers — live), API docs are populated down to leaf signatures, and rendering is clean (no literal directives). The dominant weakness is **findability of downloadable pretrained models** (the zoo doesn't link to the weights that exist), followed by a handful of scattered-info gaps and missing UX chrome (version switcher, breadcrumbs, edit links).
