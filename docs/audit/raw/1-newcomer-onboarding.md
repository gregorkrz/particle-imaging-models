# Raw report 1 — Newcomer onboarding UX audit

> **Persona:** 2nd-year physics PhD student on a neutrino (LArTPC) experiment. Knows Python + basic PyTorch, has trained a small single-GPU model, but has never used Pointcept, point-cloud DL, sparse convs, or HPC schedulers. Found the site from a paper.
> **Method:** Browsed the **live rendered site** only (`https://deeplearnphysics.org/particle-imaging-models/stable/`); no access to `docs/` source.
> **Date:** 2026-06-25 · **Resume this agent via SendMessage:** `acf416ce1f2f7082e`

---

# pimm docs — first-time-user UX audit

## 1. Journey log (URLs actually visited, in order)
1. https://deeplearnphysics.org/particle-imaging-models/stable/ (landing)
2. https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/index.html
3. https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/installation.html
4. https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/quickstart.html
5. https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/first_run.html
6. https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/concepts.html
7. https://deeplearnphysics.org/particle-imaging-models/stable/datasets/pilarnet.html
8. https://deeplearnphysics.org/particle-imaging-models/stable/datasets/index.html
9. https://deeplearnphysics.org/particle-imaging-models/stable/tutorials/index.html
10. https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html
11. https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M (external, clicked from pilarnet.html)
12. https://deeplearnphysics.org/particle-imaging-models/stable/reference/cli.html
13. https://deeplearnphysics.org/particle-imaging-models/stable/reference/model_zoo.html

## 2. Verdict
**Not today, and not on the path the site pushes me down.** The docs technically contain every piece I need, but they are mis-sequenced and internally contradictory, so a newcomer following the landing-page "Get started" button → Installation → Quickstart will hit a wall. To actually run something I need (a) a Linux box with an NVIDIA GPU + CUDA 12.4, (b) a successful heavy conda build *or* apptainer + a container pull, and (c) a **168 GB** dataset download — none of which are stated as prerequisites on the install or quickstart pages. The one page that handles all this honestly (`first_run.html`) is third in the sequence and trivially skipped.

**Single biggest blocker:** There is **no data-free or small-data first run.** Every "first" command (install "Verify", quickstart "60-second sanity check", first_run step 3) actually requires the 168 GB PILArNet-M dataset, yet two of those three pages explicitly imply it does *not*. I would copy the "60-second sanity check," run it, get a "no .h5 files found"–style failure, and have no explanation on that page.

## 3. Comprehension check
The one-sentence takeaway I'd walk away with after the landing page: *"pimm is a PyTorch framework (built on something called Pointcept) for training point-cloud deep-learning models — self-supervised pretraining and semantic/panoptic segmentation — on neutrino/particle-imaging detector events, with distributed/HPC training built in."*

Is that correct/sufficient? **Roughly correct, but under-grounded.** It never shows the concrete input→output (3D detector hits in → per-point labels like shower/track/Michel out), so I understand the *what* abstractly but can't quickly decide *"is this for my reconstruction task?"* And the load-bearing phrase **"built on Pointcept"** is never explained or linked anywhere in the docs — I'd have to Google it.

## 4. Gaps & friction (the meat)

**[P0] — installation.html / quickstart.html / first_run.html — "no data-free smoke test," stated inconsistently.**
- installation.html "Verify the install": *"runs a couple of steps on tiny **synthetic** limits"* — "synthetic" strongly implies no dataset needed. It's false: these are tiny *limits on real data*.
- quickstart.html: *"It assumes you have installed pimm and **have PILArNet-M data** (or are happy to use tiny `max_len` limits for a quick check)"* — the parenthetical implies `max_len` avoids needing data. Also misleading.
- Only first_run.html step 3 tells the truth: *"No PILArNet-M data yet? See step 7 to fetch some, or point `PILARNET_DATA_ROOT_V2` at your copy."*
- Why it blocks me: on the obvious install→quickstart path I never see the truth; I just get a failure. **Fix:** add a real data-free smoke test (random/synthetic tensors that build model+trainer and run 2 steps), and make it THE verify command on all three pages. If impossible, put a bold "Requires PILArNet-M downloaded first (168 GB) — see step 7" admonition directly above every "first run" command.

**[P0] — datasets/pilarnet.html + HF dataset page — 168 GB, all-or-nothing, gates the entire first run.**
- *"The 168 GB dataset is hosted on Hugging Face."* Downloader only supports `--version {v1,v2,both}`; the HF page shows no sample/single-shard option. So my first *real* run is gated on a multi-hour, 168 GB download with no small alternative.
- Why it blocks me: as a grad student with finite disk/quota, "run something today" is effectively impossible. **Fix:** publish a tiny sample shard (a few hundred MB) and document `download_pilarnet.py --sample` (or a direct single-file URL); wire it into the smoke test.

**[P1] — installation.html / quickstart.html — GPU requirement never stated upfront.**
Only first_run.html step 3 mentions it: *"No GPU... `torchrun` exits with `RuntimeError: no CUDA devices available`."* Install and quickstart never say "you need an NVIDIA GPU (CUDA 12.4)," and there's no CPU/macOS/Apple-Silicon path at all. **Fix:** a "Requirements" line at the top of Installation: Linux + NVIDIA GPU (CUDA 12.4), no CPU/Mac support.

**[P1] — installation.html — no "Prerequisites / before you start" box; can't assess feasibility.**
Missing entirely: OS (Linux is only implied), GPU + which compute capability, disk budget (168 GB data + multi-GB container + multi-GB conda env), and build-time expectations. **Fix:** a short prerequisites table up front with concrete numbers.

**[P1] — installation.html — "Container (recommended)" assumes apptainer is installed and understood; no Docker *run* path.**
It jumps straight to `apptainer pull ...` and `apptainer exec --nv --bind ...` with no "what is apptainer / how do I get it" (it's usually an HPC module), no link, and `--nv`/`--bind` unexplained. Only `docker build` is shown — never `docker run` — so a workstation user who has Docker but not apptainer has no run recipe. As someone who's never touched apptainer, the *recommended* path is the most opaque one. **Fix:** one sentence + link on getting apptainer; add a `docker run --gpus all ...` example.

**[P1] — installation.html — the first container command requires the full 168 GB dataset.**
The first `apptainer exec ... pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` has **no** `max_len` limits → needs full data. That's a broken first example. **Fix:** make the first container example the limited smoke test.

**[P1] — installation.html vs quickstart.html/first_run.html — two different "first commands."**
Install "Verify" uses `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask`; quickstart + first_run use `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft`. Which is THE first thing to run? **Fix:** pick one canonical first command and reuse it verbatim everywhere.

**[P1] — reference/cli.html — cryptic config names with zero in-product discovery.**
Every example is a string like `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft`, and the CLI page admits *"there is no `--list` long option; browse the `configs/` tree to find configs."* I can copy-paste but can't choose or decode the name (what's `ft`? `5cls`? `upcast`? `fft`?). Model zoo lists architectures, not runnable configs. **Fix:** add a `pimm configs` lister or an in-docs catalog of ~5 starter configs with a "how to read a config name" key.

**[P1] — installation.html — heavy conda build risk undocumented.**
`conda env create -f environment.yml` builds spconv, FlashAttention, and local CUDA extensions (notoriously slow/fragile). No time estimate, no "requires nvcc/CUDA toolkit + ninja," and `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` is unexplained — those are GPU compute capabilities and I don't know mine. **Fix:** note expected build time, prerequisites, and link "how to find your GPU's arch."

**[P1] — index.html + concepts.html — jargon fired before it's defined; glossary misses the physics/ML acronyms.**
The landing page's first two paragraphs throw Pointcept, panoptic, SSL, DDP, FSDP2, requeue, "launch layer," packed tensors, registries — undefined. concepts.html *does* define packed tensors/registries/configs/launch layer well and has a glossary, but it's the 4th nav item and its glossary omits **LArTPC, panoptic, SSL, DDP, FSDP, requeue, world-size, and Pointcept**. "Pointcept," the thing pimm is "built on," is never explained anywhere. **Fix:** link Pointcept on first use; expand the glossary to cover the acronyms; surface a "New here? Read Core concepts first" link on the landing page.

**[P2] — index.html — the page is a giant sitemap, not a pitch.** The left nav dumps the entire API (every MinkUNet variant, every transform — hundreds of links), burying the 5 onboarding links. "What's inside" is a table of contents, not a value proposition; there's no example image or input→output. **Fix:** lead with a concrete before/after figure and a 3-line "is this for you?".

**[P2] — index.html vs concepts.html — "three ideas" vs "four ideas."** Landing says *"the three ideas (packed tensors, registries, configs)"*; concepts.html says *"Four ideas"* (adds the launch layer + trainer contract). Minor but jarring.

**[P2] — reference/model_zoo.html — no actual checkpoints.** *"This is a catalog, not a benchmark table — no performance numbers are quoted here."* No downloadable URIs, so the data-light "load a pretrained model" path is undercut; the only concrete weight URI in the whole docs is `hf://youngsm/sonata-pilarnet-L/model_best.pth`, buried in first_run step 7 / models/index.

**[P2] — models/index.html — inference example references an undefined helper.** The "Running inference" snippet calls `make_packed_batch(event)` which is never defined (defers to "Feeding a model the right data"). So even the data-free inference path isn't copy-paste runnable.

## 5. "I needed this but couldn't find it"
- A "Prerequisites / Before you start" checklist (OS, GPU + CUDA, disk, build time).
- A genuinely data-free smoke test to verify the install without 168 GB.
- A small sample / single-shard dataset download for a first real run.
- "What is apptainer and how do I get it," plus a `docker run` recipe.
- A browsable catalog of runnable starter configs + how to decode a config name.
- A one-line "what is Pointcept" (with link) and a glossary covering LArTPC / panoptic / SSL / DDP / FSDP / requeue / world-size.
- Published pretrained checkpoint URIs in the model zoo.
- VRAM / disk-footprint / real-run-time estimates.

## 6. Broken / confusing links & rendering issues
- **No hard 404s** — every internal doc link I followed resolved.
- **quickstart.html "Build your run command"** appears to be an interactive widget (*"Pick your target, resources, and config — the matching command assembles below, ready to copy"*). Via plain fetch I only see the prompt text, not a functioning builder — worth confirming it actually renders/works in a real browser, because if it's the centerpiece of the Quickstart and it's blank, that's a bounce point.
- **Duplicate sidebar labels** in the API nav: three identical "PointTransformerV2" entries, three "SpUNetBase," two "Sonata," two "Detector," two "PointTransformerSeg26/38/50" — same text, different module URLs. Confusing when scanning.

## 7. Top 5 prioritized recommendations
1. **Ship a real data-free smoke test** (synthetic random tensors) and make it the single canonical "Verify your install" command on Installation, Quickstart, and First runs. This alone converts "can't run today" into a 60-second win.
2. **Publish a tiny sample dataset** (few-hundred-MB shard) with a `download_pilarnet.py --sample` flag, and state the **168 GB** full size + **GPU/CUDA requirement** as bold prerequisites at the top of Installation.
3. **Add a "Prerequisites / Before you start" box** to Installation (OS=Linux, NVIDIA GPU + CUDA 12.4, disk budget, conda-build time) and make the *first* container example the limited smoke test, not a full-data pretrain.
4. **Fix the contradictions and pick one canonical first command** reused verbatim everywhere; replace "tiny synthetic limits" wording, and reconcile "three vs four ideas."
5. **Make configs and models discoverable:** add a config catalog / `pimm configs` lister + a "how to read a config name" key, and put real pretrained checkpoint URIs in the model zoo. Also define **Pointcept** (with link) and expand the glossary to the physics/ML acronyms.

Credit where due: `first_run.html` is excellent (real commands, "what you should see" sample output, explicit no-GPU/no-data notes, dry-run discipline, export round-trip), and `concepts.html` is a strong mental-model page. The core failure is **sequencing and prerequisites**, not content depth — the honest page is buried behind two pages that over-promise a frictionless first run.
