# Running on scientific computing hardware

pimm is designed for HPC from the ground up. The launcher composes a small stack
of YAML layers into a Slurm job through
[submitit](https://github.com/facebookincubator/submitit), renders the exact
script before you submit, and handles containers, requeue chaining, and exact
resume for you.

- **Interactive vs batch** — salloc live runs vs queued submitit jobs (see below).
- {doc}`Sites & env <sites>` — S3DF, NERSC, containers, environment variables.
- {doc}`Chaining & QOS <chaining>` — requeue chains, walltime, accounts, partitions.
- {doc}`Monitoring <monitoring>` — logs, W&B, Slurm introspection.
- {doc}`Resuming <resuming>` — resume a submitted run and automatic requeue chains.

## The launcher model

`pimm launch` runs locally or inside an existing allocation; `pimm submit`
submits a managed Slurm job. Both load settings in this order, last wins:

```text
1. launch/defaults.yaml              # common defaults
2. launch/sites/<site>.yaml          # site profile (paths, account, container)
3. --recipe PATH                     # optional saved run recipe
4. CLI flags + post-`--` overrides   # final say
```

Python configs under `configs/` remain the source of truth for *what* you train.
Launch YAML describes *execution*: site paths, Slurm resources, container
runtime, checkpoint weights, resume, run naming, environment, and explicit
training overrides. Keep them separate — it's what lets one recipe move between
sites.

:::{important}
**Always `--dry-run` before submitting** when you change site, resources,
account, or recipe. For `pimm submit` the dry-run prints the authoritative
submitit manifest (a single YAML document) — check the resource parameters,
account, partition, GPU request, and pre-rendered requeue attempts before the
job hits the queue.
:::

## Interactive vs batch

```{list-table}
:header-rows: 1
:widths: 28 36 36

* -
  - Batch (`pimm submit`)
  - Interactive (`pimm submit --interactive`)
* - How it runs
  - Queued with `sbatch`; returns immediately
  - Blocking `salloc`; training runs live in your terminal
* - Best for
  - Long runs, chaining, overnight jobs
  - Fast-scheduling queues, debugging on real GPUs
* - Survives disconnect
  - Yes
  - No — ends if your shell/SSH drops
* - Chaining
  - Yes (`--chain.jobs N`)
  - No (single-shot; chaining is rejected)
* - QOS
  - site default or `--slurm.qos`
  - usually `--slurm.qos interactive`
```

### Batch submission

```bash
pimm submit --site s3df \
  --resources.nnodes 1 \
  --resources.nproc-per-node 4 \
  --resources.time 00:30:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

### Interactive allocation

`--interactive` grabs a blocking `salloc` and runs training live, reusing the
*exact* rendered launch script a batch run would use (`salloc` allocates, then
`srun` launches one task per node):

```bash
pimm submit --site nersc --interactive --slurm.qos interactive \
  --resources.nnodes 1 --resources.nproc-per-node 4 --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

`interactive` is a normal launch field, so you can pin `interactive: true` and a
`slurm: {qos: interactive}` default in a recipe and reduce it to one flag.
`--dry-run` prints the exact `salloc … srun … bash -lc <script>` command.

## Recipes — reusable execution state

When a launch has state beyond a single config path — a stable run name, custom
Slurm job name, resource overrides, checkpoint weights, resume behavior, W&B
naming, or training overrides — capture it in a `launch/runs/*.yaml` recipe:

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Recipes should stay **portable**. The same recipe submits to a different site
when its paths and resources are valid:

```bash
pimm submit --site nersc --recipe launch/runs/e050_tail.yaml --dry-run
```

## Flag cheat sheet

```{list-table}
:header-rows: 1
:widths: 36 64

* - Setting
  - Flag
* - training config
  - `--train.config panda/pretrain/...`
* - GPUs per node / nodes
  - `--resources.nproc-per-node N` / `--resources.nnodes N`
* - walltime
  - `--resources.time HH:MM:SS`
* - run name
  - `--run.name NAME`
* - Slurm account / partition
  - `--slurm.account ACCT` / `--slurm.partition PART`
* - QOS
  - `--slurm.qos QOS`
* - requeue attempts (chain)
  - `--chain.jobs N`
* - interactive salloc
  - `--interactive` (submit only)
* - run recipe
  - `--recipe PATH`
* - write rendered output
  - `--output PATH`
* - dry run
  - `--dry-run`
* - training overrides
  - after `--`, as bare `KEY=VALUE`
```

## A note on the editable install

Because `pimm` is a console-script entry point, **every login/submit host** where
you type `pimm launch`/`pimm submit` needs the checkout installed
(`pip install -e .`) in the active environment. Containerized jobs additionally
bind `paths.repo_root` over `/opt/pimm/src` so the in-image editable install
resolves to *your* checkout. See {doc}`sites`.

## Next

- {doc}`sites` — S3DF, NERSC, containers, and environment variables.
- {doc}`chaining` — walltime, QOS, accounts, and requeue chaining.
- {doc}`monitoring` — watching a run.
- {doc}`resuming` — resume a submitted run and automatic requeue chains.

```{toctree}
:hidden:

sites
chaining
monitoring
resuming
```
