# Training on a cluster

The launcher composes a small stack
of YAML layers into a Slurm job through
[submitit](https://github.com/facebookincubator/submitit), renders the exact
script before you submit, and handles containers, requeue chaining, and exact
resume for you.

- **Interactive vs batch** - salloc live runs vs queued submitit jobs (see below).
- {doc}`Sites & env <sites>` - site profiles (including your own cluster), containers, environment variables.
- {doc}`Chaining & QOS <chaining>` - requeue chains, walltime, accounts, partitions.
- {doc}`Monitoring <monitoring>` - logs, W&B, Slurm introspection.
- {doc}`Resuming <resuming>` - resume a submitted run and automatic requeue chains.

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
training overrides. Keep them separate - it's what lets one recipe move between
sites.

:::{important}
**Always `--dry-run` before submitting** when you change site, resources,
account, or recipe. For `pimm submit` the dry-run prints the authoritative
submitit manifest (a single YAML document) - check the resource parameters,
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
  - No - ends if your shell/SSH drops
* - Chaining
  - Yes (`--chain.jobs N`)
  - No (single-shot; chaining is rejected)
* - QOS
  - site default or `--slurm.qos`
  - usually your cluster's interactive QOS via `--slurm.qos`
```

```bash
uv run pimm submit --site mycluster \
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
uv run pimm submit --site mycluster --interactive --slurm.qos <qos> \
  --resources.nnodes 1 --resources.nproc-per-node 4 --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

QOS names are cluster-specific, so substitute the interactive QOS your cluster provides.
`interactive` is a normal launch field, so you can pin `interactive: true` and a
`slurm: {qos: <qos>}` default in a recipe and reduce it to one flag.
`--dry-run` prints the exact `salloc … srun … bash -lc <script>` command.

## Recipes - reusable execution state

When a launch has state beyond a single config path - a stable run name, custom
Slurm job name, resource overrides, checkpoint weights, resume behavior, W&B
naming, or training overrides - capture it in a `launch/runs/*.yaml` recipe:

```bash
uv run pimm submit --site mycluster --recipe launch/runs/e050_tail.yaml \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Recipes should stay **portable**. The same recipe submits to a different site
when its paths and resources are valid:

```bash
uv run pimm submit --site slurm --recipe launch/runs/e050_tail.yaml --dry-run
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

## Launcher environment

Every login or submit host where you invoke `pimm launch` or `pimm submit`
needs the small launcher environment.
Run `./install.sh --launcher-only` once, then invoke commands with
`uv run pimm submit ...`.
The training environment remains inside the compute-node image.
See {doc}`sites`.

```{toctree}
:hidden:

sites
chaining
monitoring
resuming
```
