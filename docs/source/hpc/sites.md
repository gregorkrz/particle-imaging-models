# Sites & environment

A **site profile** (`launch/sites/<site>.yaml`) captures everything
cluster-specific: repo and experiment paths, Slurm account/partition/QOS,
container runtime, bind mounts, the GPU directive flavor, and dataset
environment variables. Choose one with `--site`.

```text
launch/
  defaults.yaml          # common launcher defaults
  sites/
    local.yaml           # no scheduler/container — runs on the current node
    slurm.yaml           # generic Slurm defaults; cluster sites inherit via _base_
    s3df.yaml            # SLAC S3DF
    nersc.yaml           # NERSC Perlmutter
  runs/*.yaml            # portable run recipes (execution state, not architecture)
```

## `local` — your laptop or a single node

The default site for `pimm launch`. No Slurm directives, no container wrapper —
runs in your active environment on the current node:

- `paths.repo_root: .`
- the only env var it sets is `PYTHONFAULTHANDLER=1`
- `resources.cpus_per_proc: 8`
- `resources.nproc_per_node: auto` — uses **all visible GPUs** (omits `-g` so
  `train.sh` auto-detects); set an integer to pin

```bash
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

`pimm launch` is the **local executor** (run on the current node); `pimm submit`
is the **Slurm executor** (batch, or `--interactive` for a live allocation). The
executor follows the verb, *not* the site name — so `pimm launch --site s3df`
runs on the current node **inside the s3df container**, and `pimm submit --site
s3df` queues the same environment on Slurm.

## Add your own site (start from `slurm`)

For a generic Slurm cluster, build from the **`slurm`** base — not `s3df` /
`nersc`, which are SLAC/NERSC-specific. Drop a `launch/sites/<name>.yaml` that
inherits it (`--site <name>` resolves to `launch/sites/<name>.yaml`):

```yaml
# launch/sites/mycluster.yaml
_base_: slurm.yaml                 # generic Slurm defaults

site: mycluster

paths:
  repo_root: /home/me/particle-imaging-models   # shared checkout jobs run from
  exp_root: "{repo_root}/exp"                    # where runs are written

resources:
  nnodes: 1
  nproc_per_node: 4                 # GPUs per node
  cpus_per_proc: 12                 # CPUs per GPU
  time: "12:00:00"

slurm:
  account: my_account
  partition: gpu
  gpu_directive: gres               # `--gres=gpu:N`; some clusters need `gpus-per-node`

container:
  runtime: none                     # or `singularity` with an `image:` (see below)

env:
  PILARNET_DATA_ROOT_V1: /data/pilarnet/v1      # cluster-wide data roots
```

```bash
pimm submit --site mycluster --train.config <config> \
  --resources.nnodes 1 --resources.nproc-per-node 4 --dry-run
```

The keys a site profile understands:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Section
  - Keys
* - `paths`
  - `repo_root` (checkout jobs run from), `exp_root` (run outputs; default `{repo_root}/exp`)
* - `resources`
  - `nnodes`, `nproc_per_node` (GPUs/node), `cpus_per_proc` (CPUs/GPU), `time`, `mem`
* - `slurm`
  - `account`, `partition`, `qos`, `constraint`, `gpu_directive` (`gres` or `gpus-per-node`), `output`
* - `container`
  - `runtime` (`none` / `singularity` / `shifter` / `docker`), `image`, `binds`, `repo_mount`, `setup`, `interpreter` (absolute in-image python)
* - `submit`
  - `host` (optional remote login host to submit from), `setup` (commands run before submission)
* - `env`
  - environment variables injected into the job (data roots, NCCL knobs, …)
```

Always `--dry-run` first to confirm the rendered account/partition/GRES before
anything hits the queue. The `s3df` and `nersc` profiles below are concrete
examples to crib from.

## Add your own site (start from `slurm`)

For a generic Slurm cluster, build from the **`slurm`** base — not `s3df` /
`nersc`, which are SLAC/NERSC-specific. Drop a `launch/sites/<name>.yaml` that
inherits it (`--site <name>` resolves to `launch/sites/<name>.yaml`):

```yaml
# launch/sites/mycluster.yaml
_base_: slurm.yaml                 # generic Slurm defaults

site: mycluster

paths:
  repo_root: /home/me/particle-imaging-models   # shared checkout jobs run from
  exp_root: "{repo_root}/exp"                    # where runs are written

resources:
  nnodes: 1
  nproc_per_node: 4                 # GPUs per node
  cpus_per_proc: 12                 # CPUs per GPU
  time: "12:00:00"

slurm:
  account: my_account
  partition: gpu
  gpu_directive: gres               # `--gres=gpu:N`; some clusters need `gpus-per-node`

container:
  runtime: none                     # or `singularity` with an `image:` (see below)

env:
  PILARNET_DATA_ROOT_V1: /data/pilarnet/v1      # cluster-wide data roots
```

```bash
pimm submit --site mycluster --train.config <config> \
  --resources.nnodes 1 --resources.nproc-per-node 4 --dry-run
```

The keys a site profile understands:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Section
  - Keys
* - `paths`
  - `repo_root` (checkout jobs run from), `exp_root` (run outputs; default `{repo_root}/exp`)
* - `resources`
  - `nnodes`, `nproc_per_node` (GPUs/node), `cpus_per_proc` (CPUs/GPU), `time`, `mem`
* - `slurm`
  - `account`, `partition`, `qos`, `constraint`, `gpu_directive` (`gres` or `gpus-per-node`), `output`
* - `container`
  - `runtime` (`none` / `singularity` / `shifter`), `image`, `binds`, `repo_mount`, `setup`
* - `submit`
  - `host` (optional remote login host to submit from), `setup` (commands run before submission)
* - `env`
  - environment variables injected into the job (data roots, NCCL knobs, …)
```

Always `--dry-run` first to confirm the rendered account/partition/GRES before
anything hits the queue. The `s3df` and `nersc` profiles below are concrete
examples to crib from.

## `s3df` — SLAC S3DF

```bash
pimm submit --site s3df \
  --resources.nnodes 1 --resources.nproc-per-node 4 --resources.time 00:30:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

What `s3df.yaml` provides:

- `paths.repo_root` is **env-relative** (inherited from `slurm.yaml`) — it
  resolves to your checkout, so no per-user path is committed. Override
  `exp_root` per run (`--paths.exp-root`) or redirect checkpoints with `MODEL_DIR`
  in `.env` if you need outputs off `$HOME`.
- Submission runs **locally** by default (no `submit.host`); add
  `--submit.host iana` to submit from a remote login host.
- Jobs run under **Singularity** with `/sdf`, `/fs`, `/sdf/scratch`, and
  `/lscratch` bound in. The environment is **baked into the image** (no conda
  activation); the launcher enters with a hermetic `bash --noprofile --norc -c`
  and uses the in-image interpreter `container.interpreter` (`/opt/conda/bin/python`).
- GPU directive: `--gres=gpu:<N>`.
- Default account `mli:nu-ml-dev`, default partition `ampere`.

Override account/partition per invocation:

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml \
  --slurm.account neutrino:ml-dev --slurm.partition ampere --dry-run
```

## `nersc` — NERSC Perlmutter

```bash
pimm submit --site nersc \
  --resources.nnodes 4 --resources.nproc-per-node 4 --resources.time 02:00:00 \
  --recipe launch/runs/e050_tail.yaml
```

What `nersc.yaml` provides:

- `paths.repo_root` → the NERSC checkout; `paths.exp_root` → Perlmutter scratch.
- Submission assumes you are on a NERSC login node.
- Jobs use **Shifter** with the configured image and module.
- GPU directive: `--gpus-per-node` (set by `slurm.gpu_directive: gpus-per-node`).
- Defaults: `account: m5238_g`, `qos: regular`, `constraint: gpu`.
- Dataset env includes `PILARNET_DATA_ROOT_V1`.

## Container repo mounts

The Docker images install `pimm` editable at `/opt/pimm/src`. Containerized
launch configs bind your host checkout (`paths.repo_root`) onto
`container.repo_mount` (default `/opt/pimm/src`) and run `scripts/train.sh` from
there — so imports and training code point at *your* clone, not the snapshot
baked into the image.

Manual container use should preserve the same mount:

::::{tab-set}

:::{tab-item} Apptainer / Singularity
```bash
apptainer exec --nv \
  --bind "$PWD:/opt/pimm/src" --pwd /opt/pimm/src \
  /path/to/pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::

:::{tab-item} Shifter (NERSC)
```bash
shifter --image=youngsm/pimm:pytorch2.5.0-cuda12.4 \
  --volume="$PWD:/opt/pimm/src" \
  /bin/bash -lc 'cd /opt/pimm/src && \
    pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask'
```
:::

::::

## CPU / thread accounting

In launch YAML, `resources.cpus_per_proc` is **CPUs per GPU**. submitit requests
`cpus_per_task = cpus_per_proc × nproc_per_node`. For `nproc_per_node: 4` and
`cpus_per_proc: 12`, Slurm receives:

```bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
```

The job sets `OMP_NUM_THREADS` to the per-GPU value (`cpus_per_proc`).

## Environment variables

`scripts/train.sh` / `test.sh` source a repo-root `.env` if present (existing
shell variables take priority). Site YAML also injects env vars into the job.

```{list-table}
:header-rows: 1
:widths: 34 66

* - Variable
  - Purpose
* - `PILARNET_DATA_ROOT_V1` / `_V2`
  - PILArNet-M data roots per revision (dataset falls back to
    `~/.cache/pimm/pilarnet/<revision>`)
* - `MODEL_DIR`
  - Redirect checkpoints to another filesystem; the experiment `model/` becomes
    a symlink to `${MODEL_DIR}/exp/<group>/<name>/model/`
* - `WANDB_API_KEY`
  - W&B auth (alternative to `wandb login`)
* - `TORCH_CUDA_ARCH_LIST` / `CUMM_CUDA_ARCH_LIST`
  - target GPU archs for building the CUDA extensions
* - `OMP_NUM_THREADS`
  - set by the job to `cpus_per_proc`
* - `PYTHONNOUSERSITE=1`
  - set by the container entrypoint to protect imports under bind-mounted homes
```

:::{tip}
Make sure the launch path **exports the PILArNet variables before training
starts** — datasets read normal process environment. Site YAML is the right
place for cluster-wide data roots; `.env` is right for personal overrides.
:::

## Verifying what will actually run

When account or resources matter, confirm the rendered fields rather than
trusting the flags:

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml --dry-run \
  --output rendered/e050_tail.yaml      # write the manifest for review
```

For a submitted job, confirm with Slurm directly (see {doc}`monitoring`):

```bash
scontrol show job <jobid> | grep -E 'Account|Partition|QOS|NumNodes|Gres'
```

## Next

- {doc}`chaining` — walltime, QOS, and requeue chaining.
- {doc}`monitoring` — watching a run.
