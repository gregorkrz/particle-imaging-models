# Job monitoring

A pimm run leaves three trails you can watch: the **experiment directory**, the
**experiment tracker** (W&B or TensorBoard), and **Slurm** itself.

## The experiment directory

Everything a run does lands under `exp/<config-group>/<name>/`:

```text
exp/panda/pretrain/my-run/
  train.log             # human-readable training log (append-mode on resume)
  config.py             # resolved config the run actually used
  run_metadata.json     # command, host, git status, original config path
  resolved_config.json  # full config as JSON
  model/                # checkpoints (or a symlink to $MODEL_DIR/...)
  events.out.tfevents…  # TensorBoard events (when use_wandb=False)
```

Tail the log:

```bash
tail -f exp/panda/pretrain/my-run/train.log
```

Each step logs `data_time`, `batch_time`, ETA, and the scalar losses; see {doc}`../hooks/logging`.
Epoch averages are logged at epoch boundaries.

:::{tip}
`run_metadata.json` records the exact command, working directory, host, the
original config path, CLI options, and git metadata for tracked files. It (and
the saved `config.py`) are the authoritative record of *what a run started
with* - they are written once and **not** rewritten on resume.
:::

## Experiment trackers

Rank 0 writes either W&B or TensorBoard:

::::{tab-set}

:::{tab-item} Weights & Biases
```bash
export WANDB_API_KEY=...
pimm submit --site mycluster --train.config <cfg> \
  --run.wandb-name my-display-name \
  --run.wandb-project Pretraining-Sonata-PILArNet-M
```
{py:class}`~pimm.engines.hooks.logging.WandbNamer` can auto-derive the run name from config keys
(`model.type`, `data.train.max_len`, `amp_dtype`, `seed`, …). In a chain, runs
are grouped and suffixed `-job0001`, `-job0002`, … automatically.
:::

:::{tab-item} TensorBoard
```bash
pimm launch --train.config <cfg> -- use_wandb=False
tensorboard --logdir exp/panda/pretrain/my-run
```
With `use_wandb=False`, events are written under the experiment directory.
:::

::::

Useful diagnostic hooks you can add to `cfg.hooks` to enrich the tracker:
{py:class}`~pimm.engines.hooks.diagnostics.GradientNormLogger`, `ResourceUtilizationLogger` (CPU/RAM/GPU memory),
{py:class}`~pimm.engines.hooks.diagnostics.ParameterCounter`, `PrototypeUsageLogger`, `FeatureStdMonitor`. See
{doc}`../hooks/diagnostics`.

## Slurm introspection

```bash
squeue --me                                  # your queued/running jobs
scontrol show job <jobid>                     # full job record
scontrol show job <jobid> | grep -E 'JobState|RunTime|TimeLimit|NodeList'
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode  # after the fact
```

When account or resources matter, verify the rendered job rather than trusting
flags:

```bash
scontrol show job <jobid> | grep -E 'Account|Partition|QOS|Gres'
```

A timeout-then-requeue looks like a job state transition in `squeue` followed by a fresh attempt picking up from the latest complete checkpoint.

submitit also writes its own logs (stdout/stderr and the manifest) under its job
folder. The `--dry-run` manifest - or `--output PATH` - is the record of the
exact resources, account, partition, and pre-rendered requeue attempts.

## Checkpoint health

A `standard`/DCP checkpoint is **complete** only if its directory exists and
contains a `.complete` marker. To find the newest complete checkpoint the way
the launcher does:

```bash
python -m pimm.utils.path latest-checkpoint exp/panda/pretrain/my-run
```

Candidates, newest-first: `model/last`, `model/last.prev`, `model/model_last.pth`.
The `.prev` rotation means an interrupted save never destroys the previous good
checkpoint. See {doc}`../checkpoints/index`.
