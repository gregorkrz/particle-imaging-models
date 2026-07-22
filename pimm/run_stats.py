"""Plot the distributions of a config's per-particle training targets.

Builds the dataset EXACTLY as a training config does (same reader + transforms),
then aggregates each query-head target to one value per instance (mean for
continuous heads, mode for categorical) -- i.e. the values the regression heads
are actually asked to predict -- and writes histograms + a JSON summary. Use it
to spot pathological target distributions (heavy tails, unit mistakes, sentinel
spikes) that can wreck head training.

Targets are shown AS THE MODEL SEES THEM (post-transform): e.g. ``momentum`` is
log10(GeV) via MomentumTransform, ``vertex`` is in the NormalizeCoord frame, and
``momentum_vec`` (px/py/pz) is signed GeV. That is the point -- these are the
regression targets, not the raw physical values.

Writes PNGs, a target_stats.json, and a self-contained target_stats.html (plots
inlined) to ``$OUTPUT_DIR/run_stats/<run-name>`` by default -- the same s3-synced
results dir the viewers use -- or ./outputs/run_stats/<run-name> when OUTPUT_DIR
is unset. Override with --output-dir.

Run inside the pimm apptainer image (needs the dataset stack; CPU is fine):
    pimm run-stats panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-fft --split val
    # or: python -m pimm.run_stats <config> ...
    # OUTPUT_DIR=/sdf/data/neutrino/gregork/lartpc_fastsim_outputs pimm run-stats ...

Notes:
- --split train uses the TRAIN transform, which applies random rotations/flips,
  so px/py/pz and vertex look isotropized. Use val/test (no augmentation) to see
  the intrinsic target distribution.
- Sentinel handling: heads with a ``sentinel`` (undefined truth, e.g. LED) are
  reported, and optionally dropped with --drop-sentinel. Detection is an exact
  match to the configured sentinel on the post-transform value, so it catches
  momentum / momentum_vec sentinels but NOT vertex (NormalizeCoord shifts the
  vertex sentinel off -1) -- a spike there will still be visible in the plot.
"""

import argparse
import base64
import datetime
import json
import os
import sys

import numpy as np


def _np(tensor):
    """Detach a (CPU) tensor to a NumPy array."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    return np.asarray(tensor)


# Nice component labels for the multi-dim heads; index fallbacks otherwise.
_COMPONENT_NAMES = {"momentum_vec": ("px", "py", "pz"), "vertex": ("x", "y", "z")}
# Short note appended to a head's plot titles describing its transform.
_TRANSFORM_NOTE = {
    "momentum": "log10(GeV)",
    "momentum_vec": "GeV",
    "vertex": "NormalizeCoord frame",
}


def _aggregate_per_instance(inst, values, categorical):
    """One value per instance id (>=0): mode if categorical else mean over points.

    ``values`` is (N,) or (N, D); returns (K, D) over the K unique instances.
    """
    values = values.reshape(values.shape[0], -1)
    dim = values.shape[1]
    uniq = np.unique(inst)
    uniq = uniq[uniq >= 0]
    out = []
    for instance_id in uniq:
        rows = values[inst == instance_id]
        if categorical:
            keys, counts = np.unique(rows, axis=0, return_counts=True)
            out.append(keys[int(np.argmax(counts))])
        else:
            out.append(rows.astype(np.float64).mean(axis=0))
    return np.stack(out, axis=0) if out else np.empty((0, dim))


def _collect(cfg, split, num_events, drop_sentinel):
    """Return {label: {head_name: {"values","kind","dim","sentinel","dropped"}}} + pid."""
    from pimm.datasets import build_dataset

    dataset = build_dataset(cfg.data[split])
    n_total = len(dataset)
    n = n_total if num_events < 0 else min(num_events, n_total)
    print(f"Dataset[{split}]: {n_total} events; scanning {n}.", flush=True)

    label_configs = cfg.model["label_configs"]
    # Per label: instance key + its query heads (name/kind/dim/sentinel).
    heads_by_label = {}
    for label, lc in label_configs.items():
        heads = []
        for head in lc.get("query_heads", []) or []:
            heads.append(dict(
                name=head["name"],
                kind=head.get("kind", "continuous"),
                dim=int(head.get("dim", 1)),
                sentinel=head.get("sentinel"),
            ))
        if heads:
            heads_by_label[label] = dict(instance_key=lc["instance_key"], heads=heads)

    acc = {
        label: {h["name"]: [] for h in info["heads"]}
        for label, info in heads_by_label.items()
    }
    dropped = {label: {h["name"]: 0 for h in info["heads"]} for label, info in heads_by_label.items()}
    pid_acc = []  # per-particle segment_pid (context)

    for i in range(n):
        sample = dataset[i]
        for label, info in heads_by_label.items():
            inst_key = info["instance_key"]
            if inst_key not in sample:
                continue
            inst = _np(sample[inst_key]).reshape(-1).astype(np.int64)
            for head in info["heads"]:
                name = head["name"]
                if name not in sample:
                    continue
                per_inst = _aggregate_per_instance(
                    inst, _np(sample[name]), head["kind"] == "categorical"
                )
                if per_inst.shape[0] == 0:
                    continue
                if head["sentinel"] is not None:
                    is_sentinel = np.all(
                        np.abs(per_inst - float(head["sentinel"])) < 1e-6, axis=1
                    )
                    dropped[label][name] += int(is_sentinel.sum())
                    if drop_sentinel:
                        per_inst = per_inst[~is_sentinel]
                acc[label][name].append(per_inst)
            # PID context from the particle label's instances.
            if label == cfg.model.get("eval_label", "particle") and "segment_pid" in sample:
                pid = _aggregate_per_instance(inst, _np(sample["segment_pid"]), True)
                if pid.shape[0]:
                    pid_acc.append(pid.reshape(-1))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    result = {}
    for label, info in heads_by_label.items():
        result[label] = {}
        for head in info["heads"]:
            name = head["name"]
            chunks = acc[label][name]
            values = np.concatenate(chunks, axis=0) if chunks else np.empty((0, head["dim"]))
            result[label][name] = dict(
                values=values, kind=head["kind"], dim=head["dim"],
                sentinel=head["sentinel"], dropped=dropped[label][name],
            )
    pid = np.concatenate(pid_acc) if pid_acc else np.empty((0,))
    return result, pid


def _panels_for_label(label, heads):
    """Build a flat list of plot panels for one label's heads."""
    panels = []
    for name, info in heads.items():
        values, dim, kind = info["values"], info["dim"], info["kind"]
        note = _TRANSFORM_NOTE.get(name, "")
        suffix = f" [{note}]" if note else ""
        if kind == "categorical":
            panels.append(dict(title=f"{label}.{name}{suffix}", kind="bar",
                               data=values.reshape(-1).astype(np.int64)))
            continue
        comp_names = _COMPONENT_NAMES.get(name, tuple(str(d) for d in range(dim)))
        for d in range(dim):
            label_d = comp_names[d] if d < len(comp_names) else str(d)
            title = f"{label}.{name}{'' if dim == 1 else '.' + label_d}{suffix}"
            panels.append(dict(title=title, kind="hist", data=values[:, d]))
        if dim > 1 and values.shape[0]:
            panels.append(dict(title=f"{label}.{name} |.|{suffix}", kind="hist",
                               data=np.linalg.norm(values, axis=1)))
    return panels


def _summary(data):
    """Per-panel summary stats (JSON-friendly)."""
    data = data[np.isfinite(data)] if data.dtype.kind == "f" else data
    if data.size == 0:
        return dict(n=0)
    out = dict(n=int(data.size), min=float(np.min(data)), max=float(np.max(data)),
               mean=float(np.mean(data)), std=float(np.std(data)),
               median=float(np.median(data)))
    if data.dtype.kind == "f":
        out["p1"], out["p99"] = [float(v) for v in np.percentile(data, [1, 99])]
    return out


def _plot(collected, pid, out_dir, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    png_paths = []
    for label, heads in collected.items():
        panels = _panels_for_label(label, heads)
        if label == next(iter(collected)) and pid.size:
            panels.append(dict(title="particle.segment_pid", kind="bar", data=pid.astype(np.int64)))
        if not panels:
            continue
        ncols = 3
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
        summary[label] = {}
        for ax, panel in zip(axes.ravel(), panels):
            data = panel["data"]
            summary[label][panel["title"]] = _summary(data)
            if data.size == 0:
                ax.set_title(f"{panel['title']} (empty)")
                ax.axis("off")
                continue
            if panel["kind"] == "bar":
                vals, counts = np.unique(data, return_counts=True)
                ax.bar([str(int(v)) for v in vals], counts, color="#4c78a8")
                ax.set_ylabel("particles")
            else:
                finite = data[np.isfinite(data)]
                ax.hist(finite, bins=bins, color="#4c78a8")
                ax.set_ylabel("particles")
                ax.axvline(float(np.median(finite)), color="#e45756", lw=1, ls="--")
            ax.set_title(panel["title"], fontsize=9)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        # Note dropped/sentinel counts in the figure suptitle.
        drops = {n: h["dropped"] for n, h in heads.items() if h.get("dropped")}
        subtitle = f"  (sentinel instances: {drops})" if drops else ""
        fig.suptitle(f"Training targets: {label}{subtitle}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(out_dir, f"target_distributions_{label}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(path)
        print(f"Wrote {path}", flush=True)
    with open(os.path.join(out_dir, "target_stats.json"), "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {os.path.join(out_dir, 'target_stats.json')}", flush=True)
    return summary, png_paths


_STAT_COLS = ("n", "mean", "std", "min", "max", "median", "p1", "p99")


def _write_html(out_dir, run_name, meta, summary, png_paths):
    """Write a self-contained target_stats.html (plots inlined + summary tables)."""
    def esc(text):
        return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Training targets — {esc(run_name)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;}"
        "h1{font-size:20px;}h2{font-size:16px;margin-top:28px;}"
        ".meta{color:#666;font-size:13px;margin-bottom:16px;}"
        "img{max-width:100%;height:auto;border:1px solid #e2e2e2;border-radius:6px;}"
        "table{border-collapse:collapse;font-size:12px;margin:10px 0 24px;}"
        "th,td{border:1px solid #ddd;padding:3px 8px;text-align:right;white-space:nowrap;}"
        "th{background:#f4f4f4;}td.name,th.name{text-align:left;}</style></head><body>",
        f"<h1>Training-target distributions — {esc(run_name)}</h1>",
        "<div class='meta'>"
        + " &nbsp;•&nbsp; ".join(f"{esc(k)}: {esc(v)}" for k, v in meta.items())
        + "</div>",
    ]
    for label, panels in summary.items():
        parts.append(f"<h2>{esc(label)}</h2>")
        png = next((p for p in png_paths if p.endswith(f"_{label}.png")), None)
        if png and os.path.isfile(png):
            with open(png, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append(f"<img src='data:image/png;base64,{b64}'/>")
        parts.append(
            "<table><thead><tr><th class='name'>target</th>"
            + "".join(f"<th>{c}</th>" for c in _STAT_COLS)
            + "</tr></thead><tbody>"
        )
        for title, stats in panels.items():
            cells = []
            for col in _STAT_COLS:
                val = stats.get(col)
                if val is None:
                    cells.append("<td>—</td>")
                elif isinstance(val, float):
                    cells.append(f"<td>{val:.4g}</td>")
                else:
                    cells.append(f"<td>{val}</td>")
            parts.append(f"<tr><td class='name'>{esc(title)}</td>" + "".join(cells) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")

    path = os.path.join(out_dir, "target_stats.html")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote {path}", flush=True)
    return path


def main(argv=None):
    from pimm.engines.defaults import default_config_parser
    from pimm.utils.config import DictAction

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="Config name (dataset/model-exp) or path to a .py.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Dataset split to scan (default val; train applies augmentation).")
    parser.add_argument("--num-events", type=int, default=1000,
                        help="Events to scan (-1 = all in the split).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the plots + json + html. Default: "
                             "$OUTPUT_DIR/run_stats/<run-name> (the s3-synced results "
                             "dir when OUTPUT_DIR is set), else ./outputs/run_stats/<run-name>.")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins.")
    parser.add_argument("--drop-sentinel", action="store_true",
                        help="Drop instances at a head's configured sentinel (undefined truth).")
    parser.add_argument("--options", nargs="+", action=DictAction,
                        help="Config overrides, e.g. data.val.data_root=/path data.val.max_len=5000")
    args = parser.parse_args(argv)

    # A short, unique run name from the config (handles both a config name and a
    # saved exp config.py, whose basename is the uninformative "config").
    run_name = os.path.basename(args.config)
    run_name = run_name[:-3] if run_name.endswith(".py") else run_name
    if run_name in {"config", "resolved_config"}:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(args.config)))

    # Default under the s3-synced results dir ($OUTPUT_DIR), namespaced per run so
    # multiple configs coexist; ./outputs is the local fallback.
    out_dir = args.output_dir or os.path.join(
        os.environ.get("OUTPUT_DIR", "outputs"), "run_stats", run_name
    )

    cfg = default_config_parser(args.config, args.options, save_artifacts=False)
    collected, pid = _collect(cfg, args.split, args.num_events, args.drop_sentinel)
    summary, png_paths = _plot(collected, pid, out_dir, args.bins)
    meta = {
        "run": run_name,
        "config": args.config,
        "split": args.split,
        "num_events": args.num_events,
        "drop_sentinel": args.drop_sentinel,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_html(out_dir, run_name, meta, summary, png_paths)
    print(f"Done. Output in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
