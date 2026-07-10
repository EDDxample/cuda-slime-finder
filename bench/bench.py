"""Benchmark tracker for the CUDA slime finder.

Runs a finder binary on the standard benchmark region, records the throughput
reported on its STATS line into results.csv, and regenerates progress.png so
every optimization step is visible over time.

Usage:
    python bench/bench.py run build/v3.exe --label v3-my-change [--note "..."]
    python bench/bench.py plot
"""

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
PLOT = HERE / "progress.png"
FIELDS = ["timestamp", "label", "note", "x_count", "z_count", "kernel_ms", "rate"]

DEFAULT_SEED = 8354031675596398786
# Must match DEFAULT_REGION in src/bench.cuh.
DEFAULT_REGION = (-524288, -32768, 1048576, 65536)

STATS_RE = re.compile(
    r"STATS windows=(?P<windows>\d+) kernel_ms=(?P<ms>[\d.]+) rate=(?P<rate>[\d.eE+]+)"
)


def run_binary(binary: str, seed: int, region, reps: int) -> dict | None:
    cmd = [str(Path(binary).resolve()), str(seed)] + [str(v) for v in region]
    best = None
    for rep in range(reps):
        print(f"[{rep + 1}/{reps}] {' '.join(cmd)}")
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        print(out.rstrip())
        m = STATS_RE.search(out)
        if not m:
            sys.exit("error: no STATS line in binary output")
        if best is None or float(m["ms"]) < float(best["ms"]):
            best = m.groupdict()
    return best


def append_result(label: str, note: str, region, stats: dict) -> None:
    exists = RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "label": label,
                "note": note,
                "x_count": region[2],
                "z_count": region[3],
                "kernel_ms": stats["ms"],
                "rate": stats["rate"],
            }
        )


def fmt_rate(rate: float) -> str:
    if rate >= 1e9:
        return f"{rate / 1e9:.1f}B/s"
    return f"{rate / 1e6:.0f}M/s"


def make_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with RESULTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("error: no results recorded yet")

    labels = [r["label"] for r in rows]
    rates = [float(r["rate"]) for r in rows]
    xs = range(len(rows))

    surface, ink, secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    muted, grid, baseline, series = "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.plot(xs, rates, color=series, linewidth=2, marker="o", markersize=7, zorder=3)

    # Direct labels: throughput above each point, speedup vs the first entry below.
    for x, rate in zip(xs, rates):
        ax.annotate(
            fmt_rate(rate),
            (x, rate),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=ink,
        )
        if x > 0:
            ax.annotate(
                f"×{rate / rates[0]:.1f}",
                (x, rate),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=8,
                color=secondary,
            )

    ax.set_yscale("log")
    ax.set_ylim(min(rates) / 2, max(rates) * 3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, color=secondary)
    ax.set_ylabel("16x16 chunk clusters / second", fontsize=10, color=secondary)
    ax.set_title(
        "Slime finder throughput on the standard benchmark region (RTX 3060)",
        fontsize=11,
        color=ink,
        pad=14,
    )

    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    ax.tick_params(colors=muted, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(baseline)

    fig.tight_layout()
    fig.savefig(PLOT, facecolor=surface)
    print(f"wrote {PLOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="benchmark a binary and record the result")
    p_run.add_argument("binary")
    p_run.add_argument("--label", required=True, help="short version tag for the plot")
    p_run.add_argument("--note", default="", help="what changed in this version")
    p_run.add_argument(
        "--reps", type=int, default=2, help="runs (best kernel time wins)"
    )
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=DEFAULT_REGION,
        metavar=("X0", "Z0", "X_COUNT", "Z_COUNT"),
    )

    sub.add_parser("plot", help="regenerate progress.png from results.csv")

    args = parser.parse_args()
    if args.cmd == "run":
        stats = run_binary(args.binary, args.seed, args.region, args.reps)
        if not stats:
            print("ERROR: no stats from running binary", args.binary)
            exit(1)
        append_result(args.label, args.note, args.region, stats)
        make_plot()
    else:
        make_plot()


if __name__ == "__main__":
    main()
