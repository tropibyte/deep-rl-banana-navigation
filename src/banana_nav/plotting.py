"""Figures for the report.

Three figures, each chosen for the job its data actually does:

* ``learning_curve.png``     -- one hero run over time (change-over-time)
* ``ablation_curves.png``    -- small multiples, one facet per variant, baseline
                                ghosted behind each (avoids 7-series spaghetti)
* ``episodes_to_solve.png``  -- ranked magnitude across variants, per-seed dots

Colors are the validated two-hue categorical pair plus neutral ink; every
figure renders on an explicit light surface so it stays legible embedded in a
GitHub README under either site theme.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# --- validated tokens (see dataviz reference palette) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"   # blue  -- the variant under test
SERIES_2 = "#eb6834"   # orange -- the full stack / highlight
GHOST = "#c3c2b7"      # baseline reference, deliberately recessive

VARIANT_ORDER = ["dqn", "double", "dueling", "per", "nstep", "noisy", "rainbow"]
VARIANT_LABEL = {
    "dqn": "Vanilla DQN (baseline)",
    "double": "+ Double DQN",
    "dueling": "+ Dueling head",
    "per": "+ Prioritized replay",
    "nstep": "+ 3-step returns",
    "noisy": "+ NoisyNet",
    "rainbow": "All combined",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "font.size": 10,
    })


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """Group run JSON artifacts by variant name."""
    runs: dict[str, list[dict]] = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "scores" not in d:
            continue
        runs.setdefault(d.get("variant", f.stem), []).append(d)
    return runs


def _stack(runs: list[dict], key: str = "moving_avg") -> np.ndarray:
    """Stack per-seed curves into (n_seeds, n_episodes), truncated to the shortest."""
    curves = [r[key] for r in runs if r.get(key)]
    if not curves:
        return np.empty((0, 0))
    n = min(len(c) for c in curves)
    return np.array([c[:n] for c in curves], dtype=float)


def _ordered(runs: dict[str, list[dict]]) -> list[str]:
    known = [v for v in VARIANT_ORDER if v in runs]
    return known + sorted(v for v in runs if v not in VARIANT_ORDER)


# --------------------------------------------------------------- headline ----
def plot_headline(run: dict, out_path: Path, solve_score: float = 13.0) -> Path:
    _style()
    scores = np.array(run["scores"], dtype=float)
    avg = np.array(run["moving_avg"], dtype=float)
    eps = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(eps, scores, color=SERIES_1, alpha=0.22, linewidth=1.0)
    ax.plot(eps, avg, color=SERIES_1, linewidth=2.4)
    ax.axhline(solve_score, color=MUTED, linestyle="--", linewidth=1.4, zorder=1)

    # Direct labels instead of a legend box -- only two marks to name.
    ax.text(len(eps) * 0.985, solve_score + 0.45, "solved = +13",
            ha="right", va="bottom", color=INK_2, fontsize=9)
    # Anchor this in the empty upper-left quadrant: early episodes score near
    # zero, so nothing collides there, and it stays clear of the solve callout.
    anchor = int(len(avg) * 0.30)
    ax.annotate("100-episode average", xy=(eps[anchor], avg[anchor]),
                xytext=(len(eps) * 0.06, float(scores.max()) * 0.92),
                color=SERIES_1, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=SERIES_1, linewidth=1.2, alpha=0.55))
    ax.text(len(eps) * 0.06, np.percentile(scores, 3) - 0.6, "per-episode score",
            color=SERIES_1, alpha=0.75, fontsize=9)

    solved = run.get("solved_episode")
    if solved:
        cross = solved + 100
        ax.axvline(cross, color=SERIES_2, linewidth=1.6, alpha=0.85)
        ax.plot([cross], [solve_score], marker="o", markersize=8,
                color=SERIES_2, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"solved in {solved} episodes",
                    xy=(cross, solve_score), xytext=(cross + len(eps) * 0.03, solve_score - 5.2),
                    color=SERIES_2, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=SERIES_2, linewidth=1.4))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(f"{VARIANT_LABEL.get(run.get('variant'), run.get('variant'))}"
                 f"  ·  seed {run.get('seed')}",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(0, len(eps))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# -------------------------------------------------------- small multiples ----
def plot_ablation_curves(runs: dict[str, list[dict]], out_path: Path,
                         solve_score: float = 13.0) -> Path:
    _style()
    order = _ordered(runs)
    base = _stack(runs["dqn"]) if "dqn" in runs else np.empty((0, 0))
    base_med = np.median(base, axis=0) if base.size else None

    ncol = min(4, max(1, len(order)))
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow),
                             sharex=True, sharey=True, squeeze=False)

    for i, variant in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        curves = _stack(runs[variant])
        if not curves.size:
            continue
        x = np.arange(1, curves.shape[1] + 1)
        med = np.median(curves, axis=0)
        lo = np.percentile(curves, 25, axis=0)
        hi = np.percentile(curves, 75, axis=0)

        if base_med is not None and variant != "dqn":
            n = min(len(base_med), len(x))
            ax.plot(x[:n], base_med[:n], color=GHOST, linewidth=1.6, zorder=2)

        color = SERIES_2 if variant == "rainbow" else SERIES_1
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=3)
        ax.plot(x, med, color=color, linewidth=2.0, zorder=4)
        ax.axhline(solve_score, color=MUTED, linestyle="--", linewidth=1.1, zorder=1)

        ax.set_title(VARIANT_LABEL.get(variant, variant), color=INK,
                     fontsize=10.5, fontweight="bold", loc="left", pad=6)
        n_seeds = curves.shape[0]
        solved = [r["solved_episode"] for r in runs[variant] if r.get("solved_episode")]
        note = (f"solved {len(solved)}/{n_seeds} seeds"
                + (f", median {int(np.median(solved))} ep" if solved else ""))
        # Halo: fast variants plateau high enough that the curve runs through
        # this note in the top-left of the facet.
        ax.text(0.03, 0.94, note, transform=ax.transAxes, ha="left", va="top",
                color=INK_2, fontsize=8.5, zorder=6,
                path_effects=[pe.withStroke(linewidth=3.0, foreground=SURFACE)])
        ax.grid(axis="x", visible=False)

    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # With an incomplete final row, sharex would leave the bottom-most facet of
    # the short columns with no tick labels at all. Re-enable them per column.
    for col in range(ncol):
        rows = [r for r in range(nrow) if r * ncol + col < len(order)]
        if rows:
            axes[max(rows)][col].tick_params(labelbottom=True)

    fig.supxlabel("Episode", color=INK_2, fontsize=10)
    fig.supylabel("100-episode average score", color=INK_2, fontsize=10)
    fig.suptitle("Component ablation · median across seeds, shaded band = interquartile range"
                 "   ·   grey line = vanilla DQN baseline",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.945))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ------------------------------------------------------- episodes to solve ----
def plot_episodes_to_solve(runs: dict[str, list[dict]], out_path: Path,
                           budget: int | None = None) -> Path:
    _style()
    order = _ordered(runs)
    labels, medians, per_seed, unsolved = [], [], [], []
    for v in order:
        solved = [r["solved_episode"] for r in runs[v] if r.get("solved_episode")]
        n = len(runs[v])
        labels.append(VARIANT_LABEL.get(v, v))
        medians.append(float(np.median(solved)) if solved else 0.0)
        per_seed.append(solved)
        unsolved.append(n - len(solved))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.4, 0.56 * len(labels) + 2.2))

    # Reserve horizontal room so value labels never land on top of a seed dot.
    reach = [max([m] + list(pts)) if (pts or m) else 0 for m, pts in zip(medians, per_seed)]
    xmax = max(reach + ([budget] if budget else []) + [1])
    pad = xmax * 0.018

    for i, (m, pts, v) in enumerate(zip(medians, per_seed, order)):
        color = SERIES_2 if v == "rainbow" else SERIES_1
        if m > 0:
            ax.barh(i, m, height=0.34, color=color, alpha=0.9, zorder=3)
        if pts:
            # Per-seed dots: the spread is the whole point of running 5 seeds.
            ax.scatter(pts, np.full(len(pts), i), s=30, color=INK, alpha=0.5,
                       zorder=5, edgecolors=SURFACE, linewidths=1.1)
        # Anchor the value to the bar it describes, not to the far right of the
        # row: pushing it past an outlier dot makes it read as that dot's label.
        # A surface-coloured halo keeps it legible where a dot sits underneath.
        if m > 0:
            note = f"{int(m)}"
            if unsolved[i]:
                note += f"   {unsolved[i]}/{len(runs[v])} never solved"
            ax.text(m + pad, i, note, va="center", ha="left",
                    color=INK, fontsize=9.5, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])
        elif unsolved[i]:
            ax.text(pad, i, f"never solved in {len(runs[v])} seeds", va="center",
                    ha="left", color=MUTED, fontsize=9)

    if budget:
        ax.axvline(budget, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
        ax.text(budget - pad, -0.75, f"training budget ({budget})", color=MUTED,
                fontsize=8.5, va="center", ha="right")

    ax.set_xlim(0, xmax * 1.22)
    ax.set_ylim(len(labels) - 0.5, -1.1)
    ax.set_yticks(y, labels)  # ylim above is already top-down; do not invert again
    ax.set_xlabel("Episodes to reach a 100-episode average of +13  (lower is better)")
    ax.set_title("Cost to solve, by component · bar = median, dots = individual seeds",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------- summaries ----
def summary_markdown(runs: dict[str, list[dict]]) -> str:
    lines = ["| Variant | Seeds | Solved | Median episodes | Mean +/- SD | Best 100-ep avg |",
             "|---|---|---|---|---|---|"]
    for v in _ordered(runs):
        rs = runs[v]
        solved = [r["solved_episode"] for r in rs if r.get("solved_episode")]
        best = max((r.get("best_avg", float("-inf")) for r in rs), default=float("nan"))
        if solved:
            med = f"**{int(np.median(solved))}**"
            ms = f"{np.mean(solved):.0f} +/- {np.std(solved):.0f}"
        else:
            med, ms = "not solved", "-"
        lines.append(f"| {VARIANT_LABEL.get(v, v)} | {len(rs)} | {len(solved)}/{len(rs)} "
                     f"| {med} | {ms} | {best:.2f} |")
    return "\n".join(lines)


def make_all_figures(results_dir: Path, out_dir: Path,
                     solve_score: float = 13.0) -> list[Path]:
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No run artifacts (*.json) found in {results_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Headline = the fastest solve we have; failing that, the best average.
    flat = [r for rs in runs.values() for r in rs]
    solved = [r for r in flat if r.get("solved_episode")]
    hero = (min(solved, key=lambda r: r["solved_episode"]) if solved
            else max(flat, key=lambda r: r.get("best_avg", float("-inf"))))
    written.append(plot_headline(hero, out_dir / "learning_curve.png", solve_score))

    if len(runs) > 1:
        written.append(plot_ablation_curves(runs, out_dir / "ablation_curves.png", solve_score))
        budget = max((len(r["scores"]) for r in flat), default=None)
        written.append(plot_episodes_to_solve(runs, out_dir / "episodes_to_solve.png", budget))

    table = out_dir / "summary.md"
    table.write_text(summary_markdown(runs), encoding="utf-8")
    written.append(table)
    return written
