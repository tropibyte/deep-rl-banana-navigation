"""Turn finished ablation runs into figures, tables and filled-in report text.

Run after `banana-train ablate` completes:

    python scripts/finalize.py

Rebuilds every figure, then substitutes the placeholder comments in README.md
and Report.md with generated tables and numbers. Idempotent: each placeholder
is replaced between stable markers, so re-running after more seeds land just
updates the numbers in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from banana_nav.plotting import (  # noqa: E402
    VARIANT_LABEL, load_runs, make_all_figures, summary_markdown, _ordered,
)

RESULTS = REPO / "results" / "ablation"
ASSETS = REPO / "assets"
BUDGET = 900


def between(tag: str, body: str) -> str:
    return f"<!--{tag}-->\n{body}\n<!--/{tag}-->"


def splice(path: Path, tag: str, body: str) -> None:
    """Replace <!--TAG--> (or a previously spliced block) with new content."""
    text = path.read_text(encoding="utf-8")
    block = between(tag, body)
    open_t, close_t = f"<!--{tag}-->", f"<!--/{tag}-->"
    if open_t in text and close_t in text:
        pre = text.split(open_t)[0]
        post = text.split(close_t, 1)[1]
        text = pre + block + post
    elif open_t in text:
        text = text.replace(open_t, block)
    else:
        print(f"  ! placeholder {tag} not found in {path.name}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"  filled {tag} in {path.name}")


def stats_for(runs: dict) -> dict[str, dict]:
    out = {}
    for v, rs in runs.items():
        solved = [r["solved_episode"] for r in rs if r.get("solved_episode")]
        finals = [float(np.mean(r["scores"][-100:])) for r in rs if r.get("scores")]
        out[v] = {
            "n": len(rs),
            "n_solved": len(solved),
            "solved": solved,
            "median": float(np.median(solved)) if solved else None,
            "mean": float(np.mean(solved)) if solved else None,
            "std": float(np.std(solved)) if solved else None,
            "best_avg": max((r.get("best_avg", -99) for r in rs), default=float("nan")),
            "final_avg_mean": float(np.mean(finals)) if finals else float("nan"),
        }
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS), help="directory of run JSONs")
    ap.add_argument("--assets", default=str(ASSETS), help="where to write figures")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the generated blocks instead of editing the docs")
    args = ap.parse_args()

    results_dir, assets_dir = Path(args.results), Path(args.assets)
    dry = args.dry_run

    global splice
    if dry:
        _real_splice = splice

        def splice(path, tag, body):  # noqa: F811
            print(f"\n----- {tag} -> {path.name} -----\n{body}\n")

    runs = load_runs(results_dir)
    if not runs:
        print(f"No results in {results_dir}")
        return 1

    total = sum(len(v) for v in runs.values())
    print(f"loaded {total} runs across {len(runs)} variants")

    print("\nbuilding figures...")
    for p in make_all_figures(results_dir, assets_dir):
        print("  wrote", p)

    st = stats_for(runs)
    order = _ordered(runs)

    # ---- Report: full ablation table -------------------------------------
    rows = ["| Variant | Seeds solved | Median episodes to solve | Mean ± SD | "
            "Final 100-ep avg | vs baseline |",
            "|---|---|---|---|---|---|"]
    base = st.get("dqn", {})
    for v in order:
        s = st[v]
        if s["median"] is not None:
            med = f"**{int(s['median'])}**"
            ms = f"{s['mean']:.0f} ± {s['std']:.0f}"
        else:
            med, ms = f"not solved in {BUDGET}", "—"
        if v == "dqn":
            delta = "baseline"
        elif s["median"] and base.get("median"):
            d = s["median"] - base["median"]
            pct = 100.0 * d / base["median"]
            delta = f"{d:+.0f} ep ({pct:+.0f}%)"
        else:
            delta = "—"
        rows.append(f"| {VARIANT_LABEL.get(v, v)} | {s['n_solved']}/{s['n']} | {med} | {ms} "
                    f"| {s['final_avg_mean']:.2f} | {delta} |")
    splice(REPO / "Report.md", "ABLATION-TABLE", "\n".join(rows))

    # ---- Report: headline -------------------------------------------------
    flat = [r for rs in runs.values() for r in rs]
    solved = [r for r in flat if r.get("solved_episode")]
    if solved:
        hero = min(solved, key=lambda r: r["solved_episode"])
        best_v = min((v for v in order if st[v]["median"]), key=lambda v: st[v]["median"])
        headline = (
            f"**The environment was solved.** The best single run "
            f"({VARIANT_LABEL.get(hero['variant'], hero['variant'])}, seed {hero['seed']}) "
            f"reached a 100-episode average of +13 in **{hero['solved_episode']} episodes** — "
            f"against the project benchmark of 1800.\n\n"
            f"Across seeds, the fastest variant was "
            f"**{VARIANT_LABEL.get(best_v, best_v)}** at a median of "
            f"**{int(st[best_v]['median'])} episodes** "
            f"({st[best_v]['n_solved']}/{st[best_v]['n']} seeds solved), versus "
            + (f"**{int(st['dqn']['median'])} episodes** for the vanilla DQN baseline."
               if st.get("dqn", {}).get("median")
               else "a baseline that did not solve within the budget.")
        )
    else:
        headline = (f"No variant reached the +13 threshold within {BUDGET} episodes. "
                    f"Best 100-episode average observed: "
                    f"{max(s['best_avg'] for s in st.values()):.2f}.")
    splice(REPO / "Report.md", "HEADLINE-RESULT", headline)

    # ---- README: compact summary -----------------------------------------
    splice(REPO / "README.md", "RESULTS-SUMMARY", summary_markdown(runs))

    # ---- console dump for writing the discussion -------------------------
    print("\n" + "=" * 62)
    print("STATS (for the discussion section)")
    print("=" * 62)
    for v in order:
        s = st[v]
        med = f"{int(s['median'])}" if s["median"] is not None else "NOT SOLVED"
        print(f"{v:9} solved {s['n_solved']}/{s['n']}  median {med:>10}  "
              f"seeds={sorted(s['solved'])}  final_avg={s['final_avg_mean']:.2f}")
    print("=" * 62)

    if not dry:
        (REPO / "results" / "summary.json").write_text(
            json.dumps(st, indent=2), encoding="utf-8")
        print("wrote results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
