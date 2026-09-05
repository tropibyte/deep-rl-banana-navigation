"""Command-line entry point: train, eval, ablate, plot, record."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- train ----
def cmd_train(args) -> int:
    from .config import build
    from .env import BananaEnv
    from .train import train, save_result

    tag = args.tag or "{}_seed{}".format(args.config, args.seed)
    out_dir = Path(args.out or REPO / "results" / "runs")
    ckpt = Path(args.checkpoint or REPO / "checkpoints" / (tag + ".pth"))

    with BananaEnv(exe_path=args.env_path, worker_id=args.worker_id,
                   no_graphics=not args.graphics, seed=args.seed) as env:
        name, acfg, tcfg = build(args.config, env.state_size, env.action_size,
                                 overrides={"n_episodes": args.episodes,
                                            "stop_on_solve": args.stop_on_solve})
        print("=== {} | seed {} | {} episodes | worker {} ===".format(
            name, args.seed, tcfg.n_episodes, env.worker_id), flush=True)
        res = train(env, acfg, tcfg, variant=name, seed=args.seed,
                    checkpoint_path=ckpt, csv_path=out_dir / (tag + ".csv"),
                    verbose=not args.quiet)

    save_result(res, out_dir / (tag + ".json"))
    print("\n{} seed {}: solved_in={} best_avg={:.2f} wall={:.1f}min".format(
        name, args.seed, res.solved_episode, res.best_avg, res.wall_seconds / 60))
    print("  checkpoint -> {}".format(ckpt))
    print("  result     -> {}".format(out_dir / (tag + ".json")))
    return 0


# ------------------------------------------------------------------ eval ----
def cmd_eval(args) -> int:
    from .agent import DQNAgent
    from .env import BananaEnv
    from .train import evaluate

    agent = DQNAgent.load(args.checkpoint)
    with BananaEnv(exe_path=args.env_path, worker_id=args.worker_id,
                   no_graphics=not args.graphics, seed=args.seed) as env:
        stats = evaluate(env, agent, episodes=args.episodes,
                         train_mode=not args.graphics, eps=args.eps)

    summary = {k: v for k, v in stats.items() if k != "scores"}
    print(json.dumps(summary, indent=2))
    verdict = "PASS (>= 13.0)" if stats["mean"] >= 13.0 else "below threshold"
    print("\nmean over {} episodes: {:.2f}  [{}]".format(
        args.episodes, stats["mean"], verdict))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- ablate ----
def cmd_ablate(args) -> int:
    """Run variants x seeds as independent OS processes, N at a time.

    Separate processes rather than threads: each run owns a Unity process and a
    gRPC server, and this also guarantees every run a clean RNG state.
    """
    variants = [v.strip() for v in args.configs.split(",") if v.strip()]
    seeds = list(range(args.seeds))
    out_dir = Path(args.out or REPO / "results" / "ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(v, s) for v in variants for s in seeds]
    if args.skip_existing:
        jobs = [(v, s) for v, s in jobs
                if not (out_dir / "{}_seed{}.json".format(v, s)).exists()]

    est_h = len(jobs) * args.episodes * 2.8 / 3600 / max(1, args.workers)
    print("{} runs ({} variants x {} seeds), {} episodes each, {} in parallel".format(
        len(jobs), len(variants), len(seeds), args.episodes, args.workers))
    print("rough estimate: {:.1f} h wall-clock\n".format(est_h), flush=True)
    if args.dry_run:
        for v, s in jobs:
            print("  would run {} seed {}".format(v, s))
        return 0

    def launch(slot, variant, seed):
        tag = "{}_seed{}".format(variant, seed)
        cmd = [sys.executable, "-u", "-m", "banana_nav.cli", "train",
               "--config", variant, "--seed", str(seed),
               "--episodes", str(args.episodes),
               # slots spaced by 4 so a leaked process never steals a live port
               "--worker-id", str(slot * 4),
               "--tag", tag, "--out", str(out_dir),
               "--checkpoint", str(REPO / "checkpoints" / (tag + ".pth")),
               "--quiet"]
        # Cap BLAS/OMP threads *before* the child imports torch: N parallel
        # runs would otherwise each spin up a core-sized pool and thrash.
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", BANANA_TORCH_THREADS="1")
        # Each Unity player writes "unity-environment.log" into its process
        # working directory and holds it open. Sharing one cwd across parallel
        # runs serialises them onto that single file -- in practice exactly one
        # run makes progress and the rest sit connected but idle forever.
        # A private cwd per run is the whole fix; measured throughput then
        # matches a solo env (~190 steps/s each).
        rundir = out_dir / "_rundirs" / tag
        rundir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / (tag + ".log"), "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(rundir), stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        return (proc, variant, seed, log, slot)

    queue = list(jobs)
    running = []
    free_slots = list(range(args.workers))
    done, failed = [], []
    t0 = time.time()

    while queue or running:
        while queue and free_slots:
            slot = free_slots.pop(0)
            v, s = queue.pop(0)
            running.append(launch(slot, v, s))
            print("  [start] {} seed {} (slot {})".format(v, s, slot), flush=True)
        time.sleep(2)
        for entry in list(running):
            proc, v, s, log, slot = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            free_slots.append(slot)
            ok = proc.returncode == 0
            (done if ok else failed).append((v, s))
            print("  [{}] {} seed {}  ({}/{}, {:.0f} min elapsed)".format(
                "done" if ok else "FAILED rc={}".format(proc.returncode),
                v, s, len(done) + len(failed), len(jobs),
                (time.time() - t0) / 60), flush=True)

    print("\nablation finished in {:.2f} h  ({} ok, {} failed)".format(
        (time.time() - t0) / 3600, len(done), len(failed)))
    if failed:
        print("failed runs:", failed)
        print("inspect logs in {}".format(out_dir))
    return 1 if failed else 0


# ------------------------------------------------------------------ plot ----
def cmd_plot(args) -> int:
    from .plotting import make_all_figures
    paths = make_all_figures(Path(args.results),
                             Path(args.out or REPO / "assets"),
                             solve_score=args.solve_score)
    for p in paths:
        print("wrote", p)
    return 0


# ---------------------------------------------------------------- record ----
def cmd_record(args) -> int:
    from .record import record_gif
    out = record_gif(checkpoint=args.checkpoint, out_path=args.out,
                     episodes=args.episodes, fps=args.fps,
                     env_path=args.env_path, max_frames=args.max_frames)
    print("wrote", out)
    return 0


# ------------------------------------------------------------------ main ----
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="banana-train",
        description="Rainbow-style value-based agents on Unity Banana Collector")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--env-path", default=None,
                        help="path to the Unity executable")
        sp.add_argument("--worker-id", type=int, default=None)
        sp.add_argument("--graphics", action="store_true",
                        help="show the Unity window (slower)")
        sp.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", help="train a single agent")
    common(t)
    t.add_argument("--config", default="rainbow")
    t.add_argument("--episodes", type=int, default=None)
    t.add_argument("--tag", default=None)
    t.add_argument("--out", default=None)
    t.add_argument("--checkpoint", default=None)
    t.add_argument("--stop-on-solve", action="store_true")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="evaluate a saved checkpoint greedily")
    common(e)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--eps", type=float, default=0.0,
                   help="evaluation epsilon; 0 = fully greedy, 0.05 = DQN paper protocol")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_eval)

    a = sub.add_parser("ablate", help="run the multi-seed ablation study")
    a.add_argument("--configs",
                   default="dqn,double,dueling,per,nstep,noisy,rainbow")
    a.add_argument("--seeds", type=int, default=5)
    a.add_argument("--episodes", type=int, default=700)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", default=None)
    a.add_argument("--skip-existing", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_ablate)

    pl = sub.add_parser("plot", help="build figures from run artifacts")
    pl.add_argument("--results", default=str(REPO / "results" / "ablation"))
    pl.add_argument("--out", default=None)
    pl.add_argument("--solve-score", type=float, default=13.0)
    pl.set_defaults(func=cmd_plot)

    r = sub.add_parser("record", help="record a GIF of a trained agent")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--out", default=str(REPO / "assets" / "trained_agent.gif"))
    r.add_argument("--episodes", type=int, default=1)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--max-frames", type=int, default=600)
    r.add_argument("--env-path", default=None)
    r.set_defaults(func=cmd_record)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
