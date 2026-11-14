"""Command line entry point.

  python main.py run --version v4 --seed 3 --log logs/v4_seed3.csv --snap figures/v4_seed3
  python main.py experiment --versions v1 v2 v3 v4 --runs 200 --out results/main.csv
"""
import argparse
import os

from t2sim.config import ALL as PRESETS
from t2sim.experiments import run_batch, summarise, print_summary
from t2sim.render import render
from t2sim.simulation import Simulation


def cmd_run(a):
    sim = Simulation(PRESETS[a.version], a.seed, log=bool(a.log))
    snaps = set(a.snap_steps)
    if a.snap:
        render(sim, f"{a.snap}_step000.png", f"{a.version} seed {a.seed} step 0")
    while sim.outcome is None:
        sim.step()
        if a.snap and sim.step_no in snaps:
            render(sim, f"{a.snap}_step{sim.step_no:03d}.png")
    res = sim.result()
    if a.snap:
        render(sim, f"{a.snap}_final.png",
               f"{a.version} seed {a.seed} final (step {sim.step_no}): {sim.outcome}")
    if a.log:
        sim.write_log(a.log)
    print(res)


def cmd_experiment(a):
    seeds = range(a.seed0, a.seed0 + a.runs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = run_batch(a.versions, seeds, a.out, procs=a.procs)
    print_summary(summarise(rows))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--version", default="v4")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--log")
    r.add_argument("--snap")
    r.add_argument("--snap-steps", type=int, nargs="*", default=[])
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("experiment")
    e.add_argument("--versions", nargs="+", default=["v1"])
    e.add_argument("--runs", type=int, default=200)
    e.add_argument("--seed0", type=int, default=1000)
    e.add_argument("--procs", type=int, default=4)
    e.add_argument("--out", default="results/main.csv")
    e.set_defaults(fn=cmd_experiment)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
