"""Reproduces every table used in the report.

  python run_final_experiments.py            # all experiments (about 10 minutes)

Seed blocks:  1000-1199 development (used while diagnosing problems)
              5000-5199 tuning     (candidate comparison for iteration 3)
              20000+    final evaluation (never used during development)
"""
import json
import sys
import time

from t2sim.experiments import run_batch, summarise, print_summary, two_prop_p

N_FINAL = 400
FINAL0 = 20000


def seeds(n, s0):
    return range(s0, s0 + n)


def block(title, names, n, s0, out, procs=2):
    t = time.time()
    rows = run_batch(names, seeds(n, s0), out, procs=procs)
    summ = summarise(rows)
    print(f"\n== {title}  ({n} runs/version, seeds {s0}-{s0 + n - 1}, {time.time() - t:.0f}s)")
    print_summary(summ)
    return rows, summ


def main(which):
    out = {}
    if "nav" in which:
        out["nav"] = block("E0 navigation-only (no T-1000)", ["nav-v1", "nav-v2"], 200, FINAL0,
                           "results/E0_navigation.csv")[1]
    if "main" in which:
        out["main"] = block("E1 main comparison", ["v1", "v2", "v3", "v4"], N_FINAL, FINAL0,
                            "results/E1_main.csv")[1]
    if "t1000" in which:
        out["t1000"] = block("E2 T-1000 ablation (defenders = v4)",
                             ["v4", "v4-chase", "v4-nobelief", "v4-nointercept", "v4-noflank"],
                             300, FINAL0, "results/E2_t1000_ablation.csv")[1]
    if "defender" in which:
        out["defender"] = block("E3 defender ablation (T-1000 adaptive)",
                                ["v3", "v4-johnastar", "v4-follow", "v4"], 300, FINAL0,
                                "results/E3_defender_ablation.csv")[1]
    if "energy" in which:
        out["energy"] = block("E4 depleted-T-800 scenario (T-800 starts at 30 energy)",
                              ["v3-lowE", "v4-lowE-unaware", "v4-lowE"], 300, FINAL0,
                              "results/E4_energy.csv")[1]
    if "belief" in which:
        out["belief"] = block("E5 blind T-1000 (no tracker, no start knowledge): belief map vs patrol",
                              ["v4-blind-patrol", "v4-blind"], 300, FINAL0,
                              "results/E5_blind_search.csv")[1]
    if "tuning" in which:
        names = ["v3", "t8-follow", "t8-lean", "t8-intercept", "t8-intercept-trail",
                 "t8-wide-engage", "t8-stale-sightings"]
        out["tuning_t800"] = block("T1 T-800 variants (tuning seeds)", names, 200, 5000,
                                   "results/T1_tuning_t800.csv")[1]
        grid = [f"j-z{z}-p{p}" for z in (0.0, 1.5, 2.5, 4.0) for p in (0.0, 3.0, 6.0)]
        out["tuning_john"] = block("T2 John utility grid (tuning seeds)", grid, 150, 5000,
                                   "results/T2_tuning_john_grid.csv")[1]
    with open(f"results/summary_{'_'.join(which)}.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or ["nav", "main", "t1000", "defender", "energy", "belief", "tuning"])
