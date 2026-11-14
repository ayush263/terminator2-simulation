"""Generates every figure and decision-log extract used in the report from real runs."""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from t2sim.config import ALL
from t2sim.experiments import wilson, two_prop_p
from t2sim.render import draw
from t2sim.simulation import Simulation

os.makedirs("figures", exist_ok=True)
INK, MUTE = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"


def run(name, seed, log=False):
    s = Simulation(ALL[name], seed, log=log)
    s.run()
    return s


def pair(a, b, title_a, title_b, path):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=130)
    draw(axs[0], a, title_a, legend=False)
    draw(axs[1], b, title_b, legend=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load(path):
    return list(csv.DictReader(open(path)))


def find_seed(v_bad, v_good, bad_out, good_out, s0=20000, n=400, extra=None):
    for seed in range(s0, s0 + n):
        a = Simulation(ALL[v_bad], seed).run()
        if a["outcome"] != bad_out:
            continue
        b = Simulation(ALL[v_good], seed).run()
        if b["outcome"] == good_out and (extra is None or extra(a, b)):
            return seed
    return None


def log_excerpt(sim, agent, steps, path, cols=("step", "state", "x", "y", "hp", "energy", "note")):
    with open(path, "w") as f:
        for r in sim.log:
            if r["agent"] == agent and r["step"] in steps:
                f.write(" | ".join(str(r[c]) for c in cols) + "\n")


# ---------------------------------------------------------------- iteration evidence
def iteration_figures():
    seeds = {}
    # Iteration 1: navigation-only
    seeds[1] = find_seed("nav-v1", "nav-v2", "TIMEOUT", "ESCAPED",
                         extra=lambda a, b: a["stuck_john"] > 100)
    a, b = run("nav-v1", seeds[1], True), run("nav-v2", seeds[1], True)
    pair(a, b, f"v1 greedy, seed {seeds[1]}: {a.outcome} after {a.step_no} steps "
               f"(stuck {a.john.stuck})",
         f"v2 A*, seed {seeds[1]}: {b.outcome} in {b.step_no} steps (stuck {b.john.stuck})",
         "figures/iter1_before_after.png")
    # Iteration 2: v2 escapes, v3 killed
    seeds[2] = find_seed("v2", "v3", "ESCAPED", "KILLED")
    a, b = run("v2", seeds[2], True), run("v3", seeds[2], True)
    pair(a, b, f"v2 chase T-1000, seed {seeds[2]}: {a.outcome} (step {a.step_no})",
         f"v3 adaptive T-1000, seed {seeds[2]}: {b.outcome} (step {b.step_no})",
         "figures/iter2_before_after.png")
    b.write_log("logs/iter2_v3_seed%d.csv" % seeds[2])
    a.write_log("logs/iter2_v2_seed%d.csv" % seeds[2])
    log_excerpt(b, "T-1000", range(1, 30, 2), "logs/excerpt_iter2_t1000.txt",
                cols=("step", "state", "x", "y", "hp", "note"))
    # Iteration 3: v3 kills John, v4 escapes
    seeds[3] = find_seed("v3", "v4", "KILLED", "ESCAPED")
    a, b = run("v3", seeds[3], True), run("v4", seeds[3], True)
    pair(a, b, f"v3 John A*, seed {seeds[3]}: {a.outcome} (step {a.step_no})",
         f"v4 John utility, seed {seeds[3]}: {b.outcome} (step {b.step_no})",
         "figures/iter3_before_after.png")
    a.write_log("logs/iter3_v3_seed%d.csv" % seeds[3])
    b.write_log("logs/iter3_v4_seed%d.csv" % seeds[3])
    log_excerpt(b, "JOHN", range(1, 60), "logs/excerpt_iter3_john.txt",
                cols=("step", "state", "x", "y", "hp", "energy", "note"))
    log_excerpt(b, "T-800", range(1, 60), "logs/excerpt_iter3_t800.txt",
                cols=("step", "state", "x", "y", "hp", "energy", "note"))
    # failed guardian attempt figure (archived earlier) is kept as is
    with open("results/figure_seeds.txt", "w") as f:
        f.write(str(seeds))
    print("seeds", seeds)


# ---------------------------------------------------------------- results charts
def bars(ax, labels, vals, cis, color, title=None):
    xs = range(len(labels))
    ax.bar(xs, vals, color=color, width=0.55)
    for x, v, (lo, hi) in zip(xs, vals, cis):
        ax.plot([x, x], [lo, hi], color=INK, lw=1.2)
        ax.text(x, hi + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def pct(rows, v, key):
    rs = [r for r in rows if r["version"] == v]
    k = sum(r["outcome"] == key for r in rs)
    lo, hi = wilson(k, len(rs))
    return 100 * k / len(rs), (100 * lo, 100 * hi)


def results_figures():
    main = load("results/E1_main.csv")
    vs = ["v1", "v2", "v3", "v4"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), dpi=130)
    for ax, key, col, t in zip(axs, ["ESCAPED", "KILLED", "T1000_DOWN"], [AQUA, RED, BLUE],
                               ["John escapes", "John killed", "T-1000 destroyed by T-800"]):
        v = [pct(main, x, key) for x in vs]
        bars(ax, vs, [a for a, _ in v], [b for _, b in v], col, t)
    axs[0].set_ylabel("% of 400 runs (95% Wilson CI)", fontsize=8, color=MUTE)
    fig.tight_layout()
    fig.savefig("figures/results_main.png", bbox_inches="tight")
    plt.close(fig)

    # steps distribution
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=130)
    for v, c in zip(vs, [MUTE, BLUE, ORANGE, AQUA]):
        xs = sorted(int(r["steps"]) for r in main if r["version"] == v)
        ax.plot(xs, [100 * (i + 1) / len(xs) for i in range(len(xs))], color=c, lw=2, label=v)
    ax.set_xlabel("simulation steps until termination", fontsize=8)
    ax.set_ylabel("cumulative % of runs", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#e5e4e0", lw=0.6)
    fig.tight_layout()
    fig.savefig("figures/results_steps.png", bbox_inches="tight")
    plt.close(fig)

    # ablations: John killed %
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), dpi=130)
    e2 = load("results/E2_t1000_ablation.csv")
    names = ["v4-chase", "v4-nointercept", "v4-noflank", "v4"]
    lab = ["no adaptation\n(chase)", "no\ninterception", "no\nflanking", "full\nadaptive"]
    v = [pct(e2, x, "KILLED") for x in names]
    bars(axs[0], lab, [a for a, _ in v], [b for _, b in v], RED,
         "E2: T-1000 components (John killed %, defenders = v4)")
    e3 = load("results/E3_defender_ablation.csv")
    names = ["v3", "v4-johnastar", "v4-follow", "v4"]
    lab = ["v3\n(baseline)", "lean T-800\nonly", "John utility\nonly", "v4\n(both)"]
    v = [pct(e3, x, "KILLED") for x in names]
    bars(axs[1], lab, [a for a, _ in v], [b for _, b in v], RED,
         "E3: defender components (John killed %, T-1000 adaptive)")
    fig.tight_layout()
    fig.savefig("figures/results_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, a, b, text="", fs=7, rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1,
                                                      connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=fs,
                color=MUTE)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, 3.4, 4.2, 3.2, 0.8, "Simulation (simulation.py)\nstep loop, interactions, metrics, logs",
        "#fbf1ea", ORANGE)
    box(ax, 0.2, 2.5, 3.0, 1.1, "World (world.py)\ngrid 30x30, map generator,\nLOS, A*, BFS", "#eaf7f1", AQUA)
    box(ax, 3.5, 2.5, 3.0, 1.1, "Agent base (agents.py)\nperceive(): LOS + radius,\nhazard memory, sightings")
    box(ax, 6.8, 2.5, 3.0, 1.1, "Config (config.py)\nversion presets v1-v4,\nablations, constants", "#f4f2fb", "#6250d6")
    box(ax, 0.2, 0.7, 2.9, 1.2, "John\nv1 greedy | v2 A* |\nv4 utility (REST/FLEE/TRAVEL)")
    box(ax, 3.55, 0.7, 2.9, 1.2, "T-800\nv1 follower | v4 lean guardian\n(ENGAGE, ESCORT, RECHARGE)")
    box(ax, 6.9, 0.7, 2.9, 1.2, "T-1000\nchase | adaptive: belief map,\nINTERCEPT, PURSUE, RETREAT")
    for x in (1.65, 5.0, 8.35):
        arrow(ax, (5.0, 4.2), (x, 3.6))
    arrow(ax, (1.7, 2.5), (1.7, 1.9))
    arrow(ax, (5.0, 2.5), (5.0, 1.9))
    arrow(ax, (8.35, 2.5), (8.35, 1.9))
    ax.text(5.0, 0.25, "Each step: perceive -> update state -> decide (intent) -> execute -> "
            "environment (hazards, fire spread, regen) -> interactions (pin, melee) -> metrics/termination",
            ha="center", fontsize=7.5, color=MUTE)
    fig.savefig("figures/architecture.png", bbox_inches="tight")
    plt.close(fig)


def states():
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.2), dpi=130)
    spec = [
        ("John (utility model)", [("REST", 1), ("TRAVEL", 4), ("FLEE", 7)],
         [(1, 0, "stamina<25,\nno threat"), (0, 1, "stamina>=60 or\nthreat<=12"),
          (1, 2, "T-1000 sighted\n<=8 cells"), (2, 1, "threat lost")]),
        ("T-800 (lean guardian)", [("RECHARGE", 1), ("ESCORT", 4), ("ENGAGE", 7), ("ATTACK", 9.2)],
         [(1, 0, "energy<35,\nno threat"), (0, 1, "energy>=85"),
          (1, 2, "fresh T-1000\nsighting <=5"), (2, 1, "lost"), (2, 3, "adjacent")]),
        ("T-1000 (adaptive)", [("TRACK", 0.8), ("INTERCEPT", 3.2), ("PURSUE", 5.6), ("KILL", 7.8),
                               ("RETREAT", 9.4)],
         [(0, 1, "fix/sighting"), (1, 2, "John <=8"), (2, 3, "adjacent"),
          (2, 4, "hp<35%\n& T-800 near"), (4, 0, "hp>=70%")]),
    ]
    for ax, (title, nodes, edges) in zip(axs, spec):
        ax.set_xlim(0, 10.6)
        ax.set_ylim(-1.4, 2.2)
        ax.axis("off")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        pos = {}
        for i, (n, x) in enumerate(nodes):
            box(ax, x - 0.55, 0.7, 1.1, 0.6, n, fs=7.5)
            pos[i] = x
        for a, b, t in edges:
            xa, xb = pos[a], pos[b]
            fwd = xb > xa
            far = abs(a - b) > 1
            rad = (-0.25 if fwd else -0.25) if not far else (0.45 if fwd else -0.45)
            sgn = 1 if fwd else -1
            ya = 1.3 if fwd else 0.7
            ax.annotate("", xy=(xb - sgn * 0.5, ya), xytext=(xa + sgn * 0.5, ya),
                        arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1,
                                        connectionstyle=f"arc3,rad={rad}"))
            ty = (1.75 if fwd else 0.15) - (0.25 if far else 0)
            ax.text((xa + xb) / 2, ty if not far else (1.95 if fwd else -0.15), t,
                    ha="center", va="center", fontsize=6.5, color=MUTE)
    fig.tight_layout()
    fig.savefig("figures/states.png", bbox_inches="tight")
    plt.close(fig)


def significance():
    main = load("results/E1_main.csv")
    out = []
    for a, b, key in [("v1", "v2", "ESCAPED"), ("v1", "v2", "KILLED"), ("v2", "v3", "KILLED"),
                      ("v3", "v4", "KILLED"), ("v3", "v4", "ESCAPED")]:
        ra = [r for r in main if r["version"] == a]
        rb = [r for r in main if r["version"] == b]
        ka = sum(r["outcome"] == key for r in ra)
        kb = sum(r["outcome"] == key for r in rb)
        out.append(f"{a}->{b} {key}: {ka}/{len(ra)} vs {kb}/{len(rb)}  p={two_prop_p(ka, len(ra), kb, len(rb)):.4f}")
    for f, a, b, key in [("E2_t1000_ablation", "v4-chase", "v4", "KILLED"),
                         ("E2_t1000_ablation", "v4-nointercept", "v4", "KILLED"),
                         ("E2_t1000_ablation", "v4-noflank", "v4", "KILLED"),
                         ("E3_defender_ablation", "v3", "v4", "KILLED"),
                         ("E3_defender_ablation", "v4-johnastar", "v3", "KILLED"),
                         ("E3_defender_ablation", "v4-follow", "v4", "KILLED"),
                         ("E4_energy", "v4-lowE-unaware", "v4-lowE", "KILLED"),
                         ("E5_blind_search", "v4-blind-patrol", "v4-blind", "KILLED")]:
        rows = load(f"results/{f}.csv")
        ra = [r for r in rows if r["version"] == a]
        rb = [r for r in rows if r["version"] == b]
        ka = sum(r["outcome"] == key for r in ra)
        kb = sum(r["outcome"] == key for r in rb)
        out.append(f"{f}: {a} vs {b} {key}: {ka}/{len(ra)} vs {kb}/{len(rb)}  p={two_prop_p(ka, len(ra), kb, len(rb)):.4f}")
    open("results/significance.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    architecture()
    states()
    iteration_figures()
    results_figures()
    significance()
