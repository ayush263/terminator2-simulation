"""Batch experiments: identical seeds (maps + start positions) for every version."""
import csv
import math
import statistics as st
from multiprocessing import Pool

from .config import ALL
from .simulation import Simulation


def run_one(args):
    name, seed = args
    return Simulation(ALL[name], seed).run()


def run_batch(names, seeds, out_csv=None, procs=4):
    jobs = [(n, s) for n in names for s in seeds]
    with Pool(procs) as p:
        rows = p.map(run_one, jobs, chunksize=4)
    if out_csv and rows:
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def two_prop_p(k1, n1, k2, n2):
    """Two-sided two-proportion z-test p value."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def summarise(rows):
    out = {}
    for name in dict.fromkeys(r["version"] for r in rows):
        rs = [r for r in rows if r["version"] == name]
        n = len(rs)
        cnt = {k: sum(r["outcome"] == k for r in rs) for k in
               ("ESCAPED", "KILLED", "TIMEOUT", "T1000_DOWN")}
        surv = [r["steps"] for r in rs]
        out[name] = dict(
            n=n, **cnt,
            survive_rate=(n - cnt["KILLED"]) / n,
            escape_ci=wilson(cnt["ESCAPED"], n),
            kill_ci=wilson(cnt["KILLED"], n),
            mean_steps=st.mean(surv), sd_steps=st.pstdev(surv),
            mean_john_hp=st.mean(r["john_hp"] for r in rs),
            mean_stuck_john=st.mean(r["stuck_john"] for r in rs),
            mean_stuck_t800=st.mean(r["stuck_t800"] for r in rs),
            mean_stuck_t1000=st.mean(r["stuck_t1000"] for r in rs),
            mean_contacts=st.mean(r["contact_episodes"] for r in rs),
            mean_evasions=st.mean(r["evasions"] for r in rs),
            mean_t800_shutdown=st.mean(r["t800_shutdown_steps"] for r in rs),
            mean_t1000_haz=st.mean(r["t1000_hazard_steps"] for r in rs),
            mean_t800_hits=st.mean(r["t800_hits"] for r in rs),
            mean_john_haz=st.mean(r["john_hazard_steps"] for r in rs),
        )
    return out


def print_summary(summ):
    hdr = f"{'version':<14}{'n':>4}{'ESC':>5}{'KILL':>5}{'T/O':>5}{'T1k↓':>5}{'esc%':>7}{'kill%':>7}" \
          f"{'steps':>7}{'stkJ':>6}{'stk8':>6}{'stk1k':>6}{'cont':>6}{'evad':>6}{'shut8':>6}{'haz1k':>6}"
    print(hdr)
    for k, s in summ.items():
        print(f"{k:<14}{s['n']:>4}{s['ESCAPED']:>5}{s['KILLED']:>5}{s['TIMEOUT']:>5}{s['T1000_DOWN']:>5}"
              f"{100*s['ESCAPED']/s['n']:>7.1f}{100*s['KILLED']/s['n']:>7.1f}{s['mean_steps']:>7.1f}"
              f"{s['mean_stuck_john']:>6.1f}{s['mean_stuck_t800']:>6.1f}{s['mean_stuck_t1000']:>6.1f}"
              f"{s['mean_contacts']:>6.2f}{s['mean_evasions']:>6.2f}{s['mean_t800_shutdown']:>6.1f}"
              f"{s['mean_t1000_haz']:>6.1f}")
