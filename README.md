# T2 Multi-Agent Simulation (CPS5002 Assessment 2)

John Connor, a T-800 guardian and a T-1000 on a 30 x 30 city grid. Python 3.12+, standard library plus Matplotlib (for images/charts only).

## Run
    python main.py run --version v4 --seed 20014 --log logs/run.csv --snap figures/run --snap-steps 10 20
    python main.py experiment --versions v1 v2 v3 v4 --runs 400 --seed0 20000 --out results/main.csv
    python run_final_experiments.py     # all report experiments (about 10 min)
    python make_figures.py              # all report figures and significance tests
    python -m unittest discover -s tests -v

## Versions
v1 greedy baseline | v2 A* navigation | v3 adaptive T-1000 | v4 John utility model + lean guardian T-800.
Ablations and tuning candidates are named presets in `t2sim/config.py`.

## Layout
`t2sim/` world, agents, simulation, config, experiments, render. `results/` raw CSVs (including `failed_experiments/`). `logs/` decision logs. `figures/` images. `tests/` unit tests.
Seed blocks: 1000-1199 development, 5000-5199 tuning, 20000+ final evaluation.
