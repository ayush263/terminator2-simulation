"""Configuration and version presets.

Each development iteration is a preset so earlier versions stay reproducible
and can be compared under identical conditions (same seeds, same maps).
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    name: str = "v1"
    size: int = 30
    max_steps: int = 250
    tracker_every: int = 8       # T-1000 receives a noisy fix on John every N steps
    t1000_informed: bool = True  # False: T-1000 starts blind (no start position)
    t1000_enabled: bool = True   # False = navigation-only test (no adversary)
    # --- navigation (iteration 1) ---
    nav: str = "greedy"          # greedy | astar
    # --- John ---
    john_ai: str = "greedy"      # greedy | astar | utility
    john_zone: float = 2.5       # utility: threat-zone path-cost weight
    john_pen: float = 3.0        # utility: exit-race penalty weight
    # --- T-800 ---
    t800_ai: str = "follow"      # follow | guardian
    t800_energy_aware: bool = False
    t800_start_energy: float = 100.0
    t800_escort: str = "trail"   # guardian idle position: trail | near
    t800_engage: int = 0         # guardian: charge a known T-1000 within this distance
    t800_threat_age: int = 6     # guardian: ignore sightings older than this
    t800_intercept: bool = True  # guardian: predictive intercept / guard positioning
    t800_leash: int = 12         # guardian: max roaming distance from John
    # --- T-1000 (iteration 2) ---
    t1000_ai: str = "chase"      # chase | adaptive
    t1000_belief: bool = False
    t1000_ambush: bool = False   # predictive interception along John's modelled route
    t1000_flank: bool = False


# Damage / resource constants (shared by all versions)
JOHN_HP = 100
T800_HP = 100
T1000_HP = 120
JOHN_RADIUS = 6
T800_RADIUS = 8
T1000_RADIUS = 7
HAZARD_DMG = {"JOHN": 4, "T-800": 3, "T-1000": 6}
ATTACK_DMG = {"T-1000->JOHN": 20, "T-1000->T-800": 6, "T-800->T-1000": 12}
# Rules R2 (introduced after the ceiling effect seen in iteration 1)
T1000_BONUS_EVERY = 2          # T-1000 gets a second move every 2nd step (speed 1.5)
TRACKER_NOISE = 3              # fix error: +/- cells on each axis
PIN_HOLD_PROB = 0.7            # chance a T-1000 adjacent to the T-800 stays pinned each step
JOHN_MOVE_COST = 3.0           # stamina per move (rest regenerates 5)
FIRE_SPREAD_EVERY = 20
FIRE_SPREAD_MAX = 10

HAZ_COST = {"JOHN": 8.0, "T-800": 6.0, "T-1000": 14.0}   # extra path cost per known hazard cell

_ADAPT = dict(t1000_ai="adaptive", t1000_belief=True, t1000_ambush=True, t1000_flank=True)
_LEAN = dict(t800_ai="guardian", t800_engage=5, t800_leash=20, t800_escort="near",
             t800_threat_age=0, t800_intercept=False, t800_energy_aware=True)

PRESETS = {
    # Iteration 0: greedy reactive baseline
    "v1": Config(name="v1"),
    # Iteration 1: A* navigation with hazard-aware costs for every agent
    "v2": Config(name="v2", nav="astar", john_ai="astar"),
    # Iteration 2: adaptive T-1000 (belief map, predictive interception, flanking, retreat)
    "v3": Config(name="v3", nav="astar", john_ai="astar", **_ADAPT),
    # Iteration 3: John utility model + lean energy-aware guardian T-800
    "v4": Config(name="v4", nav="astar", john_ai="utility", **_LEAN, **_ADAPT),
    # --- navigation-only test for iteration 1 (no T-1000) ---
    "nav-v1": Config(name="nav-v1", t1000_enabled=False),
    "nav-v2": Config(name="nav-v2", nav="astar", john_ai="astar", t1000_enabled=False),
    # --- T-1000 ablations (defenders = v4) ---
    "v4-nobelief": Config(name="v4-nobelief", nav="astar", john_ai="utility", **_LEAN,
                          **{**_ADAPT, "t1000_belief": False}),
    "v4-nointercept": Config(name="v4-nointercept", nav="astar", john_ai="utility", **_LEAN,
                             **{**_ADAPT, "t1000_ambush": False}),
    "v4-noflank": Config(name="v4-noflank", nav="astar", john_ai="utility", **_LEAN,
                         **{**_ADAPT, "t1000_flank": False}),
    "v4-chase": Config(name="v4-chase", nav="astar", john_ai="utility", **_LEAN),
    # --- defender ablations (T-1000 = adaptive) ---
    "v4-johnastar": Config(name="v4-johnastar", nav="astar", john_ai="astar", **_LEAN, **_ADAPT),
    "v4-follow": Config(name="v4-follow", nav="astar", john_ai="utility", **_ADAPT),
    # --- blind-search scenario: no tracker, no start knowledge (tests the belief map) ---
    "v4-blind": Config(name="v4-blind", nav="astar", john_ai="utility", **_LEAN, **_ADAPT,
                       tracker_every=10**6, t1000_informed=False),
    "v4-blind-patrol": Config(name="v4-blind-patrol", nav="astar", john_ai="utility", **_LEAN,
                              **{**_ADAPT, "t1000_belief": False}, tracker_every=10**6,
                              t1000_informed=False),
    # --- energy scenario: T-800 starts depleted ---
    "v4-lowE": Config(name="v4-lowE", nav="astar", john_ai="utility", **_LEAN, **_ADAPT,
                      t800_start_energy=30.0),
    "v4-lowE-unaware": Config(name="v4-lowE-unaware", nav="astar", john_ai="utility",
                              **{**_LEAN, "t800_energy_aware": False}, **_ADAPT,
                              t800_start_energy=30.0),
    "v3-lowE": Config(name="v3-lowE", nav="astar", john_ai="astar", t800_start_energy=30.0,
                      **_ADAPT),
}

# Candidate configurations explored during iteration 3 (tuned on seeds 5000+, never
# on the evaluation seeds 1000+).
_J = dict(nav="astar", john_ai="utility", **_ADAPT)
TUNING = {
    "t8-follow": Config(name="t8-follow", **_J),
    "t8-lean": Config(name="t8-lean", **_J, **_LEAN),
    "t8-intercept": Config(name="t8-intercept", **_J, t800_ai="guardian", t800_engage=5,
                           t800_leash=8, t800_escort="near", t800_threat_age=0,
                           t800_energy_aware=True),
    "t8-intercept-trail": Config(name="t8-intercept-trail", **_J, t800_ai="guardian",
                                 t800_leash=12, t800_energy_aware=True),
    "t8-wide-engage": Config(name="t8-wide-engage", **_J, **{**_LEAN, "t800_engage": 8}),
    "t8-stale-sightings": Config(name="t8-stale-sightings", **_J,
                                 **{**_LEAN, "t800_threat_age": 6}),
}
for _z in (0.0, 1.5, 2.5, 4.0):
    for _p in (0.0, 3.0, 6.0):
        _n = f"j-z{_z}-p{_p}"
        TUNING[_n] = Config(name=_n, nav="astar", john_ai="utility", john_zone=_z, john_pen=_p,
                            **_LEAN, **_ADAPT)

ALL = {**PRESETS, **TUNING}
