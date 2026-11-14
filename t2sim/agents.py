"""Agents: John Connor, T-800 (guardian), T-1000 (adversary).

Every agent follows the same loop each step:
    perceive(sim) -> decide(sim) -> (simulation executes action)
decide() returns an intent dict:
    {action: move|attack|rest|hold, cell, target, state, note, want}
`want` is True when the agent intended to change position (used to count
steps where it got stuck).
"""
import math

from .config import (JOHN_HP, T800_HP, T1000_HP, JOHN_RADIUS, T800_RADIUS,
                     T1000_RADIUS, HAZ_COST)
from .world import manhattan, cheb


def intent(action="hold", cell=None, target=None, state="", note="", want=False):
    return {"action": action, "cell": cell, "target": target, "state": state,
            "note": note, "want": want}


class Agent:
    kind = "human"

    def __init__(self, name, pos, hp, radius, sim):
        self.name = name
        self.pos = pos
        self.hp = float(hp)
        self.max_hp = float(hp)
        self.radius = radius
        self.energy = 100.0
        self.state = "INIT"
        self.seen = {}                 # other agent name -> (pos, step_seen)
        self.known_haz = set(sim.world.hazard_cells())
        self.stuck = 0
        self.distance = 0
        self.last_intent = None

    @property
    def alive(self):
        return self.hp > 0

    def seen_pos(self, name, sim, max_age=10**9):
        rec = self.seen.get(name)
        if rec and sim.step_no - rec[1] <= max_age:
            return rec[0]
        return None

    def sees_now(self, name, sim):
        rec = self.seen.get(name)
        return bool(rec and rec[1] == sim.step_no)

    # ---- 1. perception -----------------------------------------------------
    def perceive(self, sim):
        w = sim.world
        for other in sim.agents:
            if other is self or not other.alive:
                continue
            if w.can_see(self.pos, other.pos, self.radius):
                self.seen[other.name] = (other.pos, sim.step_no)
        # learn about hazards that are in sight (fire spreads unseen)
        self.vis = w.visible_cells(self.pos, self.radius)
        for c in self.vis:
            if w.cell(c) == 2:
                self.known_haz.add(c)
            else:
                self.known_haz.discard(c)

    def occupied(self, sim):
        return {a.pos for a in sim.agents if a is not self and a.alive}


# ------------------------------------------------------------------ navigation
def greedy_step(sim, agent, goal):
    """Reactive navigation: step to the neighbour that reduces Manhattan
    distance to the goal. No memory, no planning -> can get stuck behind
    buildings (local minima). Returns next cell or None."""
    w = sim.world
    occ = agent.occupied(sim)
    cur = manhattan(agent.pos, goal)
    best, best_d = [], cur
    for q in w.neighbours(agent.pos, agent.kind):
        if q in occ:
            continue
        d = manhattan(q, goal)
        if d < best_d:
            best, best_d = [q], d
        elif d == best_d and best_d < cur:
            best.append(q)
    if best:
        return sim.rng.choice(best)
    opts = [q for q in w.neighbours(agent.pos, agent.kind) if q not in occ]
    if opts and sim.rng.random() < 0.3:          # occasional random jitter
        return sim.rng.choice(opts)
    return None


def random_step(sim, agent):
    occ = agent.occupied(sim)
    opts = [q for q in sim.world.neighbours(agent.pos, agent.kind) if q not in occ]
    return sim.rng.choice(opts) if opts else None


def cost_map(sim, agent, extra=None):
    """Extra A* cost: hazards the agent currently knows about (+ caller terms)."""
    m = {c: HAZ_COST[agent.name] for c in agent.known_haz}
    if extra:
        for c, v in extra.items():
            m[c] = m.get(c, 0.0) + v
    return m


def plan(sim, agent, goals, extra=None):
    """A* from the agent to the nearest goal. Other agents are treated as
    obstacles; if that disconnects the goal we retry ignoring them.
    Returns (path, cost)."""
    goals = list(goals)
    cm = cost_map(sim, agent, extra)
    path, cost = sim.world.astar(agent.pos, goals, agent.kind, cm, agent.occupied(sim))
    if path is None:
        path, cost = sim.world.astar(agent.pos, goals, agent.kind, cm)
    return path, cost


def move_towards(sim, agent, goal, extra=None):
    """Next cell towards a goal using the version's navigation algorithm."""
    if sim.cfg.nav == "greedy":
        return greedy_step(sim, agent, goal)
    path, _ = plan(sim, agent, [goal], extra)
    return path[0] if path else None


def team_threat(sim):
    """Radio link between John and the T-800: the newest sighting of the
    T-1000 by either of them. Returns (pos, step_seen) or None."""
    best = None
    for a in (sim.john, sim.t800):
        rec = a.seen.get("T-1000")
        if rec and (best is None or rec[1] > best[1]):
            best = rec
    return best if sim.t1000.alive else None


# ------------------------------------------------------------------ John
class John(Agent):
    def __init__(self, pos, sim):
        super().__init__("JOHN", pos, JOHN_HP, JOHN_RADIUS, sim)
        self.goal = None
        self.resting = False
        self.calm = 0              # steps since a threat was last known

    def decide(self, sim):
        if sim.cfg.john_ai == "greedy":
            ex = min(sim.world.exits, key=lambda e: manhattan(self.pos, e))
            cell = move_towards(sim, self, ex)
            self.state = "RUN_TO_EXIT"
            return intent("move", cell, ex, self.state, "greedy to nearest exit", want=True)
        if sim.cfg.john_ai == "astar":
            # cheapest known route to any safe zone (hazards cost extra)
            path, cost = plan(sim, self, sim.world.exits)
            self.state = "TRAVEL"
            if not path:
                return intent("hold", None, None, self.state, "no route", want=True)
            self.goal = path[-1]
            return intent("move", path[0], path[-1], self.state,
                          f"A* route len={len(path)} cost={cost:.0f}", want=True)
        return self._utility(sim)

    # ---- utility-based behaviour (iteration 2) ----------------------------
    def _utility(self, sim):
        w = sim.world
        thr = team_threat(sim)
        age = sim.step_no - thr[1] if thr else 99
        tpos = thr[0] if thr and age <= 6 else None
        d = manhattan(self.pos, tpos) if tpos else 99
        self.calm = 0 if tpos else self.calm + 1
        # 1. REST: recover stamina when nothing threatens us
        if self.energy < 25 and (tpos is None or d > 12):
            self.resting = True
        if self.resting and (self.energy >= 60 or (tpos is not None and d <= 12)):
            self.resting = False
        if self.resting:
            self.state = "REST"
            return intent("rest", None, None, self.state,
                          f"stamina {self.energy:.0f} low, no threat nearby")
        # 2. FLEE / TRAVEL: choose the exit with best utility, avoiding the threat
        extra = {}
        if tpos:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    dd = abs(dx) + abs(dy)
                    c = (tpos[0] + dx, tpos[1] + dy)
                    if dd <= 5 and w.in_bounds(c):
                        extra[c] = (6 - dd) * sim.cfg.john_zone
        best = None
        for e in w.exits:
            path, cost = plan(sim, self, [e], extra)
            if not path:
                continue
            pen = 0.0
            if tpos:   # can the T-1000 (speed 1.5) beat John to this exit?
                t_time = manhattan(tpos, e) / 1.5
                pen = sim.cfg.john_pen * max(0.0, len(path) + 2 - t_time)
            util = -(cost + pen) + (6.0 if e == self.goal else 0.0)
            if best is None or util > best[0]:
                best = (util, e, path, cost, pen)
        if best is None:
            return intent("hold", None, None, "STUCK", "no route", want=True)
        _, e, path, cost, pen = best
        self.goal = e
        self.state = "FLEE" if tpos and d <= 8 else "TRAVEL"
        return intent("move", path[0], e, self.state,
                      f"exit {e} len={len(path)} cost={cost:.0f} risk_pen={pen:.0f}", want=True)


# ------------------------------------------------------------------ T-800
class T800(Agent):
    kind = "t800"

    def __init__(self, pos, sim):
        super().__init__("T-800", pos, T800_HP, T800_RADIUS, sim)
        self.recharging = False
        self.energy = sim.cfg.t800_start_energy

    def decide(self, sim):
        if sim.cfg.t800_ai == "follow":
            return self._follow(sim)
        return self._guardian(sim)

    # ---- v1/v2: reactive follower -------------------------------------------
    def _follow(self, sim):
        john = sim.john
        t = sim.t1000
        if t.alive and cheb(self.pos, t.pos) <= 1:
            self.state = "ATTACK"
            return intent("attack", None, "T-1000", self.state, "T-1000 adjacent")
        tpos = self.seen_pos("T-1000", sim, max_age=0)
        if tpos and manhattan(self.pos, tpos) <= 5:
            self.state = "PURSUE"
            return intent("move", move_towards(sim, self, tpos), "T-1000", self.state,
                          "T-1000 in sight", want=True)
        if manhattan(self.pos, john.pos) > 2:
            self.state = "FOLLOW"
            return intent("move", move_towards(sim, self, john.pos), "JOHN", self.state,
                          "follow John", want=True)
        self.state = "HOLD"
        return intent("hold", None, "JOHN", self.state, "near John")

    # ---- v3: guardian (iteration 2) ---------------------------------------------
    def _guardian(self, sim):
        w = sim.world
        john, t = sim.john, sim.t1000
        thr = team_threat(sim)
        age = sim.step_no - thr[1] if thr else 99
        tpos = thr[0] if thr and age <= sim.cfg.t800_threat_age else None
        if t.alive and cheb(self.pos, t.pos) <= 1:
            self.state = "ATTACK"
            return intent("attack", None, "T-1000", self.state, "T-1000 adjacent")
        # energy management: recharge when low and John is not in danger
        if sim.cfg.t800_energy_aware:
            if self.energy < 35 and age > 10:
                self.recharging = True
            if self.recharging and (self.energy >= 85 or (tpos and age <= 3)):
                self.recharging = False
            if self.recharging:
                path, _ = plan(sim, self, w.stations)
                if path is not None and len(path) <= 18:
                    self.state = "RECHARGE"
                    if not path:
                        return intent("hold", None, None, self.state, "charging")
                    return intent("move", path[0], path[-1], self.state,
                                  f"low energy {self.energy:.0f}, to station", want=True)
                self.recharging = False
        if tpos and sim.cfg.t800_engage and manhattan(self.pos, tpos) <= sim.cfg.t800_engage \
                and manhattan(self.pos, john.pos) <= sim.cfg.t800_leash + 2:
            self.state = "ENGAGE"
            return intent("move", move_towards(sim, self, tpos), "T-1000", self.state,
                          f"charge T-1000 at {tpos}", want=True)
        if tpos and sim.cfg.t800_intercept:
            # predict the T-1000's route to John and find where we can cut it off
            tpath, _ = w.astar(tpos, [john.pos], "t1000")
            if tpath:
                dist = w.bfs_dist(self.pos, "t800", limit=25)
                occ = self.occupied(sim)
                for i, c in enumerate(tpath[:14]):
                    tt = (i + 1) / 1.5                 # T-1000 arrival time (speed 1.5)
                    if (c in dist and dist[c] <= tt + 1 and c not in occ
                            and manhattan(c, john.pos) <= sim.cfg.t800_leash and c != john.pos):
                        self.state = "INTERCEPT"
                        if c == self.pos:
                            return intent("hold", None, "T-1000", self.state,
                                          "blocking predicted path")
                        return intent("move", move_towards(sim, self, c), c, self.state,
                                      f"cut off T-1000 at {c} (t800 {dist[c]} <= t1000 {tt:.1f}+1)",
                                      want=True)
            # cannot cut it off in time: close in and pin it (pinned T-1000 cannot reach John)
            if (manhattan(tpos, john.pos) <= sim.cfg.t800_leash + 3 and
                    manhattan(self.pos, john.pos) <= sim.cfg.t800_leash + 2):
                self.state = "ENGAGE"
                return intent("move", move_towards(sim, self, tpos), "T-1000", self.state,
                              f"close in on T-1000 at {tpos}", want=True)
            # threat is far away: stand between John and it
            occ = self.occupied(sim)
            cands = [(john.pos[0] + dx, john.pos[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx or dy)]
            cands = [c for c in cands if w.passable(c, "t800") and c not in occ]
            self.state = "GUARD"
            if cands:
                g = min(cands, key=lambda c: (manhattan(c, tpos), manhattan(c, self.pos)))
                if g == self.pos:
                    return intent("hold", None, "JOHN", self.state, "between John and threat")
                return intent("move", move_towards(sim, self, g), g, self.state,
                              "move between John and threat", want=True)
        # no known threat: escort, scouting ahead on John's route
        self.state = "ESCORT"
        if john.state == "REST" or john.goal is None:
            if manhattan(self.pos, john.pos) > 2:
                return intent("move", move_towards(sim, self, john.pos), "JOHN", self.state,
                              "close up on resting John", want=True)
            return intent("hold", None, "JOHN", self.state, "guarding resting John")
        if sim.cfg.t800_escort == "near":
            if manhattan(self.pos, john.pos) <= 2:
                return intent("hold", None, "JOHN", self.state, "close to John")
            return intent("move", move_towards(sim, self, john.pos), "JOHN", self.state,
                          "close up on John", want=True)
        # trail just behind John so a pursuer meets the T-800 first
        tr = sim.traj["JOHN"]
        behind = tr[-3] if len(tr) >= 3 else john.pos
        if manhattan(self.pos, behind) <= 1:
            return intent("hold", None, "JOHN", self.state, "trailing John")
        return intent("move", move_towards(sim, self, behind), behind, self.state,
                      "trail behind John", want=True)


# ------------------------------------------------------------------ T-1000
class T1000(Agent):
    kind = "t1000"

    def __init__(self, pos, sim):
        super().__init__("T-1000", pos, T1000_HP, T1000_RADIUS, sim)
        # tracker ping: T-1000 knows where John started (unless the blind scenario is on)
        if sim.cfg.t1000_informed:
            self.seen["JOHN"] = (sim.john.pos, 0)
        self.waypoint = None
        self.wp_step = 0
        self.fix = None                # latest noisy tracker fix (pos, step)
        # adaptive-strategy state (iteration 2)
        self.mode = "TRACK"
        self.ambush = None
        self.ambush_step = 0
        self.best_d = 999
        self.stall = 0
        self.bstep = -1
        self.belief = {}
        if sim.cfg.t1000_belief or sim.cfg.t1000_ambush:
            w = sim.world
            self.exit_d = {}
            for e in w.exits:
                for c, dd in w.bfs_dist(e, "human", limit=99).items():
                    if dd < self.exit_d.get(c, 999):
                        self.exit_d[c] = dd
            if sim.cfg.t1000_informed:
                self.belief = {sim.john.pos: 1.0}
            else:                               # blind: John could be anywhere
                cells = [(x, y) for y in range(w.size) for x in range(w.size)
                         if w.passable((x, y), "human") and (x, y) != pos]
                self.belief = {c: 1.0 / len(cells) for c in cells}

    # ---- belief model of John's location -------------------------------------
    def perceive(self, sim):
        super().perceive(sim)
        if self.belief or sim.cfg.t1000_belief or sim.cfg.t1000_ambush:
            self._update_belief(sim)

    def _diffuse(self, sim):
        """Motion model: John moves at most 1 cell per step and tends to head
        for an exit, so probability flows preferentially to cells nearer one."""
        w = sim.world
        new = {}
        for c, pv in self.belief.items():
            nb = list(w.neighbours(c, "human"))
            wts = [1.0 if self.exit_d.get(q, 999) < self.exit_d.get(c, 999) else 0.3 for q in nb]
            tot = sum(wts)
            new[c] = new.get(c, 0.0) + pv * 0.15
            for q, wq in zip(nb, wts):
                new[q] = new.get(q, 0.0) + pv * 0.85 * wq / tot
        for e in w.exits:
            new.pop(e, None)                   # mass reaching an exit = John already gone
        self.belief = {c: v for c, v in new.items() if v > 1e-5}

    def _update_belief(self, sim):
        w = sim.world
        if self.bstep != sim.step_no:          # one motion update per simulation step
            self.bstep = sim.step_no
            if self.belief:
                self._diffuse(sim)
        john = sim.john
        if self.sees_now("JOHN", sim):
            self.belief = {john.pos: 1.0}
            return
        for c in self.vis:                     # visible cells without John have no mass
            self.belief.pop(c, None)
        if self.fix and self.fix[1] == sim.step_no:
            fx, fy = self.fix[0]
            for c in list(self.belief):
                if abs(c[0] - fx) > 3 or abs(c[1] - fy) > 3:
                    self.belief[c] *= 0.02
        tot = sum(self.belief.values())
        if tot < 1e-9:                         # model broken by evidence: reset uniformly
            cells = [(x, y) for y in range(w.size) for x in range(w.size)
                     if w.passable((x, y), "human") and (x, y) not in self.vis]
            self.belief = {c: 1.0 for c in cells}
            tot = float(len(cells))
        self.belief = {c: v / tot for c, v in self.belief.items()}

    def belief_grid(self, size):
        g = [[0.0] * size for _ in range(size)]
        for (x, y), v in self.belief.items():
            g[y][x] = v
        return g

    def john_estimate(self, sim, max_age=10):
        """Best available estimate of John's position: direct sighting or the
        tracker fix, whichever is newer. Returns pos or None."""
        cands = []
        if "JOHN" in self.seen:
            cands.append(self.seen["JOHN"])
        if self.fix:
            cands.append(self.fix)
        if not cands:
            return None
        pos, st = max(cands, key=lambda r: r[1])
        return pos if sim.step_no - st <= max_age else None

    def decide(self, sim):
        if sim.cfg.t1000_ai == "adaptive":
            return self._adaptive(sim)
        return self._chase(sim)

    def _chase(self, sim):
        """v1/v2: pursue the newest estimate of John, else patrol at random."""
        john = sim.john
        if cheb(self.pos, john.pos) <= 1:
            self.state = "KILL"
            return intent("attack", None, "JOHN", self.state, "John adjacent")
        jpos = self.john_estimate(sim)
        if jpos and jpos == self.pos and sim.cfg.nav != "greedy":
            self.seen.pop("JOHN", None)          # reached last-known spot: John is not here
            self.fix = None
            jpos = None
        if jpos:
            self.state = "PURSUE"
            return intent("move", move_towards(sim, self, jpos), "JOHN", self.state,
                          "greedy to last known John", want=True)
        self.state = "SEARCH"
        if sim.cfg.nav == "greedy":
            return intent("move", random_step(sim, self), None, self.state, "random walk", want=True)
        return self._patrol(sim)

    def _patrol(self, sim):
        """Search: travel to random waypoints (A*), re-picked on arrival."""
        if (self.waypoint is None or self.pos == self.waypoint
                or sim.step_no - self.wp_step > 25):
            self.waypoint = sim.rng.choice(sorted(sim.world.flood(self.pos, "t1000")))
            self.wp_step = sim.step_no
        cell = move_towards(sim, self, self.waypoint)
        return intent("move", cell, self.waypoint, self.state, "patrol to waypoint", want=True)

    # ================= iteration 2: adaptive strategy ==============================
    def _set_mode(self, sim, mode):
        if mode != self.mode:
            sim.m["strategy_switches"] += 1
            self.mode = mode
            self.best_d = 999
            self.stall = 0
            if mode == "AMBUSH":
                self.ambush_step = sim.step_no

    def _adaptive(self, sim):
        """Adaptive strategy: opponent modelling + belief tracking.

        PURSUE  John is close: run him down (flank around the T-800).
        AMBUSH  John is far: predict his exit (he takes the cheapest route, which
                the T-1000 can compute itself) and out-race him to it.
        TRACK   no usable estimate: search the belief-map peak.
        RETREAT badly damaged: back off from the T-800 and regenerate.
        """
        cfg = sim.cfg
        w = sim.world
        john, t8 = sim.john, sim.t800
        if cheb(self.pos, john.pos) <= 1:
            self.state = "KILL"
            self._set_mode(sim, "PURSUE")
            return intent("attack", None, "JOHN", self.state, "John adjacent")
        t8pos = self.seen_pos("T-800", sim, max_age=3)
        est3 = self.john_estimate(sim, max_age=3)
        est14 = self.john_estimate(sim, max_age=14)
        # --- RETREAT
        if self.hp < 0.35 * self.max_hp and t8pos and manhattan(self.pos, t8pos) <= 8:
            self._set_mode(sim, "RETREAT")
        elif self.mode == "RETREAT" and self.hp >= 0.7 * self.max_hp:
            self._set_mode(sim, "TRACK")
        if self.mode == "RETREAT":
            self.state = "RETREAT"
            dist = w.bfs_dist(self.pos, "t1000", limit=12)
            ref = t8pos or t8.pos
            far = max(dist, key=lambda c: (manhattan(c, ref), -dist[c]))
            path, _ = plan(sim, self, [far])
            return intent("move", path[0] if path else None, far, self.state,
                          f"hp {self.hp:.0f} low: back off from T-800 to regenerate", want=True)
        in_position = (self.mode == "INTERCEPT" and self.ambush is not None
                       and manhattan(self.pos, self.ambush) <= 2)
        # --- PURSUE when John is close (unless already waiting in ambush)
        if est3 and manhattan(self.pos, est3) <= 8 and not in_position:
            return self._pursue(sim, est3, t8pos)
        # --- far from John: predictive interception along his modelled route
        if est14:
            rec = (self.fix if self.fix and (not self.seen.get("JOHN") or
                   self.fix[1] >= self.seen["JOHN"][1]) else self.seen["JOHN"])
            if cfg.t1000_ambush:
                got = self._intercept(sim, rec)
                if got:
                    c, tt, tj = got
                    self._set_mode(sim, "INTERCEPT")
                    self.ambush = c
                    self.state = "INTERCEPT"
                    if self.pos == c:
                        return intent("hold", None, c, self.state, f"waiting for John at {c}")
                    path, _ = plan(sim, self, [c])
                    return intent("move", path[0] if path else None, c, self.state,
                                  f"intercept John at {c}: me {tt / 1.5:.1f} steps, John {tj}",
                                  want=True)
            return self._pursue(sim, est14, t8pos)
        # --- TRACK
        self._set_mode(sim, "TRACK")
        self.state = "TRACK"
        if cfg.t1000_belief and self.belief:
            best, bs = None, -1.0
            for c in self.belief:
                sc = 0.0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        sc += self.belief.get((c[0] + dx, c[1] + dy), 0.0)
                sc /= (1.0 + 0.04 * manhattan(self.pos, c))
                if sc > bs:
                    best, bs = c, sc
            path, _ = plan(sim, self, [best])
            return intent("move", path[0] if path else None, best, self.state,
                          f"search belief peak {best} p={self.belief[best]:.3f}", want=True)
        return self._patrol(sim)

    def _intercept(self, sim, rec):
        """Opponent model: John follows the cheapest route to an exit. Project that
        route forward from the (possibly stale) fix and return the earliest cell on
        it that the T-1000 (speed 1.5) can reach before John: (cell, my_steps,
        john_steps) or None."""
        w = sim.world
        pos, st = rec
        age = sim.step_no - st
        cm = {c: 14.0 for c in self.known_haz}
        path, _ = w.astar(pos, w.exits, "human", cm)
        if not path:
            return None
        route = [pos] + path
        dist = w.bfs_dist(self.pos, "t1000", limit=70)
        for i in range(age + 2, len(route)):
            c = route[i]
            tj = i - age                       # steps until John reaches c
            tt = dist.get(c)
            if tt is not None and tt / 1.5 <= tj - 1:
                return c, tt, tj
        return None

    def _pursue(self, sim, est, t8pos):
        cfg, w = sim.cfg, sim.world
        if self.mode != "PURSUE":
            self._set_mode(sim, "PURSUE")
        d = manhattan(self.pos, est)
        if d < self.best_d:
            self.best_d, self.stall = d, 0
        else:
            self.stall += 1
            if self.stall >= 10:
                self.stall_step = sim.step_no     # chase is not closing the gap
                self.stall = 0
        extra, note = {}, "direct chase"
        if cfg.t1000_flank and t8pos:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    dd = max(abs(dx), abs(dy))
                    c = (t8pos[0] + dx, t8pos[1] + dy)
                    if w.in_bounds(c):
                        extra[c] = 10.0 if dd <= 1 else (5.0 if dd == 2 else 2.0)
            note = "flank around T-800"
        self.state = "PURSUE"
        path, _ = plan(sim, self, [est], extra)
        return intent("move", path[0] if path else None, est, self.state,
                      f"{note}; est={est} d={d} stall={self.stall}", want=True)
