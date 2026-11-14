"""Simulation loop, interaction resolution and metrics."""
import csv
import random

from .agents import John, T800, T1000
from .config import (HAZARD_DMG, ATTACK_DMG, FIRE_SPREAD_EVERY, FIRE_SPREAD_MAX,
                     T1000_BONUS_EVERY, TRACKER_NOISE, JOHN_MOVE_COST,
                     PIN_HOLD_PROB)
from .world import World, HAZARD, STATION, SAFE, ROAD, N4, cheb


class Simulation:
    def __init__(self, cfg, seed, log=False):
        self.cfg = cfg
        self.seed = seed
        self.world = World.generate(seed, cfg.size)
        self.rng = random.Random(seed * 7919 + 13)
        self.step_no = 0
        s = self.world.starts
        self.john = John(s["john"], self)
        self.t800 = T800(s["t800"], self)
        self.t1000 = T1000(s["t1000"], self)
        self.agents = [self.john, self.t800, self.t1000]
        if not cfg.t1000_enabled:
            self.t1000.hp = 0.0                 # navigation-only test: adversary removed
        self.outcome = None
        self.log_on = log
        self.log = []
        self.traj = {a.name: [a.pos] for a in self.agents}
        # metrics
        self.m = dict(detect_time=-1, find_time=-1, first_engage=-1,
                      contact_steps=0, contact_episodes=0, evasions=0,
                      t800_shutdown_steps=0, t800_recharge_steps=0,
                      strategy_switches=0, t800_hits=0, t1000_hits_on_john=0,
                      t1000_hits_on_t800=0, min_t800_energy=100.0,
                      john_hazard_steps=0, t1000_hazard_steps=0, t800_gap_at_contact=-1, pinned_steps=0)
        self._in_contact = False
        self._threatened = False
        self._unseen = 0

    # ------------------------------------------------------------------
    def can_move(self, a):
        s = self.step_no
        if a.name == "T-800":
            if a.energy <= 0:
                return False
            if a.energy < 20 and s % 2:
                return False
        elif a.name == "JOHN":
            if a.energy < 10 and s % 2:
                return False
        else:  # T-1000: slowed in hazards or when badly damaged
            if (self.world.cell(a.pos) == HAZARD or a.hp < 0.3 * a.max_hp) and s % 2:
                return False
        return True

    def _exec_move(self, a, it):
        w = self.world
        if it["action"] == "move" and it["cell"] is not None:
            occupied = {o.pos for o in self.agents if o is not a and o.alive}
            if (it["cell"] not in occupied and w.passable(it["cell"], a.kind)
                    and self.can_move(a)):
                a.pos = it["cell"]
                a.distance += 1
                self.traj[a.name].append(a.pos)
                if a.name == "T-800":
                    a.energy -= 1.0
                elif a.name == "JOHN":
                    a.energy -= JOHN_MOVE_COST
                it["moved"] = True
                return
        if it["want"]:
            a.stuck += 1
        it["moved"] = False

    def _tracker(self):
        """Intermittent noisy fix on John's position, delivered to the T-1000."""
        if self.step_no % self.cfg.tracker_every:
            return
        j = self.john.pos
        w = self.world
        for _ in range(20):
            p = (j[0] + self.rng.randint(-TRACKER_NOISE, TRACKER_NOISE),
                 j[1] + self.rng.randint(-TRACKER_NOISE, TRACKER_NOISE))
            if w.passable(p, "t1000"):
                self.t1000.fix = (p, self.step_no)
                return

    def step(self):
        self.step_no += 1
        w = self.world
        self._tracker()
        j, t8, t1 = self.john, self.t800, self.t1000
        # 1-2. perceive and update internal state (John, T-800 and T-1000 belief motion update)
        for a in self.agents:
            a.perceive(self)
        # 3-5. John and the T-800 decide simultaneously and move in random order
        intents = {}
        first = [j, t8]
        for a in first:
            intents[a.name] = a.decide(self)
            a.last_intent = intents[a.name]
        self.rng.shuffle(first)
        for a in first:
            self._exec_move(a, intents[a.name])
        # the T-1000 observes the new situation, then acts (reactive turn order)
        if t1.alive:
            t1.perceive(self)
            intents["T-1000"] = t1.decide(self)
            t1.last_intent = intents["T-1000"]
            pinned = t8.alive and t8.energy > 0 and cheb(t8.pos, t1.pos) <= 1
            if pinned and self.rng.random() < PIN_HOLD_PROB:
                # pinned by the T-800: fails to break free this step
                self.m["pinned_steps"] += 1
                intents["T-1000"]["moved"] = False
            else:
                self._exec_move(t1, intents["T-1000"])
            # Grapple rule: a T-1000 within one cell of the T-800 is held and loses
            # its bonus move. Liquid-metal speed: a 2nd observe-decide-move every 2nd step.
            grappled = t8.alive and t8.energy > 0 and cheb(t8.pos, t1.pos) <= 1
            if self.step_no % T1000_BONUS_EVERY == 0 and not grappled:
                t1.perceive(self)
                it2 = t1.decide(self)
                if it2["action"] == "move":
                    self._exec_move(t1, it2)
                    if it2.get("moved"):
                        intents["T-1000"]["moved"] = True
        else:
            intents["T-1000"] = {"action": "hold", "cell": None, "target": None,
                                 "state": "DEAD", "note": "", "want": False, "moved": False}
        # 6. environment update
        self._environment(intents)
        # 7. interactions and consequences (melee strikes resolve on final positions)
        self._combat(intents)
        self._metrics(intents)
        if self.log_on:
            for a in self.agents:
                it = intents[a.name]
                self.log.append(dict(step=self.step_no, agent=a.name, state=it["state"],
                                     action=it["action"], x=a.pos[0], y=a.pos[1],
                                     target=str(it["target"]), hp=round(a.hp, 1),
                                     energy=round(a.energy, 1), note=it["note"]))
        # 9. termination
        self._check_end()

    # ------------------------------------------------------------------
    def _environment(self, intents):
        w = self.world
        for a in self.agents:
            if not a.alive:
                continue
            c = w.cell(a.pos)
            if c == HAZARD:
                a.hp -= HAZARD_DMG[a.name]
                if a.name == "JOHN":
                    self.m["john_hazard_steps"] += 1
                if a.name == "T-1000":
                    self.m["t1000_hazard_steps"] += 1
            it = intents[a.name]
            if a.name == "JOHN":
                if not it.get("moved"):
                    a.energy = min(100.0, a.energy + 5.0)
                if c == STATION:
                    a.hp = min(a.max_hp, a.hp + 3.0)
            elif a.name == "T-800":
                if c == STATION:
                    a.energy = min(100.0, a.energy + 10.0)
                    a.hp = min(a.max_hp, a.hp + 2.0)
                    self.m["t800_recharge_steps"] += 1
                if a.energy <= 0:
                    self.m["t800_shutdown_steps"] += 1
                if not it.get("moved") and it["action"] != "attack":
                    a.energy = min(100.0, a.energy + 0.5)
                self.m["min_t800_energy"] = min(self.m["min_t800_energy"], a.energy)
            else:
                seen_by_friend = (w.can_see(self.john.pos, a.pos, self.john.radius) or
                                  w.can_see(self.t800.pos, a.pos, self.t800.radius))
                a.hp = min(a.max_hp, a.hp + (0.5 if seen_by_friend else 1.0))
        # spreading fire: only discovered by agents who see it
        if self.step_no % FIRE_SPREAD_EVERY == 0 and w.fire_added < FIRE_SPREAD_MAX:
            haz = list(w.hazard_cells())
            cands = []
            for h in haz:
                for dx, dy in N4:
                    q = (h[0] + dx, h[1] + dy)
                    if w.in_bounds(q) and w.cell(q) == ROAD:
                        cands.append(q)
            if cands:
                q = self.rng.choice(cands)
                w.cells[q[1]][q[0]] = HAZARD
                w.fire_added += 1

    def _combat(self, intents):
        """Melee: any enemy adjacent after movement is struck once per step."""
        j, t8, t1 = self.john, self.t800, self.t1000
        if t1.alive:
            if cheb(t1.pos, j.pos) <= 1:
                j.hp -= self.rng.randint(*self._dmg_range(ATTACK_DMG["T-1000->JOHN"]))
                self.m["t1000_hits_on_john"] += 1
            elif t8.alive and cheb(t1.pos, t8.pos) <= 1:
                t8.hp -= self.rng.randint(*self._dmg_range(ATTACK_DMG["T-1000->T-800"]))
                self.m["t1000_hits_on_t800"] += 1
        if t8.alive and t8.energy > 0 and t1.alive and cheb(t8.pos, t1.pos) <= 1:
            t1.hp -= self.rng.randint(*self._dmg_range(ATTACK_DMG["T-800->T-1000"]))
            t8.energy -= 3.0
            self.m["t800_hits"] += 1

    @staticmethod
    def _dmg_range(d):
        return (int(d * 0.75), int(d * 1.25))

    def _metrics(self, intents):
        j, t8, t1 = self.john, self.t800, self.t1000
        w = self.world
        if not t1.alive:
            return
        # detection of the T-1000 by the good guys
        if self.m["detect_time"] < 0 and (j.sees_now("T-1000", self) or t8.sees_now("T-1000", self)):
            self.m["detect_time"] = self.step_no
        if self.m["find_time"] < 0 and t1.sees_now("JOHN", self):
            self.m["find_time"] = self.step_no
        if self.m["first_engage"] < 0 and cheb(t8.pos, t1.pos) <= 1:
            self.m["first_engage"] = self.step_no
        close = cheb(j.pos, t1.pos) <= 2
        if close:
            self.m["contact_steps"] += 1
            if not self._in_contact:
                self.m["contact_episodes"] += 1
                if self.m["t800_gap_at_contact"] < 0:
                    self.m["t800_gap_at_contact"] = cheb(t8.pos, j.pos)
        self._in_contact = close
        # successful evasion: threatened (T-1000 seen within 5) then escapes it
        dist = cheb(j.pos, t1.pos)
        if j.sees_now("T-1000", self) and dist <= 5:
            self._threatened = True
            self._unseen = 0
        elif self._threatened:
            self._unseen += 1
            if dist >= 9 or self._unseen >= 10:
                self.m["evasions"] += 1
                self._threatened = False

    def _check_end(self):
        j, t1 = self.john, self.t1000
        if not j.alive:
            self.outcome = "KILLED"
        elif not t1.alive and self.cfg.t1000_enabled:
            self.outcome = "T1000_DOWN"
        elif self.world.cell(j.pos) == SAFE:
            self.outcome = "ESCAPED"
        elif self.step_no >= self.cfg.max_steps:
            self.outcome = "TIMEOUT"

    def run(self):
        while self.outcome is None:
            self.step()
        return self.result()

    def result(self):
        r = dict(version=self.cfg.name, seed=self.seed, outcome=self.outcome,
                 steps=self.step_no, john_hp=round(self.john.hp, 1),
                 t800_hp=round(self.t800.hp, 1), t1000_hp=round(self.t1000.hp, 1),
                 john_dist=self.john.distance, t800_dist=self.t800.distance,
                 t1000_dist=self.t1000.distance, stuck_john=self.john.stuck,
                 stuck_t800=self.t800.stuck, stuck_t1000=self.t1000.stuck)
        r.update(self.m)
        return r

    def write_log(self, path):
        if not self.log:
            return
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(self.log[0].keys()))
            wr.writeheader()
            wr.writerows(self.log)
