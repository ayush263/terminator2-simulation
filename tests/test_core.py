"""Unit tests for the simulation core (run: python -m unittest discover -s tests -v)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t2sim.config import PRESETS, ALL
from t2sim.simulation import Simulation
from t2sim.world import World, BUILDING, HAZARD, GRATE, ROAD, manhattan


def open_world(n=20):
    w = World(n)
    w.exits = []
    return w


class TestWorld(unittest.TestCase):
    def test_grid_size_and_bounds(self):
        w = World.generate(1)
        self.assertGreaterEqual(w.size, 20)
        self.assertFalse(w.in_bounds((-1, 0)))
        self.assertFalse(w.in_bounds((0, w.size)))
        self.assertFalse(w.passable((-1, 5), "human"))

    def test_buildings_block_movement_and_sight(self):
        w = open_world()
        w.cells[5][5] = BUILDING
        self.assertFalse(w.passable((5, 5), "human"))
        self.assertTrue(w.los((3, 5), (4, 5)))
        self.assertFalse(w.los((3, 5), (7, 5)))

    def test_fence_only_passable_for_t1000(self):
        w = open_world()
        w.cells[3][3] = GRATE
        self.assertFalse(w.passable((3, 3), "human"))
        self.assertFalse(w.passable((3, 3), "t800"))
        self.assertTrue(w.passable((3, 3), "t1000"))
        self.assertTrue(w.los((1, 3), (6, 3)))     # fences are see-through

    def test_every_generated_map_connected(self):
        for seed in range(1000, 1030):
            w = World.generate(seed)
            reach = w.flood(w.starts["john"], "human")
            self.assertTrue(all(e in reach for e in w.exits))


class TestAStar(unittest.TestCase):
    def test_shortest_path_open_grid(self):
        w = open_world()
        path, cost = w.astar((0, 0), [(5, 3)], "human")
        self.assertEqual(len(path), 8)
        self.assertEqual(cost, 8)

    def test_routes_around_wall(self):
        w = open_world()
        for y in range(0, 10):
            w.cells[y][5] = BUILDING
        path, _ = w.astar((2, 2), [(8, 2)], "human")
        self.assertTrue(all(w.cell(p) != BUILDING for p in path))
        self.assertGreater(len(path), 6)

    def test_no_path_returns_none(self):
        w = open_world(10)
        for y in range(10):
            w.cells[y][5] = BUILDING
        path, cost = w.astar((1, 1), [(8, 1)], "human")
        self.assertIsNone(path)

    def test_hazard_cost_causes_detour(self):
        w = open_world()
        for x in range(4, 7):
            w.cells[5][x] = HAZARD
        haz = {(x, 5): 8.0 for x in range(4, 7)}
        direct, _ = w.astar((0, 5), [(10, 5)], "human")
        avoid, _ = w.astar((0, 5), [(10, 5)], "human", haz)
        self.assertTrue(any(p in haz for p in direct))
        self.assertFalse(any(p in haz for p in avoid))

    def test_t1000_uses_fence_shortcut(self):
        w = open_world()
        for y in range(0, 20):
            w.cells[y][5] = GRATE
        human, _ = w.astar((2, 2), [(8, 2)], "human")
        liquid, _ = w.astar((2, 2), [(8, 2)], "t1000")
        self.assertIsNone(human)
        self.assertEqual(len(liquid), 6)


class TestSimulation(unittest.TestCase):
    def test_invariants_all_versions(self):
        for name in ("v1", "v2", "v3", "v4"):
            for seed in range(1000, 1012):
                sim = Simulation(PRESETS[name], seed)
                while sim.outcome is None:
                    sim.step()
                    for a in sim.agents:
                        self.assertTrue(sim.world.in_bounds(a.pos))
                        self.assertNotEqual(sim.world.cell(a.pos), BUILDING)
                        if a.kind != "t1000":
                            self.assertNotEqual(sim.world.cell(a.pos), GRATE)
                    self.assertEqual(len({a.pos for a in sim.agents if a.alive}),
                                     len([a for a in sim.agents if a.alive]))
                self.assertIn(sim.outcome, ("ESCAPED", "KILLED", "TIMEOUT", "T1000_DOWN"))
                self.assertLessEqual(sim.step_no, sim.cfg.max_steps)

    def test_deterministic_given_seed(self):
        a = Simulation(PRESETS["v4"], 1234).run()
        b = Simulation(PRESETS["v4"], 1234).run()
        self.assertEqual(a, b)

    def test_agents_do_not_follow_fixed_script(self):
        """Different maps -> different trajectories (decisions depend on state)."""
        s1 = Simulation(PRESETS["v4"], 1); s1.run()
        s2 = Simulation(PRESETS["v4"], 2); s2.run()
        self.assertNotEqual(s1.traj["JOHN"], s2.traj["JOHN"])

    def test_pin_rule_slows_t1000(self):
        sim = Simulation(PRESETS["v2"], 1000)
        w = sim.world
        # place T-800 adjacent to T-1000 far from John; T-1000 should often be pinned
        sim.t800.pos, sim.t1000.pos = (14, 14), (14, 15)
        for pos in ((14, 14), (14, 15)):
            w.cells[pos[1]][pos[0]] = ROAD
        pinned = 0
        for _ in range(10):
            before = sim.m["pinned_steps"]
            sim.t800.pos, sim.t1000.pos = (14, 14), (14, 15)
            sim.step()
            pinned += sim.m["pinned_steps"] - before
        self.assertGreater(pinned, 0)

    def test_all_configs_construct(self):
        for name, cfg in ALL.items():
            Simulation(cfg, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
