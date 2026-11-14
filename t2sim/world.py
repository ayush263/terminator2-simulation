"""Grid world: map generation, passability, line of sight."""
import heapq
import random
from collections import deque

ROAD, BUILDING, HAZARD, GRATE, SAFE, STATION = range(6)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EXITS = [(29, 12), (12, 29), (29, 29)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class World:
    """30x30 city. cells[y][x]; positions are (x, y).

    ROAD      walkable by all
    BUILDING  blocks movement and line of sight
    HAZARD    molten-steel / fire; walkable but damaging
    GRATE     fence: only the liquid-metal T-1000 can pass (see-through)
    SAFE      John's escape points
    STATION   repair / recharge point
    """

    def __init__(self, size=30):
        self.size = size
        self.cells = [[ROAD] * size for _ in range(size)]
        self.exits = list(EXITS)
        self.stations = []
        self.starts = {}
        self.fire_added = 0

    # ---- basic queries -------------------------------------------------
    def in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def cell(self, p):
        return self.cells[p[1]][p[0]]

    def passable(self, p, kind):
        if not self.in_bounds(p):
            return False
        c = self.cell(p)
        if c == BUILDING:
            return False
        if c == GRATE and kind != "t1000":
            return False
        return True

    def neighbours(self, p, kind):
        for dx, dy in N4:
            q = (p[0] + dx, p[1] + dy)
            if self.passable(q, kind):
                yield q

    def hazard_cells(self):
        return {(x, y) for y in range(self.size) for x in range(self.size)
                if self.cells[y][x] == HAZARD}

    # ---- line of sight (Bresenham, buildings block) ----------------------
    def los(self, a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            if (x, y) != (x0, y0) and self.cells[y][x] == BUILDING:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def can_see(self, a, b, radius):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= radius * radius and self.los(a, b)

    def visible_cells(self, p, radius):
        out = set()
        for y in range(max(0, p[1] - radius), min(self.size, p[1] + radius + 1)):
            for x in range(max(0, p[0] - radius), min(self.size, p[0] + radius + 1)):
                if self.can_see(p, (x, y), radius):
                    out.add((x, y))
        return out

    def bfs_dist(self, start, kind, limit=40):
        """Step distance from start to every reachable cell (hazards ignored)."""
        dist = {start: 0}
        dq = deque([start])
        while dq:
            p = dq.popleft()
            if dist[p] >= limit:
                continue
            for q in self.neighbours(p, kind):
                if q not in dist:
                    dist[q] = dist[p] + 1
                    dq.append(q)
        return dist

    # ---- A* path planning ---------------------------------------------------
    def astar(self, start, goals, kind, extra=None, blocked=None):
        """A* over the 4-connected grid with Manhattan heuristic (admissible,
        since every step costs >= 1). `extra` maps cell -> additional cost
        (hazards, threat zones). Returns (path excluding start, cost) or
        (None, inf)."""
        goals = set(goals)
        if start in goals:
            return [], 0.0
        extra = extra or {}
        blocked = blocked or ()
        gl = list(goals)

        def h(p):
            return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for g in gl)

        openq = [(h(start), 0.0, start)]
        best = {start: 0.0}
        came = {}
        while openq:
            f, gc, p = heapq.heappop(openq)
            if gc > best.get(p, 1e18):
                continue
            if p in goals:
                path = [p]
                while path[-1] in came:
                    path.append(came[path[-1]])
                path.pop()          # drop start
                path.reverse()
                return path, gc
            for q in self.neighbours(p, kind):
                if q in blocked and q not in goals:
                    continue
                nc = gc + 1.0 + extra.get(q, 0.0)
                if nc < best.get(q, 1e18):
                    best[q] = nc
                    came[q] = p
                    heapq.heappush(openq, (nc + h(q), nc, q))
        return None, float("inf")

    # ---- generation -------------------------------------------------------
    @staticmethod
    def generate(seed, size=30):
        rng = random.Random(seed)
        for _ in range(100):
            w = World(size)
            w._build(rng)
            if w._valid():
                return w
        raise RuntimeError("could not generate a connected map")

    def _build(self, rng):
        n = self.size
        # city blocks: 5x5 blocks of 4x4 buildings separated by 2-wide roads
        for i in range(5):
            for j in range(5):
                if rng.random() < 0.15:      # open plaza
                    continue
                x0, y0 = 1 + 6 * i, 1 + 6 * j
                for y in range(y0, y0 + 4):
                    for x in range(x0, x0 + 4):
                        self.cells[y][x] = BUILDING
        # molten-steel spills (2x2) on roads
        for _ in range(5):
            x, y = rng.randrange(2, n - 3), rng.randrange(2, n - 3)
            patch = [(x + a, y + b) for a in (0, 1) for b in (0, 1)]
            if all(self.cells[q[1]][q[0]] == ROAD for q in patch):
                for q in patch:
                    self.cells[q[1]][q[0]] = HAZARD
        # fences across 2-wide roads (T-1000 can slip through)
        for _ in range(4):
            x, y = rng.randrange(2, n - 3), rng.randrange(2, n - 3)
            patch = [(x, y), (x + 1, y)] if rng.random() < 0.5 else [(x, y), (x, y + 1)]
            if all(self.cells[q[1]][q[0]] == ROAD for q in patch):
                for q in patch:
                    self.cells[q[1]][q[0]] = GRATE
        for e in self.exits:
            self.cells[e[1]][e[0]] = SAFE
        road = [(x, y) for y in range(n) for x in range(n) if self.cells[y][x] == ROAD]
        self.stations = rng.sample([p for p in road if 6 < p[0] < 24 or 6 < p[1] < 24], 3)
        for s in self.stations:
            self.cells[s[1]][s[0]] = STATION
        road = [p for p in road if p not in self.stations]
        # starting positions
        john_c = [p for p in road if p[0] <= 9 and p[1] <= 9]
        j = rng.choice(john_c)
        near = [p for p in road if 1 <= manhattan(p, j) <= 3]
        t8 = rng.choice(near)
        far = [p for p in road if 8 <= manhattan(p, j) <= 16]
        t1 = rng.choice(far)
        self.starts = {"john": j, "t800": t8, "t1000": t1}

    def _valid(self):
        s = self.starts["john"]
        reach = self.flood(s, "human")
        ok = all(e in reach for e in self.exits)
        return ok and self.starts["t800"] in reach and self.starts["t1000"] in self.flood(self.starts["t1000"], "t1000")

    def flood(self, start, kind):
        seen = {start}
        dq = deque([start])
        while dq:
            p = dq.popleft()
            for q in self.neighbours(p, kind):
                if q not in seen:
                    seen.add(q)
                    dq.append(q)
        return seen
