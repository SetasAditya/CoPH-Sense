"""Deterministic continuous terrain and geometry map groups for E2.

Only the simulator/evaluator may hold a TerrainMap. Actors receive observations
through a separate environment interface; this module must never be imported
by an actor or candidate scorer at inference time.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


GRID_SIZE = 60
METERS = 12.0
RESOLUTION = METERS / GRID_SIZE
START_CELL = (5, GRID_SIZE // 2)
GOAL_CELL = (GRID_SIZE - 6, GRID_SIZE // 2)
FAMILIES = ("open", "bottleneck", "labyrinth", "deep_labyrinth", "irregular")


@dataclass(frozen=True)
class TerrainMap:
    family: str
    parent_seed: int
    occupancy: np.ndarray
    traction: np.ndarray
    obstacle_boxes: tuple
    patch_metadata: tuple
    surface: np.ndarray = None  # latent support/roughness; None only for legacy parent maps

    @property
    def risk(self):
        traction_risk = np.clip((.55 - self.traction) / .40, 0., 1.)
        if self.surface is None:
            return traction_risk.astype(np.float32)
        return np.clip(.4 * self.surface + .4 * traction_risk
                       + .2 * self.surface * traction_risk, 0., 1.).astype(np.float32)

    def digest(self):
        digest = hashlib.sha256()
        digest.update(self.family.encode())
        digest.update(str(self.parent_seed).encode())
        digest.update(np.ascontiguousarray(self.occupancy).tobytes())
        digest.update(np.ascontiguousarray(self.traction).tobytes())
        if self.surface is not None:
            digest.update(np.ascontiguousarray(self.surface).tobytes())
        digest.update(json.dumps(self.obstacle_boxes).encode())
        return digest.hexdigest()


def _rect(occupancy, boxes, x0, x1, y0, y1):
    x0, x1 = sorted((max(0, int(x0)), min(GRID_SIZE, int(x1))))
    y0, y1 = sorted((max(0, int(y0)), min(GRID_SIZE, int(y1))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    occupancy[x0:x1, y0:y1] = True
    boxes.append((x0, x1, y0, y1))


def _geometry(rng, family):
    occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    boxes = []
    if family == "open":
        for _ in range(int(rng.integers(5, 9))):
            x = int(rng.integers(12, 48))
            y = int(rng.integers(8, 52))
            sx = int(rng.integers(3, 8))
            sy = int(rng.integers(3, 9))
            _rect(occupancy, boxes, x, x + sx, y, y + sy)
    elif family == "bottleneck":
        for x in (20, 38):
            center = int(rng.integers(14, 47))
            gap = int(rng.integers(7, 13))
            width = int(rng.integers(2, 5))
            _rect(occupancy, boxes, x, x + width, 4, center - gap // 2)
            _rect(occupancy, boxes, x, x + width, center + gap // 2, 56)
        for _ in range(3):
            x = int(rng.integers(10, 50))
            y = int(rng.integers(8, 50))
            _rect(occupancy, boxes, x, x + 3, y, y + 4)
    elif family in ("labyrinth", "deep_labyrinth"):
        walls = ((15, 25, 35, 45) if family == "labyrinth"
                 else (12, 19, 26, 33, 40, 47))
        for x in walls:
            gaps = sorted(rng.choice(np.arange(8, 52), size=2, replace=False))
            spans = ((4, int(gaps[0]) - 3),
                     (int(gaps[0]) + 4, int(gaps[1]) - 3),
                     (int(gaps[1]) + 4, 56))
            for a, b in spans:
                _rect(occupancy, boxes, x, x + 2, a, b)
        for _ in range(5 if family == "labyrinth" else 8):
            x = int(rng.integers(10, 49))
            y = int(rng.integers(8, 50))
            _rect(occupancy, boxes, x, x + int(rng.integers(3, 8)), y, y + 2)
    elif family == "irregular":
        for _ in range(int(rng.integers(8, 14))):
            x = int(rng.integers(10, 49))
            y = int(rng.integers(6, 53))
            width = int(rng.integers(2, 8))
            height = int(rng.integers(2, 8))
            _rect(occupancy, boxes, x, x + width, y, y + height)
    else:
        raise ValueError(f"unknown topology family {family}")
    # Spawn and goal clearances are part of the public task geometry. Keep
    # the grid and the VMAS box list exactly consistent.
    boxes = [box for box in boxes if not (box[0] < 10 and box[1] > 0 and box[2] < 37 and box[3] > 23)
             and not (box[0] < 60 and box[1] > 50 and box[2] < 37 and box[3] > 23)]
    occupancy.fill(False)
    for x0, x1, y0, y1 in boxes:
        occupancy[x0:x1, y0:y1] = True
    return occupancy, tuple(boxes)


def _reachable(occupancy):
    from collections import deque
    free = ~occupancy
    # Disc clearance approximated by one-cell inflation for the generator audit.
    blocked = occupancy.copy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        blocked[max(0, dx):GRID_SIZE + min(0, dx),
                max(0, dy):GRID_SIZE + min(0, dy)] |= occupancy[
                    max(0, -dx):GRID_SIZE - max(0, dx),
                    max(0, -dy):GRID_SIZE - max(0, dy)]
    free = ~blocked
    if not free[START_CELL] or not free[GOAL_CELL]:
        return False
    queue = deque([START_CELL])
    seen = {START_CELL}
    while queue:
        x, y = queue.popleft()
        if (x, y) == GOAL_CELL:
            return True
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and free[nx, ny] and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return False


def generate_map(family, parent_seed):
    """Generate one map group; episode stochasticity is sampled elsewhere."""
    if family not in FAMILIES:
        raise ValueError(family)
    for attempt in range(64):
        rng = np.random.default_rng(np.random.SeedSequence([int(parent_seed), attempt]))
        occupancy, boxes = _geometry(rng, family)
        if _reachable(occupancy):
            break
    else:
        raise RuntimeError(f"could not generate traversable {family} map for seed {parent_seed}")
    x, y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    traction = np.full((GRID_SIZE, GRID_SIZE), .86, dtype=np.float64)
    count = int(rng.integers(5, 11))
    patches = []
    for _ in range(count):
        cx = float(rng.uniform(9, 51))
        cy = float(rng.uniform(5, 55))
        sx = float(rng.uniform(3, 10))
        sy = float(rng.uniform(3, 10))
        severity = float(rng.uniform(.25, .72))
        traction -= severity * np.exp(-.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        patches.append((cx, cy, sx, sy, severity))
    traction = np.clip(traction, .15, .95).astype(np.float32)
    return TerrainMap(family, int(parent_seed), occupancy, traction, boxes,
                      tuple(patches))


def realize_map(parent, realization_seed):
    """Vary latent material severity under a fixed parent geometry/patch support."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(parent.parent_seed), int(realization_seed), 9217]))
    x, y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    traction = np.full((GRID_SIZE, GRID_SIZE), .86, dtype=np.float64)
    patches = []
    for cx, cy, sx, sy, base_severity in parent.patch_metadata:
        severity = float(np.clip(base_severity * rng.uniform(.75, 1.25), .15, .85))
        traction -= severity * np.exp(-.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        patches.append((cx, cy, sx, sy, severity))
    # Surface support/roughness is a separate latent field from friction.
    # Its independent patches make either measurement incomplete on its own.
    surface_rng = np.random.default_rng(np.random.SeedSequence(
        [int(parent.parent_seed), int(realization_seed), 19343]))
    surface = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for _ in range(int(surface_rng.integers(5, 10))):
        cx, cy = surface_rng.uniform(9, 51), surface_rng.uniform(5, 55)
        sx, sy = surface_rng.uniform(3, 10), surface_rng.uniform(3, 10)
        severity = surface_rng.uniform(.25, .75)
        surface += severity * np.exp(-.5 * (((x - cx) / sx) ** 2
                                             + ((y - cy) / sy) ** 2))
    return TerrainMap(parent.family, parent.parent_seed, parent.occupancy.copy(),
                      np.clip(traction, .15, .95).astype(np.float32),
                      parent.obstacle_boxes, tuple(patches),
                      np.clip(surface, 0., 1.).astype(np.float32))
