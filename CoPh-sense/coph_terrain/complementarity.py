"""Separate procedural E2-Complementarity worlds and oracle admission.

The Natural generator and executor are untouched. Public material regions are
constant within each rectangle, so a local surface/traction sample identifies
that region's corresponding latent parameter. All methods share this exact
inference rule. Admission uses only physical counterfactual values.
"""

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json

import numpy as np

from .environment import (CoPHTerrainEnv, TerrainAction, TerrainConfig,
                          cell_to_position)
from .generator import GRID_SIZE, TerrainMap
from .physical_audit import AuditChoice, run_continuation


REGIONS = ((9, 27, 23, 37), (28, 46, 23, 37))
TARGETS = ((10, 30), (30, 30))
MARGIN = .02


@dataclass(frozen=True)
class ChallengeSpec:
    parent_seed: int
    surface_bad: tuple
    traction_bad: tuple
    known_second: bool = True


class ComplementarityEnv(CoPHTerrainEnv):
    """Same VMAS physics and protocol, with a known block-material prior."""

    def __init__(self, spec, seed=None, executor="ph", device="cpu", config=None):
        self.challenge_spec = spec
        super().__init__("open", spec.parent_seed, 0,
                         seed=spec.parent_seed + 9 if seed is None else seed,
                         executor=executor, device=device, config=config)
        surface = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        traction = np.full((GRID_SIZE, GRID_SIZE), .90, dtype=np.float32)
        for region, g_bad, t_bad in zip(REGIONS, spec.surface_bad,
                                         spec.traction_bad):
            x0, x1, y0, y1 = region
            surface[x0:x1, y0:y1] = .95 if g_bad else .02
            traction[x0:x1, y0:y1] = .18 if t_bad else .90
        self._truth = TerrainMap("open", spec.parent_seed,
                                 self.parent_map.occupancy.copy(), traction,
                                 self.parent_map.obstacle_boxes,
                                 self.parent_map.patch_metadata, surface)
        self.reset()

    def reset(self):
        observation = super().reset()
        # The scout starts ahead, while the carrier has a genuine lead-time
        # window in which delivered evidence can alter its first route.
        for index, cell in enumerate(((5, 30), (1, 30))):
            agent = self.physics.world.agents[index]
            agent.state.pos[0] = agent.state.pos.new_tensor(cell_to_position(cell))
            agent.state.vel[0].zero_()
        for name in ("scout", "carrier"):
            self.appearance_belief[name].fill(np.nan)
            self.last_passive_cell[name] = None
        self.events.clear()
        self._default_observe()
        self.state_trace = [self._state_record()]
        if getattr(self, "challenge_spec", None) is not None and \
                self.challenge_spec.known_second:
            x0, x1, y0, y1 = REGIONS[1]
            for name in ("scout", "carrier"):
                self.known_surface[name][x0:x1, y0:y1] = \
                    self._truth.surface[x0:x1, y0:y1]
                self.known_traction[name][x0:x1, y0:y1] = \
                    self._truth.traction[x0:x1, y0:y1]
            observation = self.observations()
        return observation

    def observations(self):
        result = super().observations()
        for observation in result.values():
            observation["public_material_regions"] = REGIONS
            observation["public_material_prior"] = (
                (.5, .5), (0., 0.) if self.challenge_spec.known_second
                else (.5, .5))
        return result

    def _apply_evidence(self, name, evidence):
        super()._apply_evidence(name, evidence)
        grid = (self.known_surface[name] if evidence.modality == "geometry"
                else self.known_traction[name])
        for region in REGIONS:
            x0, x1, y0, y1 = region
            values = [value for cell, value in zip(evidence.cells, evidence.values)
                      if x0 <= cell[0] < x1 and y0 <= cell[1] < y1]
            if values:
                # The within-region constant-field assumption is public.
                grid[x0:x1, y0:y1] = float(values[0])

    def forget_delivered_evidence(self, name, evidence, prior):
        """Undo the full public-region inference in a matched memory intervention."""
        grid = (self.known_surface[name] if evidence.modality == "geometry"
                else self.known_traction[name])
        old = (prior.known_surface[name] if evidence.modality == "geometry"
               else prior.known_traction[name])
        for region in REGIONS:
            x0, x1, y0, y1 = region
            if any(x0 <= cell[0] < x1 and y0 <= cell[1] < y1
                   for cell in evidence.cells):
                grid[x0:x1, y0:y1] = old[x0:x1, y0:y1]
        for cell in evidence.cells:
            grid[cell] = old[cell]

    def replay_payload(self):
        payload = super().replay_payload()
        payload["challenge_spec"] = asdict(self.challenge_spec)
        payload.pop("sha256")
        payload["sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True).encode()).hexdigest()
        return payload

    @classmethod
    def replay(cls, payload):
        spec = dict(payload["challenge_spec"])
        spec["surface_bad"] = tuple(spec["surface_bad"])
        spec["traction_bad"] = tuple(spec["traction_bad"])
        env = cls(ChallengeSpec(**spec), seed=payload["seed"],
                  executor=payload["executor"],
                  device=payload.get("device", "cpu"),
                  config=TerrainConfig(**payload["config"]))
        for row in payload["actions"]:
            env.step({name: TerrainAction(**row[name])
                      for name in ("scout", "carrier")})
        if (env._truth.digest() != payload["map_sha256"] or
                asdict(env.ledger) != payload["ledger"] or
                env.agent_risk != payload["agent_material_exposure"]):
            raise RuntimeError("challenge deterministic replay mismatch")
        return env


def geometry_eligible(env):
    """Public feasibility prefilter; this does not inspect latent materials."""
    occupied = env._truth.occupancy
    return (not occupied[1, 30] and not occupied[5, 30] and
            not occupied[10, 30] and not occupied[30, 30] and
            not occupied[8, 30] and not occupied[27, 30] and
            all(not occupied[x, 30] for x in range(8, 46)))


def first_region_choices():
    return (AuditChoice("scout", "geometry", (8, 30), TARGETS[0]),
            AuditChoice("scout", "traction", TARGETS[0], TARGETS[0]))


def structural_acquisition_values(env):
    """Belief-expected physical branches over the four first-region states.

    This is an ex-ante value at the common initial information state, not a
    hindsight comparison within the realized hidden world. The second-region
    realization and all public geometry stay fixed. The same simulator seed
    couples appearance and packet randomness across action branches.
    """
    choices = first_region_choices()
    sets = ((), (0,), (1,), (0, 1))
    labels = ("skip", "G", "T", "GT")
    worlds = []
    for g_bad, t_bad in product((False, True), repeat=2):
        spec = ChallengeSpec(env.challenge_spec.parent_seed,
                             (g_bad, env.challenge_spec.surface_bad[1]),
                             (t_bad, env.challenge_spec.traction_bad[1]),
                             env.challenge_spec.known_second)
        worlds.append(ComplementarityEnv(spec, seed=env.seed,
                                          executor=env.executor,
                                          device=env.device))
    observations = [world.observations() for world in worlds]
    common_information = all(
        np.array_equal(observations[0][agent][key], obs[agent][key],
                       equal_nan=True)
        for obs in observations[1:]
        for agent in ("scout", "carrier")
        for key in ("known_geometry", "known_surface", "known_traction",
                    "appearance"))
    rows = [[run_continuation(world, choices, selected,
                              hold_carrier_until_delivery=True)
             for selected in sets]
            for world in worlds]
    costs = {name: float(np.mean([world_rows[index]["score_single_world"]
                                  for world_rows in rows]))
             for index, name in enumerate(labels)}
    complete = all(row["success"] for world_rows in rows for row in world_rows)
    captured = all(all(world_rows[3]["target_region_covered"])
                   for world_rows in rows)
    delivered = all(world_rows[3]["delivered"] >= 2 for world_rows in rows)
    timely = all(len(world_rows[3]["carrier_cells_at_delivery"]) >= 2 and
                 all(cell[0] < REGIONS[0][0] for cell in
                     world_rows[3]["carrier_cells_at_delivery"][:2])
                 for world_rows in rows)
    pair_margin = min(costs["skip"], costs["G"], costs["T"]) - costs["GT"]
    return {"costs": costs, "all_success": complete,
            "common_initial_information": common_information,
            "pair_readings_captured": captured,
            "pair_delivered": delivered,
            "pair_delivered_before_region": timely,
            "world_branch_costs": [[float(row["score_single_world"])
                                     for row in world_rows]
                                    for world_rows in rows],
            "pair_carrier_cells_at_delivery": [world_rows[3]
                                                ["carrier_cells_at_delivery"]
                                                for world_rows in rows],
            "pair_margin": pair_margin,
            "admitted": bool(common_information and complete and captured and delivered
                             and timely and
                             pair_margin > MARGIN)}
