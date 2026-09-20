"""Matched VMAS direct/pH state transitions with the A5 packet and cost model."""

from dataclasses import asdict

import numpy as np
import torch

from .environment import AGENTS, CostLedger
from .ph_factorial_controller import ForceConfig, execute_step
from .two_fork_environment import TwoForkEnv


class FactorialEnv(TwoForkEnv):
    def __init__(self, *args, scheme="direct", force_config=None, **kwargs):
        if scheme not in ("direct", "ph"):
            raise ValueError(scheme)
        self.scheme = scheme
        self.force_config = force_config or ForceConfig()
        self.energy_trace = []
        super().__init__(*args, **kwargs)

    def reset(self, seed=None):
        result = super().reset(seed)
        self.energy_trace = []
        return result

    def step(self, actions, targets):
        if self.done:
            raise RuntimeError("episode is done; call reset")
        if set(actions) != set(AGENTS) or set(targets) != set(AGENTS):
            raise ValueError("both agents require actions and public targets")
        for action in actions.values():
            action.validate()
        self.action_trace.append({name: asdict(actions[name]) for name in AGENTS})
        incremental = CostLedger(time=self.config.time_cost)
        positions_before = self.agent_positions
        carrier_x = float(positions_before["carrier"][0])
        decision_open = self.route_commitment is None and carrier_x < self.config.decision_x
        self._deliver_due_packets(decision_open, incremental)
        for name in AGENTS:
            self._sense(name, actions[name], incremental)
        for name in AGENTS:
            self._send(name, actions[name], incremental)
        self._deliver_due_packets(decision_open, incremental)

        diagnostics = {}
        effort = 0.
        for index, name in enumerate(AGENTS):
            actor = self.physics.world.agents[index]
            q = positions_before[name]
            v = actor.state.vel[0].detach().cpu().numpy().copy()
            mass = float(actor.mass)
            radius = float(actor.shape.radius)
            max_speed = float(actor.max_speed)
            q_next, v_next, diagnostic = execute_step(
                q, v, targets[name], self, name, self.scheme,
                self.force_config, mass, radius, max_speed,
                self._traction_at(q))
            actor.set_pos(torch.tensor(q_next, dtype=torch.float32), batch_index=0)
            actor.set_vel(torch.tensor(v_next, dtype=torch.float32), batch_index=0)
            effort += min(diagnostic["force_norm"], self.force_config.max_force) ** 2
            diagnostics[name] = diagnostic
        self.energy_trace.append(diagnostics)
        self.step_index += 1
        positions = self.agent_positions
        if self.route_commitment is None and positions["carrier"][0] >= self.config.decision_x:
            self.route_commitment = "top" if positions["carrier"][1] >= 0 else "bottom"
            self.events.append({"type": "route_commitment", "region": 1,
                                "step": self.step_index, "route": self.route_commitment})
        self.route_commitments[1] = self.route_commitment
        if self.route_commitments[2] is None and positions["carrier"][0] >= self.config.decision_x2:
            route = "top" if positions["carrier"][1] >= 0 else "bottom"
            self.route_commitments[2] = route
            self.events.append({"type": "route_commitment", "region": 2,
                                "step": self.step_index, "route": route})
        incremental.motion += self.config.motion_cost * effort * self.config.dt
        carrier = self.physics.world.agents[1]
        speed = float(torch.linalg.vector_norm(carrier.state.vel[0]).item())
        traction = self._traction_at(positions["carrier"])
        incremental.risk += (self.config.risk_cost
                             * max(0., self.config.traction_safe_threshold - traction)
                             * speed * self.config.dt)
        collisions = self._collision_count()
        incremental.collision += self.config.collision_cost * collisions
        if collisions:
            self.events.append({"type": "collision", "step": self.step_index,
                                "count": collisions,
                                "positions": {k: v.tolist() for k, v in positions.items()}})
        self.success = self._goal_reached()
        if self.success:
            self.done = True
        elif self.step_index >= self.config.horizon:
            self.done = True
            incremental.terminal_failure += self.config.terminal_failure_cost
        self.ledger.add(incremental)
        self.state_trace.append(self._state_record())
        info = {"step": self.step_index, "success": self.success,
                "done": self.done, "route_commitment": self.route_commitment,
                "route_commitments": dict(self.route_commitments),
                "incremental_cost": asdict(incremental),
                "incremental_total": incremental.total,
                "episode_ledger": asdict(self.ledger),
                "episode_total": self.ledger.total, "collisions": collisions,
                "energy": diagnostics}
        return self.observations(), -incremental.total, self.done, info
