"""Physical VMAS world for CoPH-Fork.

Information acquisition, packet delivery, and accounting live in the wrapper;
this file contains only physical entities and nonprivileged kinematic observations.
"""

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        assert batch_dim == 1, "A1 uses one world per environment for auditable replay"
        self.top_geometry = bool(kwargs.get("top_geometry", True))
        self.bottom_geometry = bool(kwargs.get("bottom_geometry", True))
        self.dt = float(kwargs.get("dt", 0.05))
        self.scout_start_y = float(kwargs.get("scout_start_y", 0.13))
        self.carrier_start_y = float(kwargs.get("carrier_start_y", -0.13))
        self.goal_y = float(kwargs.get("goal_y", 0.0))
        self.viewer_size = (1000, 700)
        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=self.dt,
            substeps=5,
            drag=0.22,
            collision_force=900,
            x_semidim=1.1,
            y_semidim=0.9,
        )

        scout = Agent(
            name="scout",
            shape=Sphere(radius=0.065),
            mass=0.7,
            color=Color.GREEN,
            collide=True,
            u_range=1.0,
            max_speed=0.75,
            render_action=True,
        )
        carrier = Agent(
            name="carrier",
            shape=Sphere(radius=0.10),
            mass=1.8,
            color=Color.BLUE,
            collide=True,
            u_range=1.0,
            max_speed=0.55,
            render_action=True,
        )
        world.add_agent(scout)
        world.add_agent(carrier)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.13),
            color=Color.RED,
        )
        world.add_landmark(goal)

        # The central island creates the fork. Route-specific blockers encode
        # latent clearance; they are physical colliders but absent from actor observations.
        self._add_obstacle(world, "fork_island", (0.05, 0.0), (0.62, 0.34))
        if not self.top_geometry:
            self._add_obstacle(world, "top_blocker", (0.12, 0.47), (0.20, 0.34))
        if not self.bottom_geometry:
            self._add_obstacle(world, "bottom_blocker", (0.12, -0.47), (0.20, 0.34))
        return world

    @staticmethod
    def _add_obstacle(world, name, position, size):
        obstacle = Landmark(
            name=name,
            collide=True,
            movable=False,
            shape=Box(length=size[0], width=size[1]),
            color=Color.GRAY,
        )
        obstacle.fixed_position = position
        world.add_landmark(obstacle)

    def reset_world_at(self, env_index=None):
        self.world.agents[0].set_pos(
            torch.tensor([-0.88, self.scout_start_y], device=self.world.device),
            batch_index=env_index
        )
        self.world.agents[0].set_vel(
            torch.zeros(2, device=self.world.device), batch_index=env_index
        )
        self.world.agents[1].set_pos(
            torch.tensor([-0.94, self.carrier_start_y], device=self.world.device),
            batch_index=env_index
        )
        self.world.agents[1].set_vel(
            torch.zeros(2, device=self.world.device), batch_index=env_index
        )
        self.world.landmarks[0].set_pos(
            torch.tensor([0.91, self.goal_y], device=self.world.device),
            batch_index=env_index
        )
        for landmark in self.world.landmarks[1:]:
            landmark.set_pos(
                torch.tensor(landmark.fixed_position, device=self.world.device),
                batch_index=env_index,
            )

    def reward(self, agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent):
        teammate = self.world.agents[1] if agent is self.world.agents[0] else self.world.agents[0]
        goal = self.world.landmarks[0]
        return torch.cat(
            (
                agent.state.pos,
                agent.state.vel,
                teammate.state.pos - agent.state.pos,
                goal.state.pos - agent.state.pos,
            ),
            dim=-1,
        )

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

    def info(self, agent):
        return {}
