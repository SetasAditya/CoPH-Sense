"""Two sequential physical VMAS forks with latent upper-corridor clearance."""

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        assert batch_dim == 1
        self.geometry = tuple(bool(v) for v in kwargs["top_geometry"])
        self.backup_waypoints = tuple(float(v) for v in
                                      kwargs.get("backup_waypoints", (-.43, -.43)))
        self.dt = float(kwargs.get("dt", .05))
        self.scout_start_y = float(kwargs.get("scout_start_y", .4))
        self.carrier_start_y = float(kwargs.get("carrier_start_y", .0))
        world = World(batch_dim=batch_dim, device=device, dt=self.dt,
                      substeps=5, drag=.22, collision_force=900,
                      x_semidim=1.35, y_semidim=.9)
        world.add_agent(Agent(name="scout", shape=Sphere(radius=.065),
                              mass=.7, color=Color.GREEN, collide=True,
                              u_range=1., max_speed=.75, render_action=True))
        world.add_agent(Agent(name="carrier", shape=Sphere(radius=.10),
                              mass=1.8, color=Color.BLUE, collide=True,
                              u_range=1., max_speed=.55, render_action=True))
        goal = Landmark(name="goal", collide=False,
                        shape=Sphere(radius=.13), color=Color.RED)
        world.add_landmark(goal)
        for region, center in ((1, -.42), (2, .42)):
            # The lower corridor is a real geometric bypass. A deeper backup
            # waypoint extends the island downward by the corresponding
            # amount, while preserving the upper corridor's clearance.
            upper_edge = .16
            lower_edge = self.backup_waypoints[region - 1] + .27
            if lower_edge >= upper_edge:
                raise ValueError("backup waypoint does not define a lower corridor")
            island_center_y = (upper_edge + lower_edge) / 2
            island_width_y = upper_edge - lower_edge
            self._obstacle(world, f"island_{region}",
                           (center, island_center_y), (.28, island_width_y))
            if not self.geometry[region - 1]:
                self._obstacle(world, f"top_blocker_{region}",
                               (center, .43), (.20, .32))
        return world

    @staticmethod
    def _obstacle(world, name, position, size):
        landmark = Landmark(name=name, collide=True, movable=False,
                            shape=Box(length=size[0], width=size[1]),
                            color=Color.GRAY)
        landmark.fixed_position = position
        world.add_landmark(landmark)

    def reset_world_at(self, env_index=None):
        self.world.agents[0].set_pos(
            torch.tensor([-1.20, self.scout_start_y], device=self.world.device),
            batch_index=env_index)
        self.world.agents[0].set_vel(torch.zeros(2, device=self.world.device),
                                     batch_index=env_index)
        self.world.agents[1].set_pos(
            torch.tensor([-1.24, self.carrier_start_y], device=self.world.device),
            batch_index=env_index)
        self.world.agents[1].set_vel(torch.zeros(2, device=self.world.device),
                                     batch_index=env_index)
        self.world.landmarks[0].set_pos(
            torch.tensor([1.17, 0.], device=self.world.device),
            batch_index=env_index)
        for landmark in self.world.landmarks[1:]:
            landmark.set_pos(torch.tensor(landmark.fixed_position,
                                          device=self.world.device),
                             batch_index=env_index)

    def reward(self, agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent):
        teammate = self.world.agents[1] if agent is self.world.agents[0] else self.world.agents[0]
        goal = self.world.landmarks[0]
        return torch.cat((agent.state.pos, agent.state.vel,
                          teammate.state.pos - agent.state.pos,
                          goal.state.pos - agent.state.pos), dim=-1)

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

    def info(self, agent):
        return {}
