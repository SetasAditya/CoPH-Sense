"""VMAS geometry/agent shell for one private procedural terrain realization."""

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        if batch_dim != 1:
            raise ValueError("E2 v1 replay currently requires batch_dim=1")
        self.obstacle_boxes = tuple(kwargs["obstacle_boxes"])
        self.dt = float(kwargs.get("dt", .05))
        world = World(batch_dim=batch_dim, device=device, dt=self.dt,
                      substeps=5, drag=.22, collision_force=900,
                      x_semidim=6., y_semidim=6.)
        world.add_agent(Agent(name="scout", shape=Sphere(radius=.065),
                              mass=.7, color=Color.GREEN, collide=True,
                              u_range=1., max_speed=.75, render_action=True))
        world.add_agent(Agent(name="carrier", shape=Sphere(radius=.10),
                              mass=1.8, color=Color.BLUE, collide=True,
                              u_range=1., max_speed=.55, render_action=True))
        world.add_landmark(Landmark(name="goal", collide=False,
                                    shape=Sphere(radius=.20), color=Color.RED))
        for index, (x0, x1, y0, y1) in enumerate(self.obstacle_boxes):
            size = ((x1 - x0) * .2, (y1 - y0) * .2)
            center = (-6 + (x0 + x1) * .1, -6 + (y0 + y1) * .1)
            obstacle = Landmark(name=f"obstacle_{index}", collide=True,
                                movable=False,
                                shape=Box(length=size[0], width=size[1]),
                                color=Color.GRAY)
            obstacle.fixed_position = center
            world.add_landmark(obstacle)
        return world

    def reset_world_at(self, env_index=None):
        self.world.agents[0].set_pos(
            torch.tensor((-5., .25), device=self.world.device),
            batch_index=env_index)
        self.world.agents[1].set_pos(
            torch.tensor((-5., -.25), device=self.world.device),
            batch_index=env_index)
        for agent in self.world.agents:
            agent.set_vel(torch.zeros(2, device=self.world.device),
                          batch_index=env_index)
        self.world.landmarks[0].set_pos(
            torch.tensor((5., 0.), device=self.world.device),
            batch_index=env_index)
        for obstacle in self.world.landmarks[1:]:
            obstacle.set_pos(torch.tensor(obstacle.fixed_position,
                                          device=self.world.device),
                             batch_index=env_index)

    def observation(self, agent):
        teammate = (self.world.agents[1] if agent is self.world.agents[0]
                    else self.world.agents[0])
        goal = self.world.landmarks[0]
        return torch.cat((agent.state.pos, agent.state.vel,
                          teammate.state.pos - agent.state.pos,
                          goal.state.pos - agent.state.pos), dim=-1)

    def reward(self, agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool,
                           device=self.world.device)

    def info(self, agent):
        return {}
