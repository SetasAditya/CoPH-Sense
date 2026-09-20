#!/usr/bin/env python3
"""Visualize physically executed safe shortcut and reliable backup routes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import OUT, load_actors, run_episode
from coph_fork.run_moving_oracle_a65_v3 import BASE


def main():
    torch.set_num_threads(2)
    a4, gate = load_actors()
    world = ForkWorld(top_geometry=True, top_traction=0.90,
                      bottom_geometry=True, bottom_traction=0.15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, policy, title in zip(axes, ("always", "never"),
                                 ("Inspect: certified upper shortcut",
                                  "Skip: lower backup")):
        row = run_episode(world, BASE, policy, 1701, a4, gate,
                          rollout_forecast=True)
        path = row["trajectory"]
        scout = np.asarray([entry["scout_position"] for entry in path])
        carrier = np.asarray([entry["carrier_position"] for entry in path])
        carrier_length = np.linalg.norm(np.diff(carrier, axis=0), axis=1).sum()
        ax.add_patch(Rectangle((-.26, -.17), .62, .34, color="gray", alpha=.35))
        ax.plot(scout[:, 0], scout[:, 1], color="#54a24b", label="Scout")
        ax.plot(carrier[:, 0], carrier[:, 1], color="#4c78a8", label="Carrier")
        ax.scatter([-.34], [.43], marker="x", s=70, color="black", label="Probe")
        ax.scatter([.91], [BASE.goal_y], marker="*", s=180, color="#e45756",
                   label="Goal")
        ax.axvline(BASE.decision_x, color="black", ls="--", alpha=.45)
        ax.set(xlim=(-1.04, 1.05), ylim=(-.88, .76),
               xlabel="x position (m)", ylabel="y position (m)",
               title=f"{title}\ncarrier {carrier_length:.3f} m; team cost {row['team_cost']:.3f}")
        ax.set_aspect("equal")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / "v3_physical_routes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
