"""Offline route/risk rendering from a saved evaluator artifact, never actor input."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _actor_belief_frames(events, steps, agent_name):
    """Reconstruct only delivered/local evidence; no evaluator map is read."""
    geometry = np.full((60, 60), np.nan, dtype=np.float32)
    appearance = np.full((60, 60), np.nan, dtype=np.float32)
    traction = np.full((60, 60), np.nan, dtype=np.float32)
    acquired = {}
    received = set()
    cursor = 0
    frames = []

    def apply_evidence(event):
        grid = geometry if event["modality"] == "geometry" else traction
        for cell, value in zip(event["cells"], event["values"]):
            grid[tuple(cell)] = float(value)

    for step in steps:
        while cursor < len(events) and events[cursor]["step"] <= step:
            event = events[cursor]
            cursor += 1
            if event["type"] == "passive_observation" and event["agent"] == agent_name:
                for cell, occupied, visible in zip(event["cells"],
                                                    event["geometry"],
                                                    event["appearance"]):
                    geometry[tuple(cell)] = float(occupied)
                    appearance[tuple(cell)] = float(visible)
            elif event["type"] == "acquisition" and event["valid"]:
                acquired[event["evidence_id"]] = event
                if event["agent"] == agent_name:
                    apply_evidence(event)
            elif (event["type"] == "packet_delivery"
                  and event["kind"] == "evidence" and event["delivered"]
                  and event["receiver"] == agent_name):
                evidence_id = event["evidence_id"]
                if evidence_id not in acquired:
                    raise ValueError(f"missing acquired evidence {evidence_id}")
                apply_evidence(acquired[evidence_id])
                received.add(evidence_id)
        frames.append((geometry.copy(), appearance.copy(), traction.copy(),
                       len(received)))
    return frames


def render_artifact(directory, stride=12):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    states = np.load(directory / "states.npz")
    truth = np.load(directory / "truth_map.npz")
    positions = states["positions"]
    risk = np.ma.masked_where(truth["occupancy"].T, truth["risk"].T)
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#30343b")
    output = directory / "trajectory.png"
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.imshow(risk, origin="lower", extent=(-6, 6, -6, 6), cmap=cmap,
                vmin=0, vmax=1)
    for i, (name, color) in enumerate((("scout", "#12bd56"),
                                       ("carrier", "#1887d3"))):
        axis.plot(positions[:, i, 0], positions[:, i, 1], color=color,
                  linewidth=1.6, label=name)
        axis.scatter(*positions[0, i], color=color, marker="o", s=50,
                     edgecolors="black", linewidths=.5)
        axis.scatter(*positions[-1, i], color=color, marker="X", s=70,
                     edgecolors="black", linewidths=.5)
    axis.scatter(5, 0, color="#b643ff", marker="*", s=150,
                 edgecolors="black", linewidths=.5, label="goal")
    axis.set(xlim=(-6, 6), ylim=(-6, 6), xlabel="x (m)", ylabel="y (m)",
             title=f"{metadata['family']} / {metadata['parent_seed']} — "
                   f"{'success' if metadata['success'] else metadata['failure_reason']}")
    axis.set_aspect("equal")
    axis.legend(loc="upper right")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)

    # A compact animated replay is useful for checking stale messages and
    # apparent late sensing; the authoritative data remain the saved trace.
    frames = []
    frame_indices = list(range(0, len(positions), max(1, stride)))
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    actor_states = _actor_belief_frames(events, frame_indices, "carrier")
    actor_frames = []
    for step, (known_geometry, appearance, traction, received_count) in zip(
            frame_indices, actor_states):
        fig, axis = plt.subplots(figsize=(5, 5), dpi=85)
        axis.imshow(risk, origin="lower", extent=(-6, 6, -6, 6), cmap=cmap,
                    vmin=0, vmax=1)
        for i, color in enumerate(("#12bd56", "#1887d3")):
            axis.plot(positions[:step + 1, i, 0], positions[:step + 1, i, 1],
                      color=color, linewidth=1.5)
            axis.scatter(*positions[step, i], color=color, s=65,
                         edgecolors="black", linewidths=.5)
        axis.scatter(5, 0, color="#b643ff", marker="*", s=130)
        axis.set(xlim=(-6, 6), ylim=(-6, 6), xticks=(), yticks=(),
                 title=f"step {step}  |  risk {states['material_exposure'][step]:.3f}")
        axis.set_aspect("equal")
        fig.canvas.draw()
        frames.append(Image.frombytes("RGB", fig.canvas.get_width_height(),
                                      fig.canvas.tostring_rgb()))
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(5, 5), dpi=85)
        belief = np.ma.masked_invalid(appearance.T)
        belief_cmap = plt.colormaps["viridis"].copy()
        belief_cmap.set_bad("#e8ebed")
        axis.imshow(belief, origin="lower", extent=(-6, 6, -6, 6),
                    cmap=belief_cmap, vmin=0, vmax=1)
        obstacle = np.ma.masked_where(known_geometry.T != 1., known_geometry.T)
        axis.imshow(obstacle, origin="lower", extent=(-6, 6, -6, 6),
                    cmap="gray_r", vmin=0, vmax=1)
        measured = np.argwhere(np.isfinite(traction))
        if len(measured):
            axis.scatter(-6 + (measured[:, 0] + .5) * .2,
                         -6 + (measured[:, 1] + .5) * .2,
                         marker="s", s=12, facecolors="none",
                         edgecolors="#ea4f12", linewidths=.8)
        axis.plot(positions[:step + 1, 1, 0], positions[:step + 1, 1, 1],
                  color="#1887d3", linewidth=1.5)
        axis.scatter(*positions[step, 1], color="#1887d3", s=65,
                     edgecolors="black", linewidths=.5)
        axis.scatter(5, 0, color="#b643ff", marker="*", s=130)
        axis.set(xlim=(-6, 6), ylim=(-6, 6), xticks=(), yticks=(),
                 title=f"carrier belief | step {step} | received {received_count}")
        axis.set_aspect("equal")
        fig.canvas.draw()
        actor_frames.append(Image.frombytes("RGB", fig.canvas.get_width_height(),
                                            fig.canvas.tostring_rgb()))
        plt.close(fig)
    gif = directory / "trajectory.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True)
    actor_gif = directory / "carrier_actor_view.gif"
    actor_frames[0].save(actor_gif, save_all=True,
                         append_images=actor_frames[1:], duration=90,
                         loop=0, optimize=True)
    return output, gif, actor_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--stride", type=int, default=12)
    args = parser.parse_args()
    print(*render_artifact(args.artifact, args.stride), sep="\n")


if __name__ == "__main__":
    main()
