"""Offline paired HOLD/SEND animation for the fixed look-ahead world."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from .algorithm import acquire_to_evidence, post_reading_branch
from .environment import cell_to_position
from .lookahead import LookaheadEnv, LookaheadSpec, REGIONS, region_index
from .lookahead_dev_audit import VIEW
from .lookahead_policy import choose_region_to_inspect
from .physical_audit import run_continuation


def simulate(seed, risk):
    surface, traction = ((.95, .20) if risk == "risky" else (.04, .90))
    spec = LookaheadSpec(seed, surface=(surface, .06),
                         traction=(traction, .90), scout_start=(12, 27))
    acquired = acquire_to_evidence(LookaheadEnv(spec), VIEW)
    if acquired is None:
        raise RuntimeError("scout could not acquire canonical reading")
    snapshot = acquired.env
    packet_id = next(iter(snapshot.innovation_packets))
    branches = {}
    for label, send in (("HOLD", False), ("SEND", True)):
        _, decision_state = post_reading_branch(
            snapshot, packet_id, send, continue_mission=False)
        choice = choose_region_to_inspect(decision_state, "carrier")
        choices = () if choice is None else (choice,)
        selected = () if choice is None else (0,)
        result, final = run_continuation(
            decision_state, choices, selected, transmit=False,
            return_environment=True)
        carrier_new = [item for item in final.owned["carrier"].values()
                       if item.evidence_id not in snapshot.owned["carrier"] and
                       item.evidence_id not in final.innovation_packets]
        scan_step = min((item.acquired_step for item in carrier_new),
                        default=None)
        scan_region = (region_index(choice.target_region)
                       if carrier_new and choice is not None else None)
        delivery_step = next((event["step"] for event in final.events if
                              event["type"] == "packet_delivery" and
                              event.get("evidence_id") == packet_id and
                              event.get("kind") == "evidence" and
                              event.get("delivered")), None)
        branches[label] = {"trace": final.state_trace,
                           "cost": result["score_single_world"],
                           "scan_step": scan_step,
                           "scan_region": scan_region,
                           "delivery_step": delivery_step,
                           "choice": choice}
    return snapshot, branches


def draw_background(ax):
    wall = Rectangle((-2.8, 0.), 5.2, .2, facecolor="#263238",
                     edgecolor="none", zorder=1)
    ax.add_patch(wall)
    for region, (_, _, y0, y1) in enumerate(REGIONS):
        left = cell_to_position((18, y0))[0]-.1
        bottom = cell_to_position((18, y0))[1]-.1
        ax.add_patch(Rectangle((left, bottom), 2., (y1-y0)*.2,
                               facecolor="#e5e7eb", edgecolor="#64748b",
                               alpha=.45, zorder=0))
        ax.text(-2.3, bottom+.12, f"Region {region+1}", fontsize=8)
    ax.scatter([5.], [0.], marker="*", s=75, color="gold",
               edgecolor="black", linewidth=.3, zorder=5)
    ax.set_xlim(-5.6, 5.5)
    ax.set_ylim(-2., 2.)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")


def render(seed=981001, risk="risky", out=None, max_frames=85):
    snapshot, branches = simulate(seed, risk)
    out = Path(out or f"lookahead_{risk}_seed{seed}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    maximum = max(len(branch["trace"]) for branch in branches.values())-1
    times = np.unique(np.linspace(0, maximum, min(max_frames, maximum+1),
                                  dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    value = branches["HOLD"]["cost"]-branches["SEND"]["cost"]

    def update(frame):
        t = int(times[frame])
        for ax, (label, branch) in zip(axes, branches.items()):
            ax.clear()
            draw_background(ax)
            trace = branch["trace"]
            last = min(t, len(trace)-1)
            for agent, color in (("scout", "#2563eb"),
                                 ("carrier", "#e11d48")):
                path = np.asarray([entry["positions"][agent]
                                   for entry in trace[:last+1]], dtype=float)
                ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4,
                        alpha=.8, zorder=3)
                ax.scatter(path[-1, 0], path[-1, 1], s=42,
                           color=color, edgecolor="white", linewidth=.6,
                           zorder=4)
            delivered = (branch["delivery_step"] is not None and
                         t >= branch["delivery_step"])
            scanned = (branch["scan_step"] is not None and
                       t >= branch["scan_step"])
            if delivered:
                ax.text(.02, .96, "48 B innovation delivered",
                        transform=ax.transAxes, va="top", fontsize=8,
                        color="#2563eb")
            if scanned:
                ax.text(.02, .86,
                        f"carrier scanned Region {branch['scan_region']+1}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        color="#be123c")
            ax.set_title(f"{label}  |  team cost {branch['cost']:.3f}")
        fig.suptitle(f"Same world and scout reading · t={t*snapshot.config.dt:.1f}s"
                     f" · J(HOLD)−J(SEND)={value:+.3f} · compact 120 charged B",
                     fontsize=10)
        return []

    animation = FuncAnimation(fig, update, frames=len(times), interval=110,
                              blit=False)
    animation.save(str(out), writer=PillowWriter(fps=9), dpi=95)
    update(len(times)-1)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    delivery = branches["SEND"]["delivery_step"] or snapshot.step_index
    scanned = max(branch["scan_step"] or delivery
                  for branch in branches.values())
    stages = ((snapshot.step_index, "scout reading"),
              (delivery, "message opportunity"),
              (scanned, "carrier's next scan"))
    film, panels = plt.subplots(2, 3, figsize=(12, 5.2), sharex=True,
                                sharey=True)
    for row, (label, branch) in enumerate(branches.items()):
        trace = branch["trace"]
        for col, (step, stage) in enumerate(stages):
            ax = panels[row, col]
            draw_background(ax)
            last = min(step, len(trace)-1)
            for agent, color in (("scout", "#2563eb"),
                                 ("carrier", "#e11d48")):
                path = np.asarray([entry["positions"][agent]
                                   for entry in trace[:last+1]], dtype=float)
                ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4,
                        zorder=3)
                ax.scatter(path[-1, 0], path[-1, 1], color=color,
                           edgecolor="white", s=32, zorder=4)
            if branch["delivery_step"] is not None and step >= branch[
                    "delivery_step"]:
                ax.text(.02, .95, "48 B delivered", transform=ax.transAxes,
                        va="top", fontsize=8, color="#2563eb")
            if branch["scan_step"] is not None and step >= branch["scan_step"]:
                ax.text(.02, .82,
                        f"scanned R{branch['scan_region']+1}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        color="#be123c")
            ax.set_title(f"{label}: {stage}", fontsize=9)
    film.suptitle(f"Same world, one message · net SEND value {value:+.3f}",
                  fontsize=11)
    film.tight_layout()
    film.savefig(out.with_name(out.stem+"_filmstrip.png"), dpi=170)
    film.savefig(out.with_name(out.stem+"_filmstrip.pdf"))
    plt.close(film)
    print(out)
    print("value", value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=981001)
    parser.add_argument("--risk", choices=("safe", "risky"), default="risky")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.seed, args.risk, args.out)


if __name__ == "__main__":
    main()
