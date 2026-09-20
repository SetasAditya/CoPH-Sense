"""Compact three-map GIF from a real, scripted E2 maze replay.

The left panel is evaluator truth. The center and right panels reconstruct
each agent's local terrain belief only from its passive observations, own
acquisitions, and delivered evidence. No hidden risk enters a belief panel.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coph_terrain.execution import belief_risk_grid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "results" / "visualizations" / "maze_pair_comm_seed600042"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 1500, 760
BG = "#f7f9fc"
INK = "#1c2c3d"
MUTED = "#5c6d7d"
WALL = "#292e36"
SCOUT = "#087d69"
CARRIER = "#2766c3"
SURFACE = "#16a890"
GRIP = "#d58924"
UNKNOWN = np.asarray((225, 231, 237), dtype=np.float32)
MAP_SIZE = 440
PANELS = ((31, 148), (530, 148), (1029, 148))


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def txt(draw, where, value, size=18, color=INK, bold=False, anchor=None):
    draw.text(where, value, font=font(size, bold), fill=color, anchor=anchor)


def shade(risk):
    """Yellow → orange → dark red risk, with no external image asset."""
    x = np.clip(np.asarray(risk, dtype=np.float32), 0, 1)[..., None]
    lo = np.asarray((255, 247, 208), dtype=np.float32)
    mid = np.asarray((245, 156, 79), dtype=np.float32)
    hi = np.asarray((158, 23, 57), dtype=np.float32)
    first = lo * (1 - np.minimum(x * 2, 1)) + mid * np.minimum(x * 2, 1)
    return np.where(x <= .5, first,
                    mid * (1 - np.minimum((x - .5) * 2, 1)) +
                    hi * np.minimum((x - .5) * 2, 1))


def as_map_image(risk, occupancy, visible=None, measured=None):
    rgb = shade(risk)
    if visible is not None:
        rgb[~visible] = UNKNOWN
        passive = visible & ~measured
        rgb[passive] = .55 * rgb[passive] + .45 * UNKNOWN
    rgb[occupancy] = np.asarray((41, 46, 54), dtype=np.float32)
    rows = np.uint8(rgb.transpose(1, 0, 2)[::-1])
    return Image.fromarray(rows, "RGB").resize((MAP_SIZE, MAP_SIZE),
                                               Image.Resampling.NEAREST)


def pos(point, panel):
    left, top = panel
    return (left + (float(point[0]) + 6) / 12 * MAP_SIZE,
            top + (6 - float(point[1])) / 12 * MAP_SIZE)


def grid_cell(cell, panel):
    left, top = panel
    return (left + (cell[0] + .5) / 60 * MAP_SIZE,
            top + (59.5 - cell[1]) / 60 * MAP_SIZE)


def circle(draw, point, radius, fill, outline="white", width=3):
    x, y = point
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=fill, outline=outline, width=width)


def star(draw, point, radius=12):
    import math
    x, y = point
    vertices = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * .46
        angle = -math.pi/2 + i*math.pi/5
        vertices.append((x+r*math.cos(angle), y+r*math.sin(angle)))
    draw.polygon(vertices, fill="#7752bd", outline="white")


def load(artifact):
    meta = json.loads((artifact / "metadata.json").read_text())
    states = np.load(artifact / "states.npz")
    truth = np.load(artifact / "truth_map.npz")
    events = [json.loads(line) for line in
              (artifact / "events.jsonl").read_text().splitlines()]
    assert meta["family"] == "labyrinth" and meta["success"]
    acquisitions = [e for e in events if e["type"] == "acquisition" and
                    e.get("valid") and e.get("agent") == "scout"]
    deliveries = [e for e in events if e["type"] == "packet_delivery" and
                  e.get("kind") == "evidence" and e.get("delivered") and
                  e.get("receiver") == "carrier"]
    by_id = {e["evidence_id"]: e for e in deliveries}
    pair = [(e, by_id[e["evidence_id"]]) for e in acquisitions
            if e["evidence_id"] in by_id]
    assert len(pair) == 2 and {a["modality"] for a, _ in pair} == {
        "geometry", "traction"}
    assert len(states["positions"]) == len(meta["actions"]) + 1
    transmissions = {e["evidence_id"]: e for e in events
                     if e["type"] == "transmission" and
                     e.get("sender") == "scout"}
    timeline = {a["modality"]: {
        "acquired": a["step"],
        "sent": transmissions[a["evidence_id"]]["step"],
        "delivered": d["step"]} for a, d in pair}
    return meta, states, truth, events, pair, timeline


def matched_silent_cost(artifact, metadata):
    sibling = artifact.parent / f"maze_pair_no_send_seed{metadata['parent_seed']}"
    path = sibling / "metadata.json"
    if not path.exists():
        return None
    silent = json.loads(path.read_text())
    assert silent["seed"] == metadata["seed"]
    assert silent["map_sha256"] == metadata["map_sha256"]
    assert silent["success"] and metadata["success"]
    total = lambda item: sum(item["ledger"].values())
    return total(metadata), total(silent)


def frame_steps(last, timeline):
    steps = list(range(0, last, max(1, last // 55))) + [last]
    for event in timeline.values():
        for key in ("acquired", "sent", "delivered"):
            value = event[key]
            steps.extend(range(max(0, value-3), min(last, value+5)+1))
    return sorted(set(min(last, step) for step in steps))


def beliefs_at(events, steps):
    agents = {name: {
        "surface": np.full((60, 60), np.nan, dtype=np.float32),
        "traction": np.full((60, 60), np.nan, dtype=np.float32),
        "appearance": np.full((60, 60), np.nan, dtype=np.float32),
    } for name in ("scout", "carrier")}
    evidence = {}
    cursor = 0
    result = []
    for step in steps:
        while cursor < len(events) and events[cursor]["step"] <= step:
            event = events[cursor]
            cursor += 1
            if event["type"] == "passive_observation":
                appearance = agents[event["agent"]]["appearance"]
                for cell, value in zip(event["cells"], event["appearance"]):
                    appearance[tuple(cell)] = value
            elif event["type"] == "acquisition" and event.get("valid"):
                evidence[event["evidence_id"]] = event
                agent = event["agent"]
                modality = "surface" if event["modality"] == "geometry" else "traction"
                for cell, value in zip(event["cells"], event["values"]):
                    agents[agent][modality][tuple(cell)] = value
            elif (event["type"] == "packet_delivery" and
                  event.get("kind") == "evidence" and event.get("delivered")):
                item = evidence[event["evidence_id"]]
                agent = event["receiver"]
                modality = "surface" if item["modality"] == "geometry" else "traction"
                for cell, value in zip(item["cells"], item["values"]):
                    agents[agent][modality][tuple(cell)] = value
        result.append({name: {key: value.copy() for key, value in arrays.items()}
                       for name, arrays in agents.items()})
    return result


def belief_image(arrays, occupancy):
    observation = {"appearance": arrays["appearance"],
                   "known_surface": arrays["surface"],
                   "known_traction": arrays["traction"]}
    risk = belief_risk_grid(observation)
    measured = np.isfinite(arrays["surface"]) | np.isfinite(arrays["traction"])
    visible = measured | np.isfinite(arrays["appearance"])
    return as_map_image(risk, occupancy, visible, measured)


def pulse_cells(draw, cells, panel, color, modality):
    if modality == "traction":
        x, y = grid_cell(cells[0], panel)
        draw.ellipse((x-13, y-13, x+13, y+13),
                     outline=color, width=4)
        return
    # Outline only sampled cells, never the entire enclosing circle.
    for cell in cells:
        x, y = grid_cell(cell, panel)
        draw.rectangle((x-3.1, y-3.1, x+3.1, y+3.1),
                       outline=color, width=1)


def packet_lane(draw, step, timeline):
    a, b, y = 770, 1180, 668
    draw.line((a, y, b, y), fill="#d3dce6", width=6)
    draw.polygon(((b, y), (b-18, y-10), (b-18, y+10)), fill="#d3dce6")
    circle(draw, (a, y), 10, SCOUT, None, 0)
    circle(draw, (b, y), 10, CARRIER, None, 0)
    for modality, color, icon in (("geometry", SURFACE, "surface"),
                                  ("traction", GRIP, "grip")):
        sent = timeline[modality]["sent"]
        arrived = timeline[modality]["delivered"]
        if sent <= step < arrived:
            fraction = (step-sent+1) / (arrived-sent+1)
            x = a + fraction * (b-a)
            draw.rounded_rectangle((x-46, y-17, x+46, y+17),
                                   radius=14, fill=color)
            txt(draw, (x, y), icon, 15, "white", True, "mm")


def phase(step, timeline):
    g, t = timeline["geometry"], timeline["traction"]
    if step < g["acquired"]:
        return "Scout approaches a sensing spot"
    if step < g["delivered"]:
        return "Surface scan is on its way"
    if step < t["acquired"]:
        return "Carrier has received the surface scan"
    if step < t["sent"]:
        return "Scout holds the grip reading while out of radio range"
    if step < t["delivered"]:
        return "Grip reading is on its way"
    if step < t["delivered"] + 15:
        return "Carrier has both terrain readings"
    return "Both robots continue through the maze"


def draw_frame(step, belief, states, truth, pair, timeline, comparison):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    txt(draw, (31, 25), "A scout shares what it learns in a maze", 32,
        INK, True)
    txt(draw, (1470, 32), f"{step * .05:.1f} s", 20, MUTED, True, "ra")
    txt(draw, (31, 76), phase(step, timeline), 21, MUTED)
    titles = ("REAL TERRAIN", "SCOUT'S MAP", "CARRIER'S MAP")
    occupancy = truth["occupancy"].astype(bool)
    rasters = (as_map_image(truth["risk"], occupancy),
               belief_image(belief["scout"], occupancy),
               belief_image(belief["carrier"], occupancy))
    for index, (panel, title, raster) in enumerate(zip(PANELS, titles, rasters)):
        left, top = panel
        draw.rounded_rectangle((left-9, top-9, left+MAP_SIZE+9,
                                top+MAP_SIZE+9), radius=14, fill="white",
                               outline="#d7e0e9", width=2)
        image.paste(raster, panel)
        txt(draw, (left, top-34), title, 20, INK, True)
        star(draw, pos((5., 0.), panel), 10)
        for agent, column, color in (("scout", 0, SCOUT),
                                     ("carrier", 1, CARRIER)):
            if index == 1 and agent == "carrier" or \
                    index == 2 and agent == "scout":
                continue
            path = [pos(point, panel) for point in
                    states["positions"][:step+1:3, column]]
            if path and path[-1] != pos(states["positions"][step, column], panel):
                path.append(pos(states["positions"][step, column], panel))
            if len(path) > 1:
                draw.line(path, fill=color, width=4, joint="curve")
            circle(draw, pos(states["positions"][step, column], panel),
                   10 if agent == "carrier" else 9, color)
    for acquired, delivered in pair:
        if acquired["step"] <= step <= acquired["step"] + 6:
            pulse_cells(draw, acquired["cells"], PANELS[1],
                        SURFACE if acquired["modality"] == "geometry" else GRIP,
                        acquired["modality"])
        if delivered["step"] <= step <= delivered["step"] + 8:
            pulse_cells(draw, acquired["cells"], PANELS[2],
                        SURFACE if acquired["modality"] == "geometry" else GRIP,
                        acquired["modality"])
    # Risk legend uses ordinary words, and unknown is visually distinct.
    txt(draw, (35, 607), "low risk", 16, MUTED)
    for j in range(140):
        color = tuple(int(v) for v in shade(j / 139))
        draw.line((106+j, 609, 106+j, 625), fill=color, width=1)
    txt(draw, (255, 607), "high risk", 16, MUTED)
    draw.rectangle((350, 609, 367, 625), fill=WALL)
    txt(draw, (376, 607), "wall", 16, MUTED)
    draw.rectangle((450, 609, 467, 625), fill=tuple(int(v) for v in UNKNOWN))
    txt(draw, (476, 607), "unknown", 16, MUTED)
    packet_lane(draw, step, timeline)
    txt(draw, (31, 658), "scout", 18, SCOUT, True)
    txt(draw, (1470, 658), "carrier", 18, CARRIER, True, "ra")
    # Only two modality names; no unexplained G/T/R1/R2 shorthand.
    txt(draw, (31, 716), "surface scan", 16, SURFACE, True)
    txt(draw, (190, 716), "grip probe", 16, GRIP, True)
    if comparison is not None and step == len(states["positions"])-1:
        shared, silent = comparison
        txt(draw, (485, 716),
            f"shared {shared:.2f}    silent {silent:.2f}",
            18, SCOUT, True)
    txt(draw, (1470, 716),
        "True risk at left; the carrier gets scout readings only after delivery.",
        15, MUTED, False, "ra")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path,
                        default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    artifact = args.artifact
    gif = artifact / "maze_information_flow.gif"
    preview = artifact / "maze_information_flow_preview.png"
    sheet_path = artifact / "maze_information_flow_keyframes.png"
    metadata, states, truth, events, pair, timeline = load(artifact)
    comparison = matched_silent_cost(artifact, metadata)
    steps = frame_steps(len(states["positions"])-1, timeline)
    beliefs = beliefs_at(events, steps)
    frames = [draw_frame(step, belief, states, truth, pair, timeline, comparison)
              for step, belief in zip(steps, beliefs)]
    durations = [135] * len(frames)
    highlights = {value for event in timeline.values()
                  for key, value in event.items()
                  if key in ("acquired", "delivered")}
    for i, step in enumerate(steps):
        if step in highlights:
            durations[i] = 900
    durations[-1] = 1700
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    frames[-1].save(preview)
    key = [frames[min(range(len(steps)), key=lambda i: abs(steps[i]-want))]
           for want in (0, timeline["geometry"]["acquired"],
                        timeline["geometry"]["delivered"],
                        timeline["traction"]["acquired"],
                        timeline["traction"]["delivered"], steps[-1])]
    sheet = Image.new("RGB", (750*2, 370*3), "white")
    for index, item in enumerate(key):
        sheet.paste(item.resize((750, 370)),
                    ((index % 2)*750, (index//2)*370))
    sheet.save(sheet_path)
    print(gif, len(frames), "frames")


if __name__ == "__main__":
    main()
