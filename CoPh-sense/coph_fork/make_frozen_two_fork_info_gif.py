#!/usr/bin/env python3
"""Animate archived, executed VMAS positions and evidence delivery.

The main trajectory is the frozen persistent-memory canonical replay. The
reset-memory choice is quoted from its separately executed matched replay.
No positions, sensing events, packet deliveries, or belief updates are
invented for the animation.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "a5_moving_bridge" / "physical_search"
OUTPUT = SOURCE / "frozen_two_fork_information.gif"
PREVIEW = SOURCE / "frozen_two_fork_information_preview.png"
CONTACT = SOURCE / "frozen_two_fork_information_keyframes.png"

W, H = 1440, 830
BG = "#f5f7fb"
WHITE = "#ffffff"
INK = "#132238"
MUTED = "#596a7d"
LINE = "#dce4ee"
SCOUT = "#158472"
CARRIER = "#366ac6"
PACKET = "#ad62ce"
SAFE = "#d8efe4"
RISK = "#f9e0ca"
RISK_INK = "#ab6831"
ISLAND = "#7c8995"
PRIOR = "#b8c5d4"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MAP = (65, 164, 923, 735)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(draw, point, value, size=17, color=INK, bold=False, anchor=None):
    draw.text(point, value, font=font(size, bold), fill=color, anchor=anchor)


def rounded(draw, rect, fill=WHITE, outline=LINE, radius=17, width=2):
    draw.rounded_rectangle(rect, radius=radius, fill=fill,
                           outline=outline, width=width)


def xy(point):
    left, top, right, bottom = MAP
    return (left + (float(point[0]) + 1.35) / 2.7 * (right - left),
            bottom - (float(point[1]) + .9) / 1.8 * (bottom - top))


def rect_world(draw, x0, y0, x1, y1, fill, outline=None, width=1):
    a, b = xy((x0, y1)), xy((x1, y0))
    draw.rectangle((a[0], a[1], b[0], b[1]), fill=fill,
                   outline=outline, width=width)


def line_world(draw, points, color, width=5):
    draw.line([xy(point) for point in points], fill=color,
              width=width, joint="curve")


def ring(draw, point, radius, fill, outline=WHITE, width=3):
    x, y = xy(point)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=fill, outline=outline, width=width)


def load():
    persistent = json.loads((SOURCE / "frozen_persistent_canonical_replay.json").read_text())
    reset = json.loads((SOURCE / "frozen_shared_reset_canonical_replay.json").read_text())
    layout = json.loads((SOURCE / "frozen_physical_layout_v1.json").read_text())
    assert persistent["success"] and reset["success"]
    assert persistent["seed"] == reset["seed"] == 1701
    assert persistent["world"] == reset["world"]
    assert len(persistent["states"]) == len(persistent["actions"]) + 1
    assert [e["step"] for e in persistent["events"]
            if e["type"] == "acquisition"] == [31, 32, 113, 114]
    deliveries = [e for e in persistent["events"]
                  if e["type"] == "packet_delivery" and
                  e.get("kind") == "evidence" and e.get("delivered")]
    assert [(e["observation_id"], e["step"]) for e in deliveries] == [
        ("obs-00000", 36), ("obs-00001", 37)]
    reset_acquisitions = [e for e in reset["events"]
                          if e["type"] == "acquisition" and
                          e["agent"] == "carrier"]
    assert [(e["region"], e["modality"]) for e in reset_acquisitions] == [
        (1, "geometry"), (1, "traction")]
    return persistent, reset, layout


def frame_steps(last):
    # Dense frames around the four acquisitions and two packet deliveries;
    # later physical motion is compressed without moving a robot artificially.
    steps = (list(range(0, 29, 3)) + list(range(29, 43)) +
             list(range(43, 109, 7)) + list(range(109, 123)) +
             list(range(123, 297, 15)) + list(range(297, 715, 31)) +
             list(range(715, last, 30)) + [last])
    return sorted(set(min(last, step) for step in steps))


def belief_state(step, agent, region, modality, prior):
    if agent == "scout" and region == 1:
        acquired = 31 if modality == "G" else 32
        if step >= acquired:
            return 1., "measured"
    if agent == "carrier" and region == 1:
        delivered = 36 if modality == "G" else 37
        if step >= delivered:
            return 1., "received"
    if agent == "carrier" and region == 2:
        acquired = 113 if modality == "G" else 114
        if step >= acquired:
            return 1., "measured"
    return prior, "prior"


def belief_card(draw, left, top, agent, step, prior1, prior2):
    accent = SCOUT if agent == "scout" else CARRIER
    rounded(draw, (left, top, left+430, top+177), WHITE, LINE)
    label(draw, (left+19, top+14), f"{agent.title()}'s local belief", 23,
          accent, True)
    label(draw, (left+20, top+48), "Evidence", 14, MUTED, True)
    label(draw, (left+180, top+48), "P(safe)", 14, MUTED, True)
    for index, (region, modality) in enumerate(((1, "G"), (1, "T"),
                                                 (2, "G"), (2, "T"))):
        y = top + 72 + index * 24
        prior = prior1 if region == 1 else prior2
        probability, status = belief_state(step, agent, region, modality, prior)
        label(draw, (left+20, y), f"R{region} {modality}", 16, INK,
              status != "prior")
        draw.rounded_rectangle((left+178, y+5, left+327, y+17),
                               radius=5, fill=LINE)
        draw.rounded_rectangle((left+178, y+5,
                                left+178+149*probability, y+17),
                               radius=5, fill=accent if status != "prior" else PRIOR)
        label(draw, (left+340, y-1), f"{probability:.2f}", 15,
              accent if status != "prior" else MUTED, status != "prior")


def phase(step):
    if step < 31:
        return "Both regions are uncertain. Scout approaches the Region-1 probe."
    if step < 33:
        return "Scout measures R1 geometry and traction; only the scout knows them."
    if step < 36:
        return "Two charged evidence packets travel from scout to carrier."
    if step < 38:
        return "Packets arrive: the carrier's R1 belief updates on delivery."
    if step < 113:
        return "Carrier uses received R1 evidence and heads for the R2 probe."
    if step < 115:
        return "Carrier measures R2; reset-memory would repeat R1 instead."
    if step < 715:
        return "Both robots execute their routes using the information they hold."
    return "The persistent-memory run completes; repeated R1 sensing was avoided."


def dashed(draw, a, b, fill, width=3, dash=11):
    import math
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    if not length:
        return
    n = int(length // dash)
    for i in range(0, n, 2):
        t0, t1 = i*dash/length, min((i+1)*dash/length, 1.)
        draw.line((a[0]+t0*dx, a[1]+t0*dy,
                   a[0]+t1*dx, a[1]+t1*dy), fill=fill, width=width)


def map_panel(draw, replay, step, layout):
    left, top, right, bottom = MAP
    rounded(draw, (left-11, top-10, right+11, bottom+10), WHITE, LINE,
            radius=22)
    rect_world(draw, -1.35, -.9, 1.35, .9, "#edf4f3")
    for region, x0, x1, backup in ((1, -.60, -.20, layout["backup_y1"]),
                                    (2, .24, .62, layout["backup_y2"])):
        rect_world(draw, x0, .18, x1, .72, SAFE)
        rect_world(draw, x0, backup-.06, x1, -.18, RISK)
        # Actual VMAS island geometry; the upper and lower passages remain open.
        center = -.42 if region == 1 else .42
        lower = backup + .27
        rect_world(draw, center-.14, lower, center+.14, .16, ISLAND)
        x, y = xy((center, .79))
        label(draw, (x, y), f"REGION {region}", 17, INK, True, "mm")
        x, y = xy((center, .45))
        label(draw, (x, y), "top: safe", 15, SCOUT, True, "mm")
        x, y = xy((center, backup-.13))
        label(draw, (x, y), "lower: low traction", 13,
              RISK_INK, True, "mm")
    # Decision boundaries, actual declared VMAS coordinates.
    for x, title in ((-.64, "fork 1"), (.20, "fork 2")):
        a, b = xy((x, -.76)), xy((x, .73))
        dashed(draw, a, b, "#b8c5d0", 2, 12)
        label(draw, (a[0]+5, b[1]+7), title, 13, MUTED)
    ring(draw, (-1.0, .20), 8, "#a6ddd0", SCOUT, 2)
    label(draw, xy((-.97, .28)), "R1 probe", 15, SCOUT, True)
    ring(draw, (-1.0, -.20), 8, "#b6d2fa", CARRIER, 2)
    label(draw, xy((-.97, -.30)), "remote R2 probe", 15, CARRIER, True)
    ring(draw, (1.17, 0.), 12, "#e98875", WHITE)
    label(draw, (xy((1.17, 0.))[0], xy((1.17, 0.))[1]+24),
          "GOAL", 16, INK, True, "mt")
    states = replay["states"]
    for agent, color, key in (("scout", SCOUT, "scout_position"),
                              ("carrier", CARRIER, "carrier_position")):
        points = [state[key] for state in states[:step+1:2]]
        if points and points[-1] != states[step][key]:
            points.append(states[step][key])
        if len(points) > 1:
            line_world(draw, points, color, 5 if agent == "carrier" else 4)
        ring(draw, states[step][key], 12 if agent == "carrier" else 10,
             color, WHITE, 3)
        px, py = xy(states[step][key])
        label(draw, (px+17, py-22), agent.title(), 16, color, True)
    # The two packet animations use the archived transmission/delivery steps.
    for sent, delivered, text_value, offset in ((33, 36, "G1", -15),
                                                  (34, 37, "T1", 15)):
        if sent <= step < delivered:
            scout = xy(states[step]["scout_position"])
            carrier = xy(states[step]["carrier_position"])
            a, b = (scout[0], scout[1]+offset), (carrier[0], carrier[1]+offset)
            dashed(draw, a, b, PACKET, 3, 8)
            progress = (step-sent+1)/(delivered-sent+1)
            cx, cy = a[0]+progress*(b[0]-a[0]), a[1]+progress*(b[1]-a[1])
            draw.rounded_rectangle((cx-19, cy-13, cx+19, cy+13),
                                   radius=6, fill=PACKET)
            label(draw, (cx, cy), text_value, 14, WHITE, True, "mm")


def render(replay, reset, layout, step):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    label(draw, (65, 30), "When shared terrain evidence changes the next measurement",
          34, INK, True)
    label(draw, (67, 81), phase(step), 20, MUTED)
    rounded(draw, (66, 112, 1378, 148), "#e7eef7", None, 11)
    label(draw, (85, 130), "Frozen two-fork bridge  •  canonical safe world  •  seed 1701",
          16, INK, True, "lm")
    label(draw, (1357, 130), f"t = {step*.05:4.1f} s  |  step {step}",
          16, INK, True, "rm")
    map_panel(draw, replay, step, layout["layout"])
    right = 958
    prior1 = float(layout["layout"]["safe_prior_1"])
    prior2 = float(layout["layout"]["safe_prior_2"])
    belief_card(draw, right, 163, "scout", step, prior1, prior2)
    belief_card(draw, right, 354, "carrier", step, prior1, prior2)
    rounded(draw, (right, 546, right+430, 681), WHITE, LINE)
    label(draw, (right+18, 562), "What the message changes", 20, INK, True)
    if step < 36:
        label(draw, (right+18, 598), "Carrier R1: prior only", 17, MUTED)
        label(draw, (right+18, 627), "G1 + T1 still private / in transit", 17, PACKET)
    else:
        label(draw, (right+18, 594), "R1 G + T received: no repeat probe", 16,
              SCOUT, True)
        label(draw, (right+18, 619), "Persistent: inspect R2 G + T", 16,
              CARRIER, True)
        label(draw, (right+18, 644), "Reset-memory: repeats R1 G + T", 16,
              RISK_INK)
    rounded(draw, (right, 692, right+430, 752), "#e8f4ee", None, 11)
    if step == len(replay["states"])-1:
        label(draw, (right+17, 698), "Completed team cost", 14, MUTED)
        persistent_cost = sum(replay["ledger"].values())
        reset_cost = sum(reset["ledger"].values())
        label(draw, (right+17, 734),
              f"Persistent {persistent_cost:.3f}   vs   reset {reset_cost:.3f}",
              17, SCOUT, True, "lm")
    else:
        label(draw, (right+17, 709), "Both replays charge sensing, packets,",
              15, MUTED)
        label(draw, (right+17, 733), "motion, risk, and elapsed time.",
              15, MUTED, False, "lm")
    label(draw, (66, 778),
          "G = geometry/clearance; T = traction. Beliefs change only on physical acquisition or delivered packet.",
          16, MUTED)
    label(draw, (66, 805),
          "Illustrative restricted scripted-VMAS bridge; actual archived replay, not a held-out E2 result.",
          14, MUTED)
    return image


def main():
    replay, reset, layout = load()
    steps = frame_steps(len(replay["states"])-1)
    frames = [render(replay, reset, layout, step) for step in steps]
    durations = [150] * len(frames)
    for index, step in enumerate(steps):
        if step in (32, 37, 114, steps[-1]):
            durations[index] = 950 if step != steps[-1] else 1900
    frames[-1].save(PREVIEW)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    key = [frames[min(range(len(steps)), key=lambda i: abs(steps[i]-step))]
           for step in (0, 32, 36, 113, 714, steps[-1])]
    thumb_width = 720
    sheet = Image.new("RGB", (thumb_width*2, 415*3), WHITE)
    for i, item in enumerate(key):
        sheet.paste(item.resize((thumb_width, 415)),
                    ((i%2)*thumb_width, (i//2)*415))
    sheet.save(CONTACT)
    print(f"{OUTPUT} ({len(frames)} frames)")
    print(PREVIEW)


if __name__ == "__main__":
    main()
