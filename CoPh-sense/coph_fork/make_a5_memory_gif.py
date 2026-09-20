#!/usr/bin/env python3
"""Animate the exact A5-v2 all-good memory intervention as a shareable GIF."""

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.memory_a5 import exact_evaluation  # noqa: E402
from coph_fork.memory_critic_a5 import MemorySetCritic, delivery_posteriors  # noqa: E402
from coph_fork.oracle import ExactOracleConfig  # noqa: E402
from coph_fork.run_memory_a5 import load_models  # noqa: E402


OUT = HERE / "results" / "a5_memory_v2"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 1280, 720
WHITE = "#FFFFFF"
BG = "#F6F8FC"
INK = "#17243A"
MUTED = "#5C6A7F"
LINE = "#D9E2EC"
PURPLE = "#7254A3"
PURPLE_LIGHT = "#F1EAF8"
TEAL = "#00877F"
TEAL_LIGHT = "#E5F5F2"
AMBER = "#A66B1B"
AMBER_LIGHT = "#FFF1DD"
GREEN = "#358369"
RED = "#AE4F55"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def text(draw, xy, message, size=22, color=INK, bold=False, anchor=None):
    draw.text(xy, message, font=font(size, bold), fill=color, anchor=anchor)


def box(draw, xy, fill, outline=None, radius=18, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill,
                           outline=outline, width=width)


def pill(draw, xy, label, fill, color, size=17):
    x, y = xy
    f = font(size, True)
    bbox = draw.textbbox((0, 0), label, font=f)
    tw = bbox[2] - bbox[0]
    box(draw, (x, y, x + tw + 26, y + 32), fill, radius=14)
    draw.text((x + 13, y + 5), label, font=f, fill=color)
    return tw + 26


def fork(draw, x1, x2, y, route, active):
    """Two visible choices at each region; highlight the executed one."""
    top = [(x1, y), (x1 + 22, y - 28), (x2 - 22, y - 28), (x2, y)]
    bottom = [(x1, y), (x1 + 22, y + 30), (x2 - 22, y + 30), (x2, y)]
    for name, points in (("top", top), ("bottom", bottom)):
        selected = active and name == route
        draw.line(points, fill=GREEN if selected and name == "top" else
                  AMBER if selected else LINE, width=8 if selected else 4, joint="curve")


def route_map(draw, left, stage, modern):
    y = 338
    draw.line((left + 48, y, left + 102, y), fill=LINE, width=4)
    fork(draw, left + 102, left + 215, y, "top", stage >= 4)
    draw.line((left + 215, y, left + 290, y), fill=LINE, width=4)
    fork(draw, left + 290, left + 403, y,
         "top" if modern else "bottom", stage >= 4)
    draw.line((left + 403, y, left + 455, y), fill=LINE, width=4)
    if stage >= 4:
        text(draw, (left + 160, y + 65), "shortcut", 17, GREEN, True, "mm")
        text(draw, (left + 348, y + 65), "shortcut" if modern else "detour",
             17, GREEN if modern else AMBER, True, "mm")
    else:
        text(draw, (left + 160, y + 65), "geometry + traction", 15, MUTED, False, "mm")
        text(draw, (left + 348, y + 65), "geometry + traction", 15, MUTED, False, "mm")


def card(draw, left, stage, modern, trace):
    width = 520
    accent = TEAL if modern else PURPLE
    light = TEAL_LIGHT if modern else PURPLE_LIGHT
    box(draw, (left, 190, left + width, 610), WHITE, LINE, radius=22)
    box(draw, (left + 18, 207, left + width - 18, 259), light, radius=12)
    text(draw, (left + 36, 218), "Memory-aware A5-v2" if modern else "Frozen one-fork A3",
         24, accent, True)
    if stage >= 2:
        pill(draw, (left + 35, 273), "R1: G known, T known", TEAL_LIGHT, TEAL)
    elif stage >= 1:
        pill(draw, (left + 35, 273), "Scout measured G1 + T1", AMBER_LIGHT, AMBER)
    else:
        pill(draw, (left + 35, 273), "R1: unknown", BG, MUTED)
    pill(draw, (left + 300, 273), "R2: unknown" if stage < 4 or not modern
         else "R2: resolved", BG if stage < 4 or not modern else TEAL_LIGHT,
         MUTED if stage < 4 or not modern else TEAL)
    route_map(draw, left, stage, modern)
    if stage >= 3:
        action = "Inspect G2 + T2" if modern else "Repeat G1 + T1"
        value = trace["exact_acquisition_value"]
        box(draw, (left + 28, 452, left + width - 28, 517), light, radius=14)
        text(draw, (left + 43, 462), action, 25, accent, True)
        text(draw, (left + width - 44, 481), "value {:+.2f}".format(value),
             19, TEAL if value > 0 else RED, True, "rm")
    else:
        text(draw, (left + 36, 472), "Carrier sensing decision pending", 20, MUTED)
    if stage >= 4:
        text(draw, (left + 35, 541), "Branch team cost", 18, MUTED)
        text(draw, (left + width - 35, 536), "{:.4f}".format(trace["total_team_cost"]),
             31, accent, True, "ra")
    else:
        text(draw, (left + 35, 542), "Scout + messages + carrier + routes", 17, MUTED)


def load_trace():
    seed = 1701
    a3, a4 = load_models(seed, 1801)
    model = MemorySetCritic("additive")
    model.load_state_dict(torch.load(OUT / "additive_1701.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    config = ExactOracleConfig()
    posteriors, _, _ = delivery_posteriors(a4, config)
    result = {}
    for condition, actor in (("persistent", None), ("learned_memory_v2", model)):
        rows = exact_evaluation(condition, a3, a4, config, actor, posteriors)["branches"]
        matching = [row for row in rows if row["world"] == [True] * 4
                    and set(row["delivered_ids"]) == {"r1-geometry", "r1-traction"}]
        if len(matching) != 1:
            raise RuntimeError("canonical all-good, both-delivered branch is ambiguous")
        result[condition] = matching[0]
    if result["persistent"]["carrier_action"] != [[1, "geometry"], [1, "traction"]] or \
       result["learned_memory_v2"]["carrier_action"] != [[2, "geometry"], [2, "traction"]]:
        raise RuntimeError("stored checkpoints no longer reproduce the A5 canonical trace")
    return result


CAPTIONS = (
    "Two route decisions ahead. Geometry and traction are initially hidden.",
    "The scout inspects geometry and traction in Region 1.",
    "A4 sends both readings; they reach the carrier and enter its ledger.",
    "With the same delivered evidence, the sensing choices diverge.",
    "The frozen A3 carrier repeats Region 1; A5-v2 resolves Region 2.",
    "The difference changes the second route and the charged team cost.",
)


def frame(stage, old, new, aggregate):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (54, 31), "Cooperative memory redirects sensing", 38, INK, True)
    text(draw, (55, 85), CAPTIONS[stage], 22, MUTED)
    phases = ("Hidden terrain", "Scout senses", "Evidence arrives", "Carrier chooses", "Routes execute")
    for i, phase in enumerate(phases):
        left = 52 + i * 239
        active = min(stage, 4) >= i
        box(draw, (left, 133, left + 224, 168), TEAL_LIGHT if active else WHITE,
            TEAL if active and i == min(stage, 4) else LINE, radius=15, width=2)
        text(draw, (left + 112, 150), phase, 16, TEAL if active else MUTED,
             active, "mm")
    card(draw, 52, stage, False, old)
    card(draw, 708, stage, True, new)
    if stage >= 5:
        diff = old["total_team_cost"] - new["total_team_cost"]
        box(draw, (560, 373, 719, 433), TEAL, radius=20)
        text(draw, (639, 401), "−{:.2f} cost".format(diff), 22, WHITE, True, "mm")
    text(draw, (54, 639),
         "Exact illustrative branch: all terrain good; both Region-1 packets delivered; seed 1701.",
         17, MUTED)
    if stage >= 5:
        text(draw, (54, 673),
             "Across all worlds/delivery outcomes (3-seed means): frozen {:.3f}  →  A5-v2 {:.3f} expected team cost.".format(
                 aggregate["persistent"]["expected_total_team_cost"]["mean"],
                 aggregate["additive"]["expected_total_team_cost"]["mean"]),
             17, INK, True)
    else:
        text(draw, (54, 673),
             "Costs include scout sensing, sent packets/ACKs, carrier sensing, and both executed routes.",
             16, MUTED)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traces = load_trace()
    aggregate = json.loads((OUT / "comparison.json").read_text())["aggregate"]
    frames = [frame(i, traces["persistent"], traces["learned_memory_v2"], aggregate)
              for i in range(6)]
    frames[-1].save(OUT / "a5_memory_story_preview.png")
    frames[0].save(OUT / "a5_memory_story.gif", save_all=True,
                   append_images=frames[1:], duration=[1100, 1100, 1400, 1700, 1700, 2700],
                   loop=0, optimize=True, disposal=2)
    print(OUT / "a5_memory_story.gif")


if __name__ == "__main__":
    main()
