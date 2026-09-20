"""Fixed motion controllers for the A5 moving bridge; no learned pH proposal."""

import numpy as np

from .run_moving_bridge_a65 import control, state


def certified_top(env, region):
    evidence = env.observations()["carrier"]["evidence"][region]["top"]
    return (evidence["geometry"] is True
            and evidence["traction"] is not None
            and evidence["traction"] >= env.config.traction_safe_threshold)


def carrier_motion(env):
    position, velocity = state(env, "carrier")
    x = float(position[0])
    if x < -.20:
        high = certified_top(env, 1)
        y = .43 if high else env.config.backup_y1
        target = (-.82, y) if x < -.58 and abs(float(position[1]) - y) > .10 \
            else (-.12, y)
    elif x < 0.:
        high = certified_top(env, 1)
        target = (.02, .43 if high else env.config.backup_y1)
    elif x < .19:
        high = certified_top(env, 2)
        y = .43 if high else env.config.backup_y2
        target = (.08, y) if abs(float(position[1]) - y) > .10 else (.24, y)
    elif x < .65:
        high = certified_top(env, 2)
        target = (.70, .43 if high else env.config.backup_y2)
    else:
        target = (1.17, -.13)
    return control(position, velocity, target, gain=3.2, damping=1.5, scale=.9)


def scout_motion(env, phase="navigate"):
    position, velocity = state(env, "scout")
    x = float(position[0])
    if phase == "probe_1":
        target = tuple(env.probe_position(1, "top"))
    elif x < -.77:
        target = (-.72, .72)
    elif x < -.20:
        target = (-.12, .72)
    elif x < .22:
        target = (.25, .72)
    elif x < .67:
        target = (.70, .72)
    else:
        target = (1.17, .17)
    return control(position, velocity, target, gain=3.4, damping=1.5)
