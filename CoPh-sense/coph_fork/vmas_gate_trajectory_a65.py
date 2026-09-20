"""Public, provisional post-probe reconnect forecast for the A6.5 gate."""

import copy

import numpy as np

from coph_fork.environment import ForkAction
from coph_fork.run_moving_bridge_a65 import (
    carrier_motion, forecast_commit_slack, scout_motion, state,
)


def reconnect_features(decision_state):
    """Forecast two probe ticks, then the declared return controller.

    The clone executes no sensing, sends no packet, and receives no new
    evidence. Its route decision and controls therefore use only pre-probe
    public/local information. The output is checked across hidden worlds.
    """
    shadow = copy.deepcopy(decision_state)
    config = shadow.config
    commit_time = forecast_commit_slack(shadow)
    initial_step = shadow.step_index
    scout_start, _ = state(shadow, "scout")
    path_length = 0.0
    motion_charge = 0.0
    reached = False
    for step in range(config.horizon - initial_step):
        scout_pos, scout_vel = state(shadow, "scout")
        carrier_pos, carrier_vel = state(shadow, "carrier")
        if step >= 2 and np.linalg.norm(scout_pos - carrier_pos) <= config.communication_range - .02:
            reached = True
            break
        phase = "sense_geometry" if step == 0 else "sense_traction" if step == 1 else "return"
        scout_cmd = scout_motion(shadow, phase, scout_pos, scout_vel)
        carrier_cmd = carrier_motion(shadow, carrier_pos, carrier_vel)
        shadow.step({"scout": ForkAction(motion=scout_cmd),
                     "carrier": ForkAction(motion=carrier_cmd)})
        next_pos, _ = state(shadow, "scout")
        path_length += float(np.linalg.norm(next_pos - scout_pos))
        motion_charge += config.motion_cost * float(np.linalg.norm(scout_cmd)) * config.dt
        if shadow.done:
            break
    elapsed = (shadow.step_index - initial_step) * config.dt
    reconnect_time = max(0., elapsed - 2 * config.dt)
    network_time = config.communication_delay_steps * config.dt
    margin = commit_time - elapsed - network_time
    final_scout, _ = state(shadow, "scout")
    final_carrier, _ = state(shadow, "carrier")
    return [
        float(reconnect_time), float(path_length), float(motion_charge),
        float(config.time_cost * elapsed), float(margin),
        float(np.linalg.norm(final_scout - final_carrier)),
        float(np.linalg.norm(final_scout - scout_start)),
        float(reached),
    ]


def actionability_features(decision_state, reconnect=None):
    """Latest switch that can still select the top route under scripted motion.

    Candidate switch times are searched at VMAS tick resolution. The search
    simulates the public uninformed carrier controller, then forces its top
    branch and checks which corridor is physically entered at commitment.
    This is a feasibility forecast for this scripted bridge, not an oracle
    over the latent top terrain or a guarantee of downstream task value.
    """
    config = decision_state.config
    if reconnect is None:
        reconnect = reconnect_features(decision_state)
    formal = forecast_commit_slack(decision_state)
    max_ticks = max(0, round(formal / config.dt) - 1)

    def can_switch_after(ticks):
        shadow = copy.deepcopy(decision_state)
        # Isolate carrier steering feasibility from the scout's future path.
        # The full paired VMAS oracle later charges actual scout interactions.
        parked = shadow.physics.world.agents[0]
        parked.state.pos[0] = parked.state.pos[0].new_tensor((-.95, .85))
        parked.state.vel[0] = parked.state.vel[0].new_zeros(2)
        initial_collision = shadow.ledger.collision
        initial_failure = shadow.ledger.terminal_failure
        for step in range(config.horizon - shadow.step_index):
            if shadow.done or shadow.route_commitment is not None:
                break
            carrier_pos, carrier_vel = state(shadow, "carrier")
            shadow.step({
                "scout": ForkAction(motion=(0., 0.)),
                "carrier": ForkAction(motion=carrier_motion(
                    shadow, carrier_pos, carrier_vel, force_top=step >= ticks)),
            })
        return (shadow.route_commitment == "top"
                and shadow.ledger.collision <= initial_collision + 1e-9
                and shadow.ledger.terminal_failure <= initial_failure + 1e-9)

    if not can_switch_after(0):
        latest_tick = -1
    else:
        low, high = 0, max_ticks + 1
        while low + 1 < high:
            middle = (low + high) // 2
            if can_switch_after(middle):
                low = middle
            else:
                high = middle
        latest_tick = low
    latest = latest_tick * config.dt
    # Reconnect rollout includes the two physical probe ticks.
    delivery = 2 * config.dt + reconnect[0] + config.communication_delay_steps * config.dt
    return [float(latest), float(max(0., formal - latest)),
            float(latest - delivery), float(latest_tick >= 0)]
