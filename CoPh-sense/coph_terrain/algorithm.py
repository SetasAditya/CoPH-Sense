"""Causal acquisition, post-reading sharing, and persistent-memory interfaces.

These helpers use the frozen E2-Natural environment without changing its
generator, physics, prices, or sensor/packet protocol. World state is used by
the teacher only; deployed decisions receive observation-derived features.
"""

from dataclasses import dataclass

import numpy as np

from .environment import TerrainAction, cell_to_position
from .physical_audit import (AGENTS, _public_travel_waypoint,
                             run_continuation)
from .pilot_environment import waypoint


@dataclass(frozen=True)
class AcquisitionSnapshot:
    env: object
    evidence_id: str
    choice: object


def acquire_to_evidence(snapshot, choice):
    """Execute one physical measurement and stop at its realized reading.

    Returns None if the request fails or the episode terminates first. This
    function never reveals the hidden field to the policy; only the simulator
    writes the acquired evidence into the owner's observation.
    """
    env = snapshot.clone()
    owner = choice.agent
    target = cell_to_position(choice.viewpoint)
    seen = set(env.owned[owner])
    targets = {agent: waypoint(env, agent) for agent in AGENTS}
    started = False
    while not env.done:
        positions = env.positions
        observations = env.observations()
        velocity = np.asarray(observations[owner]["kinematics"][2:4], dtype=float)
        request = None
        if not started and env.active_sense[owner] is None:
            if (np.linalg.norm(positions[owner] - target) <=
                    .5 * env.config.probe_range and
                    np.linalg.norm(velocity) <= .12):
                request = choice.payload()
                started = True
                targets[owner] = tuple(float(v) for v in positions[owner])
            else:
                targets[owner] = _public_travel_waypoint(env, owner, target)
        elif started:
            targets[owner] = tuple(float(v) for v in positions[owner])
        for agent in AGENTS:
            if agent != owner and env.step_index % 20 == 0:
                targets[agent] = waypoint(env, agent)
        env.step({agent: TerrainAction(
            waypoint=targets[agent], sense=request if agent == owner else None)
            for agent in AGENTS})
        new = set(env.owned[owner]) - seen
        if new:
            raw = [item for item in new
                   if item not in getattr(env, "innovation_packets", {})]
            evidence_id = sorted(raw or new)[0]
            evidence = env.owned[owner][evidence_id]
            target_region = choice.target_region or choice.viewpoint
            if target_region not in evidence.cells:
                return None
            return AcquisitionSnapshot(env, evidence_id, choice)
        if started and env.active_sense[owner] is None:
            return None
    return None


def post_reading_branch(snapshot, evidence_id, send, continue_mission=True):
    """One post-observation SEND/HOLD choice, followed by scripted pH motion.

    The SEND branch returns toward the recipient if needed, attempts exactly
    one packet, waits for declared delivery, and only then continues. HOLD
    takes the normal mission route. Both branches include all sunk costs.
    With continue_mission=False, return (None, branch_state) at the next
    receiver decision so both branches can use the same downstream rule.
    """
    env = snapshot.clone()
    evidence = next((owned for owned in env.owned.values()
                     if evidence_id in owned), None)
    if evidence is None:
        raise ValueError("sender does not own evidence")
    sender = evidence[evidence_id].source
    receiver = "carrier" if sender == "scout" else "scout"
    if send and env.packet_attempts < env.config.max_team_packets:
        targets = {agent: waypoint(env, agent) for agent in AGENTS}
        attempted = False
        deadline = env.config.horizon_steps
        while not env.done and env.step_index < deadline:
            positions = env.positions
            in_range = (np.linalg.norm(positions[sender] - positions[receiver])
                        <= env.config.communication_range)
            if not attempted:
                targets[sender] = (waypoint(env, sender) if in_range else
                                   _public_travel_waypoint(env, sender,
                                                           positions[receiver]))
            elif env.step_index % 20 == 0:
                targets[sender] = waypoint(env, sender)
            if env.step_index % 20 == 0:
                targets[receiver] = waypoint(env, receiver)
            packet = evidence_id if in_range and not attempted else None
            env.step({agent: TerrainAction(
                waypoint=targets[agent],
                send_evidence_id=packet if agent == sender else None)
                for agent in AGENTS})
            if packet is not None:
                attempted = True
            if attempted and not any(item.evidence_id == evidence_id and
                                     item.kind == "evidence" for item in env.pending):
                break
    if not continue_mission:
        return None, env
    result = run_continuation(env, (), ()) if not env.done else {
        "mission_cost": env.ledger.mission_cost,
        "material_exposure": env.ledger.material_exposure,
        "score_single_world": env.ledger.mission_cost + env.ledger.material_exposure,
        "success": env.success,
    }
    return result, env


def a4_teacher(snapshot, evidence_id):
    """Exact paired simulator returns for this realized post-reading state."""
    held, _ = post_reading_branch(snapshot, evidence_id, False)
    sent, delivered_snapshot = post_reading_branch(snapshot, evidence_id, True)
    return {"hold": held, "send": sent,
            "delivered_snapshot": delivered_snapshot}


def a5_teacher(snapshot, recipient, choices, sets):
    """Post-delivery acquisition values under the specified continuation."""
    if recipient not in AGENTS:
        raise ValueError(recipient)
    return [run_continuation(snapshot, choices, selected) for selected in sets]


def delivery_memory_pair(pre_send, delivered, evidence_id):
    """Match positions and sunk costs while intervening on recipient memory."""
    source = next(name for name in AGENTS if evidence_id in pre_send.owned[name])
    recipient = "carrier" if source == "scout" else "scout"
    if evidence_id not in delivered.received[recipient]:
        raise ValueError("evidence was not delivered")
    retained = delivered.clone()
    withheld = delivered.clone()
    evidence = withheld.received[recipient].pop(evidence_id)
    if hasattr(withheld, "forget_delivered_evidence"):
        withheld.forget_delivered_evidence(recipient, evidence, pre_send)
        return retained, withheld, recipient
    old_grid = (pre_send.known_surface[recipient]
                if evidence.modality == "geometry" else
                pre_send.known_traction[recipient])
    new_grid = (withheld.known_surface[recipient]
                if evidence.modality == "geometry" else
                withheld.known_traction[recipient])
    for cell in evidence.cells:
        new_grid[cell] = old_grid[cell]
    return retained, withheld, recipient
