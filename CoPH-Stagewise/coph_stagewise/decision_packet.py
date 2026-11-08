"""Experimental fixed-opportunity packet ranking from advertised route decisions.

No evaluator or simulator imports. Costs are spatial route scores at the existing
nominal 1 m/s convention, not predictions of complete executed mission cost.
This module is opt-in and does not change any existing link or learner defaults.
"""
import copy
import heapq
import json
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .communication_repairs import canonical, encode_packet
from .cooperative import bootstrap_field, field_vector
from .maze_geometry import corridor_clear, network, obstacle_clear
from .route_field import values
from .terrain_warning import dense


@dataclass(frozen=True)
class DecisionAdvertisement:
    schema: int
    receiver: str
    recipient: str
    timestamp: float
    expires: float
    version: int
    goal_node: int
    risk_weight: float
    records: tuple

    def validate(self, sender, recipient, now):
        if self.schema != 1 or self.receiver != str(recipient) or self.recipient != str(sender):
            raise ValueError('decision_advertisement_identity')
        if self.timestamp > now + 1e-9 or self.expires < now or self.expires < self.timestamp:
            raise ValueError('decision_advertisement_age')
        if not 0 <= self.risk_weight or len(self.records) > 256:
            raise ValueError('decision_advertisement_bounds')
        numbers = [self.timestamp, self.expires, self.risk_weight]
        for r in self.records:
            n = len(r['points'])
            if n > 256 or np.asarray(r['points']).shape != (n, 2):
                raise ValueError('decision_advertisement_shape')
            if any(len(r[k]) != n for k in ('lengths', 'values', 'support')):
                raise ValueError('decision_advertisement_shape')
            numbers.extend(np.asarray(r['points']).ravel())
            numbers.extend(r['lengths']); numbers.extend(r['values'])
            if any(x < 0 for x in r['lengths']): raise ValueError('negative_length')
        if not np.isfinite(numbers).all(): raise ValueError('decision_advertisement_nonfinite')

    def wire(self):
        # Actual standalone compressed JSON size. No hypothetical cache savings.
        return zlib.compress(canonical(asdict(self)), 9)


def advertise(receiver, state, sender):
    context = receiver.navigation
    if context.get('route_field_mode', 'anchors') != 'anchors':
        raise ValueError('decision_proxy_v1_requires_anchor_routing')
    nodes, edges = network(context); nodes = np.asarray(nodes); q = np.asarray(state[:2])
    obstacles = context.get('reported_geometry', []); records = []
    def record(i, j, a, b, spacing):
        path = dense([a, b], spacing); mid = (path[:-1] + path[1:])/2
        risk, meta = values(mid, context, details=True)
        return dict(i=int(i), j=int(j), points=mid.tolist(),
                    lengths=np.linalg.norm(np.diff(path, axis=0), axis=1).tolist(),
                    values=risk.tolist(), support=meta['support'])
    for i, j in edges:
        if obstacle_clear(nodes[i], nodes[j], obstacles):
            records.append(record(i, j, nodes[i], nodes[j], .3))
    connections = {}
    for i, j in edges:
        a, b = nodes[i], nodes[j]; t = np.clip((q-a)@(b-a)/np.sum((b-a)**2), 0, 1)
        if np.linalg.norm(q-a-t*(b-a)) > .7: continue
        for k in (i, j):
            if corridor_clear(q, nodes[k], nodes=nodes, edges=edges) and obstacle_clear(q, nodes[k], obstacles):
                connections[k] = record(-1, k, q, nodes[k], .15)
    records.extend(connections.values())
    return DecisionAdvertisement(1, receiver.agent_id, str(sender), receiver.now,
        receiver.now + receiver.cfg.dt*receiver.cfg.update_every, receiver.version,
        int(np.argmin(np.linalg.norm(nodes-np.asarray(receiver.goal), axis=1))),
        float(context.get('risk_weight', 1.)), tuple(records))


def costs(ad, predictions=None):
    return [float(np.dot(r['lengths'], 1+4*ad.risk_weight*np.asarray(
        r['values'] if predictions is None else predictions[k]))) for k, r in enumerate(ad.records)]


def shortest(ad, edge_costs):
    """Production Dijkstra ordering and connector initialization, including ties."""
    graph = {}; distances = {}; parents = {}; queue = []
    for k, (r, cost) in enumerate(zip(ad.records, edge_costs)):
        i, j = r['i'], r['j']
        if i == -1:
            distances[j] = float(cost); parents[j] = (None, k); heapq.heappush(queue, (cost, j))
        else:
            graph.setdefault(i, []).append((j, cost, k)); graph.setdefault(j, []).append((i, cost, k))
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > distances[i]+1e-12: continue
        if i == ad.goal_node:
            ids = [i]; indices = []
            while True:
                previous, k = parents[ids[-1]]; indices.append(k)
                if previous is None: break
                ids.append(previous)
            return dict(nodes=ids[::-1], records=indices[::-1], cost=float(cost))
        for j, edge, k in graph.get(i, []):
            proposed = cost+edge
            if proposed < distances.get(j, float('inf'))-1e-12:
                distances[j] = proposed; parents[j] = (i, k); heapq.heappush(queue, (proposed, j))
    return None


def candidates(sender, state, recipient):
    """One ordinary locally reconstructed packet per retained stage, on copies."""
    result = []; work = 0
    for k, stage in enumerate(sender.stages):
        branch = copy.deepcopy(sender); branch._ledger_turn = len(branch.stages)-1-k
        before = branch.export_work; packet = branch.export(state, recipient)
        work += branch.export_work-before
        if packet is not None: result.append(packet)
    return result, work


def sender_reference(sender, ad):
    """Use local-only bootstrap means, never recursively fused operational fields.

    Leave advertised receiver estimates intact outside sender measured support.
    Thus this is a conditional, incomplete estimate rather than a truth oracle.
    """
    reference = [np.asarray(r['values']).copy() for r in ad.records]
    masks = [np.zeros(len(r['points']), bool) for r in ad.records]; work = 0
    for stage in reversed(sender.stages):
        fitted = bootstrap_field(stage, sender.cfg, sender.communication, sender.now,
                                 12345+int(sender.now*1000))
        if fitted is None: continue
        mean, ensemble, readings = fitted; work += len(readings)*(len(ensemble)+1)
        for k, record in enumerate(ad.records):
            points = np.asarray(record['points']); mask = np.array([
                stage.supported(p, sender.cfg) for p in points]) & ~masks[k]
            if mask.any():
                reference[k][mask] = np.clip(field_vector(mean, stage.centers,
                    points[mask], sender.cfg.rbf_width).reshape(-1, 3)[:, 0], 0, 20)
                masks[k] |= mask; work += int(mask.sum())
    return reference, masks, work


def packet_prediction(ad, packet):
    """Approximate existing conservative anchor envelope; no support expansion.

    This omits bounded remote-cell replacement and the subsequent local fit.
    Isolated real-receiver audits explicitly measure the resulting discrepancy.
    """
    anchors = np.asarray(packet.anchors); result = []
    for r in ad.records:
        points = np.asarray(r['points']); delta = points[:, None]-anchors[None, :, :2]
        supported = np.linalg.norm(delta, axis=-1) <= .65
        pred = anchors[None, :, 2]+np.sum(delta*anchors[None, :, 3:5], axis=-1)
        covered = supported.any(1); envelope = np.max(np.where(supported, pred, -np.inf), axis=1)
        old = np.asarray(r['values']); old_support = np.asarray(r['support'])
        merged = np.where(old_support, np.maximum(old, np.clip(envelope, 0, 20)), np.clip(envelope, 0, 20))
        result.append(np.where(covered, merged, old))
    return result


def rank(sender, state, recipient, ad, packets=None):
    ad.validate(sender.agent_id, recipient, sender.now)
    if packets is None: packets, export_work = candidates(sender, state, recipient)
    else: export_work = 0
    reference, masks, work = sender_reference(sender, ad)
    current = shortest(ad, costs(ad)); reference_costs = costs(ad, reference)
    if current is None: return None, dict(reason='no_route', candidates=[])
    rows = []
    for packet in packets:
        after = shortest(ad, costs(ad, packet_prediction(ad, packet)))
        benefit = None if after is None else float(sum(reference_costs[k] for k in current['records'])-
                                                    sum(reference_costs[k] for k in after['records']))
        rows.append(dict(packet=packet.identity, stage=packet.stage, benefit=benefit,
                         route=after, payload_bytes=len(encode_packet(packet))))
    positive = [r for r in rows if r['benefit'] is not None and r['benefit'] > 1e-10]
    # No invented cost-unit conversion for bytes: rank decision benefit, then size.
    winner = max(positive, key=lambda r:(r['benefit'], -r['payload_bytes'], r['stage'])) if positive else None
    packet = next((p for p in packets if winner and p.identity == winner['packet']), None)
    return packet, dict(reason='predicted_decision_benefit' if packet else 'rotation_fallback',
        incumbent=current, candidates=rows, sender_supported_queries=sum(int(m.sum()) for m in masks),
        queries=sum(len(m) for m in masks), export_work=export_work, reference_work=work,
        status_bytes=len(ad.wire()), raw_status_bytes=len(canonical(asdict(ad))))
