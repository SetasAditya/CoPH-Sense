"""Opt-in receiver channel ablations. No evaluator or scene truth enters here."""
import copy
from dataclasses import dataclass, asdict
from .maze_learning import MazeLearner


@dataclass(frozen=True)
class AttributionConfig:
    geometry: bool = True
    route_anchors: bool = True
    field_assimilation: bool = True
    replay: str = 'historical'

    def __post_init__(self):
        if self.replay not in ('none', 'current', 'historical'):
            raise ValueError('invalid attribution replay start')


ARMS = {
    'local_evidence': AttributionConfig(False, False, False, 'none'),
    'local_historical': AttributionConfig(False, False, False, 'historical'),
    'geometry_only': AttributionConfig(True, False, False, 'none'),
    'remote_route': AttributionConfig(True, True, False, 'none'),
    'reconstruction': AttributionConfig(True, True, True, 'none'),
    'current_replay': AttributionConfig(True, True, True, 'current'),
    'historical_replay': AttributionConfig(),
}


class AttributionLearner(MazeLearner):
    def __init__(self, *args, attribution=None, **kwargs):
        self.attribution = attribution or AttributionConfig()
        kwargs['mode'] = 'evidence' if self.attribution.replay == 'none' else 'full'
        super().__init__(*args, **kwargs)
        self.replay_start_mode = self.attribution.replay

    def assimilate(self, packet):
        # Validate the complete, unchanged wire payload using the established path.
        # Restore only disabled effects; retain sequence/provenance acknowledgments.
        saved = copy.deepcopy(dict(received=self.received,
            remote_terrain=self.remote_terrain, remote_cells=self.remote_cells,
            remote_geometry_reports=self.remote_geometry_reports,
            geometry_store=self.geometry_store, geometry_overflow=self.geometry_overflow,
            geometry_events=self.geometry_events, navigation=self.navigation,
            obstacles=[s.obstacles for s in self.stages]))
        event = super().assimilate(packet)
        field_eligible = self.received.get(packet.sender) is packet
        cfg = self.attribution
        if not cfg.field_assimilation:
            self.received = saved['received']
        if not cfg.route_anchors:
            self.remote_terrain = saved['remote_terrain']
            self.remote_cells = saved['remote_cells']
            self.refresh_terrain()
        if not cfg.geometry:
            for name in ('remote_geometry_reports', 'geometry_store', 'geometry_overflow', 'geometry_events'):
                setattr(self, name, saved[name])
            for stage, obstacles in zip(self.stages, saved['obstacles']):
                stage.obstacles = obstacles
            self.navigation['reported_geometry'] = saved['navigation'].get('reported_geometry', [])
        valid = event['reason'] == 'eligible'
        event['components'] = {
            'geometry': self._status(valid and bool(packet.geometry), cfg.geometry, valid),
            'route_anchors': self._status(valid, cfg.route_anchors, valid),
            'field_assimilation': self._status(valid and field_eligible, cfg.field_assimilation, valid),
        }
        return event

    @staticmethod
    def _status(eligible, enabled, valid):
        return dict(enabled=enabled, eligible=eligible, used=eligible and enabled,
                    reason=('packet_rejected' if not valid else 'disabled_by_ablation' if not enabled
                            else 'eligible' if eligible else 'unsupported_or_empty'))

    def update(self, x):
        result = super().update(x)
        result['attribution'] = asdict(self.attribution)
        result['replay_start_mode'] = self.replay_start_mode
        return result
