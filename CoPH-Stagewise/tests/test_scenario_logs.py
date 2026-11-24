import copy
import json
import numpy as np
import pytest
from coph_stagewise.scenario_logs import validate_times, load_encounter, encounter_frame, stagewise_frame


def test_timestamp_validation_rejects_duplicates_nan_and_reversal():
    for values in ([0, 0], [1, 0], [0, np.nan], []):
        with pytest.raises(ValueError):
            validate_times(values)


def test_native_encounter_post_step_subsampling_and_initial_state(tmp_path):
    root = tmp_path / 'native_final_layer_v1'
    root.mkdir()
    (root / 'manifest.json').write_text('{}')
    rows = []
    for layer in ('base', 'final'):
        rows.append(dict(layer=layer, family='circle_swap', seed=300, arm='coph_online',
                         path=np.ones((100, 4, 2)).tolist(), starts=np.zeros((4, 2)).tolist(),
                         goals=np.zeros((4, 2)).tolist(), success=True, new=0, time=8))
    (root / 'liveness_rows.json').write_text(json.dumps(rows))
    log = load_encounter(tmp_path)
    np.testing.assert_allclose(log['times'][:4], [0, .1, .5, .9])
    assert log['times'][-1] == pytest.approx(39.7)
    assert encounter_frame(log, 0)['arms']['final']['positions'].sum() == 0
    assert encounter_frame(log, 1)['arms']['final']['positions'].sum() == 8
    assert 'nominal/applied controls' in log['unavailable']


def test_stagewise_receipt_timeline_uses_logged_state_without_mutation():
    row = dict(time=1., end_time=1.12, x=[1, 2, 0, 0], after=[2, 2, 0, 0], u=[1, 0],
               nominal=[2, 0], control_model={'version': 3}, version=3, navigation={})
    ep = dict(trajectories=[[row]], observations=[[{'time': 1., 'readings': []}]],
              events=[{'time': .9, 'kind': 'delivered'}, {'time': 2., 'kind': 'sent'}],
              agents=[{'success': True}], spec={'initials': [[0, 2, 0, 0]]})
    before = copy.deepcopy(ep)
    assert stagewise_frame(ep, .5)['agents'][0]['active'] is False
    assert stagewise_frame(ep, 1.)['agents'][0]['state'] == row['x']
    assert stagewise_frame(ep, 2.)['agents'][0]['state'] == row['after']
    assert stagewise_frame(ep, 1.)['events'] == [ep['events'][0]]
    assert ep == before
