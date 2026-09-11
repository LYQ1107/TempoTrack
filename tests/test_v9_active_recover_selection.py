"""Recovery must never promote malformed or mismapped stdout rows."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('v9_active_recover_selection', TOOLS / 'v9_active_recover_selection.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def fixture_row():
    config = {'match_score_thr': .33, 'memo_frames': 30, 'momentum_embed': .4, 'label': 'released_reproduced'}
    row = {'config_index': 0, **config, 'base_pair_f1': .8, 'base_pair_precision': .9,
           'base_pair_recall': .72, 'observations': 200}
    return config, row


def test_truncated_line_is_not_completed(tmp_path):
    config, row = fixture_row()
    path = tmp_path / 'worker.log'
    path.write_text(json.dumps(row) + '\n{"config_index": 3, "base_pair_f1":')
    rows, rejected = recovery.recover_log([path], [config], {0})
    assert [r['config_index'] for r in rows] == [0]
    assert rows[0]['config'] == config
    assert rejected == [{'path': str(path), 'line': 2, 'reason': 'incomplete JSON'}]


def test_conflicting_duplicate_is_rejected(tmp_path):
    config, row = fixture_row()
    path = tmp_path / 'worker.log'
    path.write_text(json.dumps(row) + '\n' + json.dumps({**row, 'base_pair_f1': .1}))
    with pytest.raises(ValueError, match='conflicting duplicate config 0'):
        recovery.recover_log([path], [config], {0})


def test_wrong_config_is_rejected(tmp_path):
    config, row = fixture_row()
    path = tmp_path / 'worker.log'
    path.write_text(json.dumps({**row, 'memo_frames': 60}))
    with pytest.raises(ValueError, match='config mapping mismatch at 0'):
        recovery.recover_log([path], [config], {0})


def test_full_binding_rejects_unready_input(tmp_path):
    with pytest.raises(ValueError, match='not READY'):
        recovery.resolve_full_input_binding(tmp_path, tmp_path, tmp_path, {'status': 'PARTIAL'})


def test_full_binding_rejects_wrong_requested_root(tmp_path):
    with pytest.raises(ValueError, match='different requested calls root'):
        recovery.resolve_full_input_binding(tmp_path, tmp_path, tmp_path,
            {'status': 'READY', 'replaces_calls_root': str(tmp_path / 'other')})


def test_full_binding_verifies_calls_before_routing(tmp_path):
    legacy = tmp_path / 'legacy'
    calls = tmp_path / 'calls'
    calls.mkdir()
    call = calls / 'match_calls_1.pkl'
    call.write_bytes(b'cached-observations')
    annotation = tmp_path / 'annotation.json'
    annotation.write_text('{}')
    baseline = tmp_path / 'baseline.json'
    baseline.write_text('[]')
    ready = tmp_path / 'readiness.json'
    ready.write_text(json.dumps({'status': 'READY', 'equivalence': {'status': 'PASS'}, 'calls_root': str(calls),
        'annotation_hash': recovery.sha(annotation), 'baseline_hash': recovery.sha(baseline),
        'calls': [{'path': str(call), 'sha256': recovery.sha(call)}]}))
    binding = {'status': 'READY', 'replaces_calls_root': str(legacy), 'calls_root': str(calls),
        'annotation_hash': recovery.sha(annotation), 'baseline_hash': recovery.sha(baseline),
        'readiness': str(ready), 'readiness_hash': recovery.sha(ready)}
    assert recovery.resolve_full_input_binding(legacy, annotation, baseline, binding) == calls
    call.write_bytes(b'changed')
    with pytest.raises(ValueError, match='call hash mismatch'):
        recovery.resolve_full_input_binding(legacy, annotation, baseline, binding)


def test_full_replay_rejects_post_association_input():
    with pytest.raises(ValueError, match='post-association'):
        recovery.require_prefilter_call({'track_ids': [-1, 0, 1]})
    recovery.require_prefilter_call({'track_ids': [-1, -1]})
