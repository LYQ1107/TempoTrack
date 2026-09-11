#!/usr/bin/env python3
"""Recover existing active selections and strictly validated completed log rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v9_dual_recover_selection import read, recover_rows, sha, write_new


def resolve_full_input_binding(calls_root, annotation, baseline, binding):
    """A new V9.3 audit may replace only the explicitly bound legacy input."""
    if binding.get('status') != 'READY':
        raise ValueError('full input binding is not READY')
    if str(Path(calls_root).resolve()) != binding['replaces_calls_root']:
        raise ValueError('full input binding targets a different requested calls root')
    if sha(annotation) != binding['annotation_hash'] or sha(baseline) != binding['baseline_hash']:
        raise ValueError('full input binding annotation/baseline hash mismatch')
    gate = read(binding['readiness'])
    if sha(binding['readiness']) != binding['readiness_hash'] or gate.get('status') != 'READY':
        raise ValueError('full input readiness changed or failed')
    if gate['calls_root'] != binding['calls_root'] or gate['annotation_hash'] != binding['annotation_hash'] or gate['baseline_hash'] != binding['baseline_hash']:
        raise ValueError('full input binding differs from audited inputs')
    paths = sorted(str(p) for p in Path(binding['calls_root']).glob('match_calls_*') if p.is_file())
    if paths != sorted(c['path'] for c in gate['calls']):
        raise ValueError('audited full input files changed')
    if any(sha(c['path']) != c['sha256'] for c in gate['calls']):
        raise ValueError('audited full input call hash mismatch')
    return Path(binding['calls_root'])


def validate_row(row, configs):
    index = row['config_index']
    if type(index) is not int or not 0 <= index < len(configs):
        raise ValueError('invalid config_index')
    expected = configs[index]
    config = row.get('config', {key: row[key] for key in expected if key in row})
    if config != expected:
        raise ValueError(f'config mapping mismatch at {index}')
    for field in ('base_pair_f1', 'base_pair_precision', 'base_pair_recall', 'observations'):
        if field not in row:
            raise ValueError(f'incomplete config result {index}: {field}')
    return {**{k: v for k, v in row.items() if k not in expected}, 'config': config}


def recover_log(paths, configs, expected):
    recovered = {}
    rejected = []
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                if 'config_index' in line:
                    rejected.append({'path': str(path), 'line': line_number, 'reason': 'incomplete JSON'})
                continue
            if not isinstance(raw, dict) or 'config_index' not in raw:
                continue
            row = validate_row(raw, configs)
            index = row['config_index']
            if index not in expected:
                raise ValueError(f'config {index} outside requested shard')
            if index in recovered and recovered[index] != row:
                raise ValueError(f'conflicting duplicate config {index}')
            recovered[index] = row
    return [recovered[i] for i in sorted(recovered)], rejected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--frontend', choices=['vovtrack', 'covtrack'], required=True)
    p.add_argument('--source', required=True, help='existing sweep, or completed sibling shard for log metadata')
    p.add_argument('--output', required=True)
    p.add_argument('--top12')
    p.add_argument('--summary-root')
    p.add_argument('--log', nargs='+')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=3)
    p.add_argument('--materialize-selected')
    p.add_argument('--calls-root')
    p.add_argument('--annotation')
    p.add_argument('--baseline-prediction')
    p.add_argument('--external-root')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--equivalence-evidence', help='Reuse PASS only with identical calls, inputs, model and replay code hashes')
    args = p.parse_args()
    from v9_active_tracker_search import _configs
    source = read(args.source)
    if source['frontend'] != args.frontend:
        raise ValueError('source frontend mismatch')
    configs = _configs(args.frontend)
    output = Path(args.output)
    if args.log:
        expected = {i for i in range(len(configs)) if i % args.shard_count == args.shard_index}
        rows, rejected = recover_log(args.log, configs, expected)
        remaining = sorted(expected - {r['config_index'] for r in rows})
        result = {**source, 'artifact': 'v9_3_active_recovered_shard',
            'status': 'PARTIAL' if remaining else 'SHARD_COMPLETE',
            'rows': rows, 'configs': len(rows), 'config_shard_index': args.shard_index,
            'config_shard_count': args.shard_count, 'remaining': remaining,
            'source_logs': [{'path': f, 'sha256': sha(f)} for f in args.log],
            'rejected_partial_lines': rejected,
            'launch_policy': 'Do not launch remaining while an existing worker owns them'}
        write_new(output / ('recovered_partial.json' if remaining else 'active_shard.json'), result)
        print(json.dumps({'completed': len(rows), 'expected': len(expected), 'remaining': remaining}), flush=True)
        return 0
    if args.materialize_selected:
        from v9_active_tracker_search import (_load_released_components, _replay, _equivalence,
            _image_index, _subset_video_ids)
        selected = read(args.materialize_selected)
        validate_row(selected, configs)
        sys.path.insert(0, args.external_root)
        annotation = Path(args.annotation)
        calls = Path(args.calls_root)
        binding_path = output.parent / 'full_test_input_binding.json'
        if binding_path.exists():
            calls = resolve_full_input_binding(calls, annotation, args.baseline_prediction, read(binding_path))
            print(json.dumps({'status': 'AUDITED_FULL_INPUT_BOUND', 'calls_root': str(calls),
                'binding': str(binding_path), 'binding_hash': sha(binding_path)}), flush=True)
        call_hashes = {str(f): sha(f) for f in sorted(calls.glob('match_calls_*')) if f.is_file()}
        if not call_hashes:
            raise ValueError('missing native recorder calls')
        components = _load_released_components(args.frontend, Path(source['model_config']),
            Path(source['model_checkpoint']), args.device)
        images, _, _, _ = _image_index(annotation)
        videos = _subset_video_ids(Path(source['equivalence_annotation']), 10)
        binding = {'calls': call_hashes, 'annotation_hash': sha(annotation),
            'baseline_hash': sha(args.baseline_prediction), 'videos': sorted(videos),
            'model_config_hash': sha(source['model_config']), 'checkpoint_hash': sha(source['model_checkpoint']),
            'replay_code_hash': sha(Path(__file__).with_name('v9_active_tracker_search.py')),
            'tracker_code_hash': sha(Path(args.external_root) / 'ovtrack/models/trackers/ovtracker.py')}
        prior = read(args.equivalence_evidence) if args.equivalence_evidence else None
        if prior and prior.get('status') == 'PASS' and prior.get('input_binding') == binding:
            gate = prior
        else:
            try:
                _, baseline_rows = _replay(args.frontend, calls, annotation, configs[0], args.device,
                    components, video_ids=videos)
                gate = {**_equivalence(baseline_rows, Path(args.baseline_prediction), images, videos),
                    'input_binding': binding}
            except Exception as exc:
                write_new(output / 'released_full_cache_equivalence.json', {'status': 'FAIL',
                    'input_binding': binding, 'error': str(exc)})
                raise
        write_new(output / 'released_full_cache_equivalence.json', gate)
        print(json.dumps({'full_cache_equivalence': gate['status']}), flush=True)
        metrics, rows = _replay(args.frontend, calls, annotation, selected['config'], args.device, components)
        baseline = read(args.baseline_prediction)
        baseline = baseline.get('data', baseline) if isinstance(baseline, dict) else baseline
        image_video = {int(i['id']): int(i['video_id']) for i in read(annotation)['images']}
        expected_videos = {image_video[int(r['image_id'])] for r in baseline}
        if {r['video_id'] for r in rows} != expected_videos:
            raise ValueError('full Test video coverage differs from native baseline')
        write_new(output / 'prediction.json', rows)
        write_new(output / 'prediction.meta.json', {'status': 'COMPLETED', 'config': selected['config'],
            'selected_hash': sha(args.materialize_selected), 'calls': call_hashes,
            'annotation': str(annotation), 'annotation_hash': sha(annotation),
            'checkpoint': source['model_checkpoint'], 'checkpoint_hash': sha(source['model_checkpoint']),
            'model_config_hash': sha(source['model_config']), 'prediction_hash': sha(output / 'prediction.json'),
            'record_count': len(rows), 'video_count': len(expected_videos), 'metrics': metrics, 'equivalence': gate})
        return 0
    top = read(args.top12)['rows']
    source_rows = {r['config_index']: validate_row(r, configs) for r in source['rows']}
    for row in top:
        validate_row(row, configs)
        if row['config'] != source_rows[row['config_index']]['config']:
            raise ValueError('Top12 mapping differs from source sweep')
    expected_top = sorted(source_rows.values(), key=lambda r: (-r['base_pair_f1'], -r['base_pair_precision'], r['config_index']))[:12]
    if [r['config_index'] for r in top] != [r['config_index'] for r in expected_top]:
        raise ValueError('Top12 differs from exact pairwise shortlist')
    official = source['official_subset']
    annotation = official['subset_annotation']
    if sha(annotation) != official['subset_annotation_sha256']:
        raise ValueError('subset annotation hash mismatch')
    prediction_root = Path(annotation).parent / 'official_subset_predictions'
    rows, selected = recover_rows(top, annotation, prediction_root, Path(args.summary_root))
    write_new(output / 'selected.json', selected)
    write_new(output / 'recovered_active_sweep.json', {**source, 'status': 'COMPLETED',
        'artifact': 'v9_3_active_recovered_selection', 'source_hash': sha(args.source),
        'top12_hash': sha(args.top12), 'novel_used_for_selection': False,
        'official_subset': {'status': 'COMPLETED', 'rows': rows, 'selected': selected},
        'best_baseline': selected})
    print(json.dumps({'selected': selected['config_index'], 'base': selected['official_subset_assoc_only']['base'],
        'novel': selected['official_subset_assoc_only']['novel']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
