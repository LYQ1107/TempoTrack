#!/usr/bin/env python3
"""Recover immutable D2 association summaries; optionally replay the frozen winner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path).absolute()
    if 'v9_3' not in path.resolve().parts:
        raise ValueError('recovery writes require a v9_3 output directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def recover_rows(rows, annotation, prediction_root, summary_root):
    from tempotrack_research.evaluation.teta_parser import parse_teta_summary
    categories = read(annotation)['categories']
    protocol = SimpleNamespace(
        benchmark_categories=tuple(categories),
        base_ids=frozenset(int(c['id']) for c in categories if c.get('frequency') != 'r'),
        novel_ids=frozenset(int(c['id']) for c in categories if c.get('frequency') == 'r'),
        content_hash=lambda: sha(annotation),
    )
    if len(rows) != 12 or len({r['config_index'] for r in rows}) != 12:
        raise ValueError('exactly 12 unique mapped candidate rows required')
    if len({i['video_id'] for i in read(annotation)['images']}) != 128:
        raise ValueError('selection annotation must cover exactly 128 videos')
    recovered = []
    for row in rows:
        name = f"config_{int(row['config_index']):04d}"
        prediction = Path(prediction_root) / (name + '.json')
        summary = Path(summary_root) / name / 'teta_summary_results.pth'
        # The batch evaluator's association-only input must be byte-identical
        # to the mapped candidate, not just share its filename.
        evaluated = Path(summary_root).parent / 'predictions' / name / 'data' / 'tao_track.json'
        prediction_hash = sha(prediction)
        if sha(evaluated) != prediction_hash:
            if read(evaluated) != read(prediction):
                raise ValueError(f'evaluated prediction differs from candidate {name}')
        parsed = parse_teta_summary(summary, category_protocol=protocol,
            evaluation_manifest={'protocol': 'association_only', 'prediction_hash': prediction_hash,
                                 'annotation_hash': sha(annotation)})
        if parsed['unmatched_class_names'] or not parsed['base']:
            raise ValueError(f'incomplete category mapping for {name}')
        if not all(math.isfinite(float(parsed['base'][key])) for key in ('AssocA', 'TETA')):
            raise ValueError(f'nonfinite selection metric for {name}')
        recovered.append({**row, 'prediction': str(prediction), 'prediction_hash': prediction_hash,
            'summary_hash': sha(summary), 'official_subset_assoc_only': parsed})
    selected = max(recovered, key=lambda r: (r['official_subset_assoc_only']['base']['AssocA'],
        r['official_subset_assoc_only']['base']['TETA'], r['base_pair_f1'], -r['config_index']))
    return recovered, selected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--materialize', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    source = read(args.source)
    official = source['d2_official_subset']
    root = Path(official['subset_annotation']).parent
    all_rows = {r['config_index']: r for r in source['rows']}
    for row in source['d2_pre_top12']:
        if row['config'] != all_rows[row['config_index']]['config'] or row['config']['stage'] != 'D2':
            raise ValueError('D2 source config mapping mismatch')
    for key in ('manifest', 'annotation'):
        if sha(source[key]) != source[key + '_hash']:
            raise ValueError(f'source {key} hash mismatch')
    rows, selected = recover_rows(source['d2_pre_top12'], official['subset_annotation'],
        root / 'predictions', root / 'evaluation/association_only/teta')
    if source.get('final_selected') and selected['config_index'] != source['final_selected']['config_index']:
        raise ValueError('recovered selection disagrees with completed D2 selection')
    output = Path(args.output)
    result = {'status': 'COMPLETED', 'artifact': 'v9_3_dual_recovered_selection',
        'source': args.source, 'source_hash': sha(args.source), 'annotation': official['subset_annotation'],
        'annotation_hash': sha(official['subset_annotation']), 'selection_protocol': 'TEST_BASE_ADAPTED',
        'selection': 'Base AssocA, Base TETA, Base pairwise F1, lower config_index',
        'novel_used_for_selection': False, 'rows': rows, 'final_selected': selected, 'best': selected}
    selection_path = output / 'masa_d2_recovered_selection.json'
    if args.materialize and selection_path.exists():
        if read(selection_path) != result:
            raise ValueError('existing V9.3 selection differs from recovered inputs')
    else:
        write_new(selection_path, result)
    print(json.dumps({'selected': selected['config_index'], 'base': selected['official_subset_assoc_only']['base'],
                      'novel': selected['official_subset_assoc_only']['novel']}), flush=True)
    if args.materialize:
        if (output / 'full_test/prediction.json').exists():
            raise FileExistsError('full Test prediction already exists; evaluate/reuse it')
        from tempotrack_research.orchestration.v9_parameter_search import _native_prediction_rows
        allowed = {'alpha_fast', 'alpha_slow', 'dual_logit_scale', 'fast_accept_threshold', 'assignment_mode'}
        cfg = {k: v for k, v in selected['config'].items() if k in allowed}
        # Exactly the same whitelist used by the D2 official shortlist replay.
        prediction = _native_prediction_rows(Path(source['manifest']), Path(source['annotation']),
            mode='dual', device=args.device, tracker_config=cfg)
        write_new(output / 'full_test/prediction.json', prediction)
        write_new(output / 'full_test/prediction.meta.json', {'status': 'COMPLETED',
            'config': selected['config'], 'effective_tracker_config': cfg,
            'config_hash': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            'manifest': source['manifest'], 'manifest_hash': sha(source['manifest']),
            'annotation': source['annotation'], 'annotation_hash': sha(source['annotation']),
            'checkpoint': None, 'checkpoint_note': 'frozen native observation cache; no learned Dual checkpoint',
            'prediction_hash': sha(output / 'full_test/prediction.json'), 'record_count': len(prediction)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
