#!/usr/bin/env python3
"""Read-only comparison of full and exact pre-filter COV recorder streams."""
import argparse
from pathlib import Path
import numpy as np

from v9_dual_recover_selection import read, sha, write_new


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from v9_active_tracker_search import _calls, _equivalence, _image_index, _image_for
    root = Path('/data2/usr_for_deadline/tempotrack_v9_relocated_20260910')
    annotation = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json')
    images, _, categories, _ = _image_index(annotation)
    streams = {}
    for name, calls in [('full', root / 'covtrack/test/native_recorder_v5/calls'),
                        ('pre_filter', root / 'active_v91_correct/covtrack/test/calls')]:
        records = {}
        for call in _calls(calls):
            if 'Slacklining_v_FlULukiEwA0_scene_0_0-1086/' not in call['filename']:
                continue
            image = _image_for(call['filename'], images)
            records[int(image['id'])] = call
        streams[name] = records
        print(name, 'target_video_frames', len(records), flush=True)
    full_rows = []
    differences = []
    for image_id, call in streams['full'].items():
        image = _image_for(call['filename'], images)
        for box, label, tid in zip(call['bboxes'], call['labels'], call['track_ids']):
            full_rows.append({'video_id': image['video_id'], 'image_id': image_id,
                'bbox': [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                'score': float(box[4]), 'category_id': categories[int(label)], 'track_id': int(tid)})
        other = streams['pre_filter'].get(image_id)
        if other is None:
            continue
        matches = []
        for i, (box, label) in enumerate(zip(call['bboxes'], call['labels'])):
            choices = [j for j, (b, l) in enumerate(zip(other['bboxes'], other['labels']))
                if int(l) == int(label) and np.array_equal(box, b)]
            if len(choices) == 1:
                j = choices[0]
                matches.append({'full_row': i, 'pre_filter_row': j,
                    'embedding_exact': bool(np.array_equal(call['embeds'][i], other['embeds'][j])),
                    'embedding_max_abs': float(np.abs(call['embeds'][i]-other['embeds'][j]).max()),
                    'class_embedding_exact': bool(np.array_equal(call['cls_embeds'][i], other['cls_embeds'][j]))})
        differences.append({'image_id': image_id, 'frame_id': call['frame_id'],
            'full_rows': len(call['bboxes']), 'pre_filter_rows': len(other['bboxes']),
            'full_source_indices_present': call.get('source_indices') is not None,
            'pre_filter_source_indices_present': other.get('source_indices') is not None,
            'matches': matches})
    baseline = root / 'covtrack/test/native_recorder_v5/stream/tao_track.json'
    try:
        exact = _equivalence(full_rows, baseline, images, {178})
    except Exception as exc:
        exact = {'status': 'FAIL', 'error': str(exc)}
    result = {'artifact': 'v9_3_cov_full_cache_diagnosis', 'diagnostic_only': True,
        'video_id': 178, 'annotation': str(annotation), 'annotation_hash': sha(annotation),
        'recorded_ids_to_baseline_equivalence': exact, 'frame_comparisons': differences,
        'recorded_first_frame': [r for r in full_rows if r['image_id'] == 6957]}
    write_new(Path(args.output), result)
    print('recorded_ids_to_baseline_equivalence', exact, flush=True)
    print('matched_boxes', sum(len(d['matches']) for d in differences),
        'changed_embeddings', sum(not m['embedding_exact'] for d in differences for m in d['matches']), flush=True)


if __name__ == '__main__':
    main()
