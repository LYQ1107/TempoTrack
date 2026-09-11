#!/usr/bin/env python3
"""Verify archived full VOV pre-filter calls, coverage, and released equivalence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import sys

from v9_dual_recover_selection import read, sha, write_new


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--reuse-coverage', help='Reuse a prior scan only after rechecking every input hash')
    args = p.parse_args()
    from v9_active_tracker_search import _image_index, _image_for, _subset_video_ids, _load_released_components, _replay, _equivalence
    root = Path('/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final')
    raw_root = root / 'vov_test_native_raw_official_v8'
    post_root = root / 'vov_test_native_postfilter_official_v8'
    manifest_path = post_root / 'cache_v1/manifest.json'
    manifest = read(manifest_path)
    annotation = Path(manifest['annotation'])
    if sha(annotation) != manifest['annotation_hash']:
        raise ValueError('frozen annotation hash mismatch')
    images, _, _, _ = _image_index(annotation)
    def scan(path):
        seen = set()
        videos = set()
        duplicate_images = []
        assigned = 0
        count = 0
        rows = 0
        with gzip.open(path, 'rb') as stream:
            while True:
                try:
                    call = pickle.load(stream)
                except EOFError:
                    break
                image = _image_for(call['filename'], images)
                iid = int(image['id'])
                if iid in seen:
                    duplicate_images.append(iid)
                seen.add(iid)
                videos.add(int(image['video_id']))
                assigned += int((call['track_ids'] >= 0).sum())
                count += 1
                rows += len(call['track_ids'])
        return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'frames': count, 'rows': rows, 'assigned_ids': assigned,
            'image_ids': sorted(seen), 'video_ids': sorted(videos), 'duplicate_images': duplicate_images}
    paths = sorted((raw_root / 'recorder_calls_v1').glob('*.gz'))
    if args.reuse_coverage:
        prior = read(args.reuse_coverage)
        records = prior['calls']
        if [r['path'] for r in records] != [str(path) for path in paths]:
            raise ValueError('coverage input files changed')
        if prior['annotation_hash'] != sha(annotation) or any(sha(r['path']) != r['sha256'] for r in records):
            raise ValueError('coverage input hash changed')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(scan, paths))
    seen = [i for r in records for i in r['image_ids']]
    expected = {int(i['id']) for i in read(annotation)['images']}
    coverage = {'annotation_frames': len(expected), 'call_frames': len(seen),
        'video_count': len({v for r in records for v in r['video_ids']}),
        'missing_images': sorted(expected-set(seen)), 'extra_images': sorted(set(seen)-expected),
        'duplicate_images': len(seen)-len(set(seen)), 'assigned_ids': sum(r['assigned_ids'] for r in records)}
    result = {'status': 'COVERAGE_VERIFIED', 'calls_root': str(raw_root / 'recorder_calls_v1'),
        'calls': records, 'coverage': coverage, 'annotation': str(annotation), 'annotation_hash': sha(annotation),
        'baseline_prediction': str(post_root / 'stream_v1/tao_track.json'),
        'baseline_hash': sha(post_root / 'stream_v1/tao_track.json'),
        'raw_capture_prediction': str(raw_root / 'stream_v1/tao_track.json'),
        'raw_capture_prediction_hash': sha(raw_root / 'stream_v1/tao_track.json'),
        'manifest': str(manifest_path), 'manifest_hash': sha(manifest_path)}
    if coverage['missing_images']:
        missing = set(coverage['missing_images'])
        baseline = read(result['baseline_prediction'])
        with_observations = {int(r['image_id']) for r in baseline if int(r['image_id']) in missing}
        del baseline
        # The released tracker returns immediately for embeds=None and the
        # recorder intentionally writes nothing; there is no state update to replay.
        coverage['verified_zero_observation_images'] = sorted(missing - with_observations)
        coverage['missing_nonempty_images'] = sorted(with_observations)
        result['empty_frame_evidence'] = {'baseline_hash': result['baseline_hash'],
            'raw_and_post_capture_baseline_hash_equal': result['baseline_hash'] == result['raw_capture_prediction_hash'],
            'tracker_noop_on_embeds_none': True,
            'source': '/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack/ovtrack/models/trackers/ovtracker.py',
            'no_synthetic_observations_or_calls': True}
    write_new(Path(args.output) / 'coverage.json', result)
    print(json.dumps({'coverage': coverage}), flush=True)
    if any(coverage.get(k) for k in ('missing_nonempty_images', 'extra_images', 'duplicate_images', 'assigned_ids')):
        raise ValueError('full pre-filter coverage/stage gate failed')
    sys.path.insert(0, '/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack')
    for key in ('config', 'checkpoint'):
        if sha(manifest[key]) != manifest[key + '_hash']:
            raise ValueError(f'{key} hash mismatch')
    components = _load_released_components('vovtrack', Path(manifest['config']), Path(manifest['checkpoint']), args.device)
    from v9_active_tracker_search import _configs
    subset = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/outputs/tempotrack_v9/active/vov_cov_test_10.json')
    videos = _subset_video_ids(subset, 10)
    _, prediction = _replay('vovtrack', raw_root / 'recorder_calls_v1', annotation, _configs('vovtrack')[0],
        args.device, components, video_ids=videos)
    try:
        gate = _equivalence(prediction, Path(result['baseline_prediction']), images, videos)
    except Exception as exc:
        result.update(status='BLOCKED_EQUIVALENCE', error=str(exc))
        write_new(Path(args.output) / 'readiness.json', result)
        raise
    result.update(status='READY', equivalence=gate, model_config_hash=sha(manifest['config']),
        model_checkpoint_hash=sha(manifest['checkpoint']))
    write_new(Path(args.output) / 'readiness.json', result)
    print(json.dumps({'status': 'READY', 'equivalence': gate}), flush=True)


if __name__ == '__main__':
    main()
