#!/usr/bin/env python3
"""Replay only explicit, uncompleted config indices after validated owner retirement."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

from v9_active_recover_selection import validate_row
from v9_dual_recover_selection import read, sha, write_new


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config-indices', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--external-root', required=True)
    args = p.parse_args()
    from v9_active_tracker_search import _configs, _load_released_components, _replay, _subset_video_ids
    checkpoint = read(args.checkpoint)
    if checkpoint['status'] != 'OWNER_RETIRED':
        raise ValueError('owner retirement is not confirmed')
    retired = checkpoint['owner']
    proc = Path('/proc') / str(retired['pid'])
    if proc.exists():
        stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
        if stat[0] != 'Z' and stat[19] == retired['start_ticks']:
            raise ValueError('old owner still exists; do not duplicate its remaining configs')
    source = read(args.source)
    configs = _configs('vovtrack')
    indices = [int(i) for i in args.config_indices.split(',')]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('explicit unique config indices required')
    completed = {r['config_index'] for r in checkpoint['rows']}
    if any(i in completed or i not in checkpoint['remaining'] for i in indices):
        raise ValueError('requested config is completed or not in remaining set')
    for row in checkpoint['rows']:
        validate_row(row, configs)
    if source['equivalence']['status'] != 'PASS':
        raise ValueError('source exact replay gate was not PASS')
    for path, expected in checkpoint['input_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'input changed after migration checkpoint: {path}')
    sys.path.insert(0, args.external_root)
    components = _load_released_components('vovtrack', Path(source['model_config']),
        Path(source['model_checkpoint']), 'cuda:0')
    annotation = Path(source['selection_annotation'])
    videos = _subset_video_ids(annotation, 128)
    if len(videos) != 128:
        raise ValueError('expected frozen 128-video selection')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        locks = Path(args.checkpoint).parent / 'config_locks'
        locks.mkdir(exist_ok=True)
        with (locks / f'{index}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            row_path = Path(args.checkpoint).parent / 'completed' / f'config_{index:04d}.json'
            if row_path.exists():
                raise FileExistsError(f'config {index} was already completed')
            print(json.dumps({'status': 'CONFIG_STARTED', 'config_index': index, 'pid': os.getpid()}), flush=True)
            metrics, _ = _replay('vovtrack', Path(source['calls_root']), annotation, configs[index],
                'cuda:0', components, video_ids=videos)
            row = {'config_index': index, 'config': configs[index], **metrics,
                'production_entrypoint': 'released external OVTracker.match',
                'input_calls': source['calls_root'], 'selection_metric': 'base_pair_f1'}
            validate_row(row, configs)
            write_new(row_path, row)
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_new(output / 'worker_complete.json', {'status': 'COMPLETED', 'rows': rows,
        'config_indices': indices, 'migration_checkpoint_hash': sha(args.checkpoint)})


if __name__ == '__main__':
    main()
