#!/usr/bin/env python3
"""Continue recovered VOV shard0 through official Top12 and full Test (one core)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from v9_active_recover_selection import validate_row
from v9_dual_recover_selection import read, recover_rows, sha, write_new

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data2/usr_for_deadline/tempotrack_v9_relocated_20260910')
OVTR = '/home/lwr/anaconda3/envs/ovtr/bin/python'
MASA = '/home/lwr/anaconda3/envs/masaenv/bin/python'
EXTERNAL = '/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack'


def write_or_verify(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'existing immutable artifact differs: {path}')
    else:
        write_new(path, value)


def evaluate(prediction, annotation, output, name):
    output.mkdir(parents=True, exist_ok=True)
    cmd = [MASA, str(REPO / 'tools/eval_ovmot_teta.py'), '--gt', str(annotation), '--pred', str(prediction),
        '--out', str(output), '--name', name, '--cores', '1']
    env = dict(os.environ, PYTHONPATH=str(REPO) + ':/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI',
        LD_PRELOAD='/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    provenance = {'command': cmd, 'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'evaluator_script_hash': sha(REPO / 'tools/eval_ovmot_teta.py'),
        'environment': {k: env.get(k) for k in ('PYTHONPATH', 'LD_PRELOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'python': MASA, 'cores': 1}
    with (output / (name + '.log')).open('x') as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    from tempotrack_research.evaluation.teta_parser import parse_teta_summary
    cats = read(annotation)['categories']
    protocol = SimpleNamespace(benchmark_categories=tuple(cats),
        base_ids=frozenset(int(c['id']) for c in cats if c.get('frequency') != 'r'),
        novel_ids=frozenset(int(c['id']) for c in cats if c.get('frequency') == 'r'),
        content_hash=lambda: sha(annotation))
    summary = output / name / 'teta_summary_results.pth'
    parsed = parse_teta_summary(summary, category_protocol=protocol)
    return {'prediction': str(prediction), 'prediction_hash': sha(prediction),
        'annotation': str(annotation), 'annotation_hash': sha(annotation),
        'summary': str(summary), 'summary_hash': sha(summary), 'parsed': parsed, 'provenance': provenance}


def candidate(args):
    from v9_active_tracker_search import _load_released_components, _replay, _subset_video_ids
    source = read(args.source)
    row = read(args.row)
    sys.path.insert(0, EXTERNAL)
    components = _load_released_components('vovtrack', Path(source['model_config']),
        Path(source['model_checkpoint']), 'cuda:0')
    annotation = Path(source['selection_annotation'])
    _, prediction = _replay('vovtrack', Path(source['calls_root']), annotation, row['config'],
        'cuda:0', components, video_ids=_subset_video_ids(annotation, 128))
    write_new(args.prediction, prediction)


def supervise(args):
    from v9_active_tracker_search import _configs
    base = ROOT / 'v9_3/active/vovtrack'
    output = base / 'closeout'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume_inflight:
        from v9_active_migrate import exited
        children = read(args.resume_inflight)['preserved_children']
        print(json.dumps({'status': 'PRESERVING_INFLIGHT_CANDIDATES', 'pids': [c['pid'] for c in children]}), flush=True)
        while any(not exited(c['pid'], c['start_ticks']) for c in children):
            time.sleep(5)
    print(json.dumps({'status': 'WAITING_RECOVERED_SHARD0', 'pid': os.getpid()}), flush=True)
    while not (base / 'active_shard.json').exists():
        time.sleep(15)
    paths = [base / 'active_shard.json'] + [ROOT / f'active/vovtrack/test/v92_shard_{i}/active_shard.json' for i in (1, 2)]
    shards = [read(p) for p in paths]
    configs = _configs('vovtrack')
    all_rows = []
    for i, shard in enumerate(shards):
        if shard['status'] != 'SHARD_COMPLETE' or shard['config_shard_index'] != i:
            raise ValueError('invalid recovered shard status/index')
        for key in ('calls_root', 'selection_annotation_sha256', 'model_config', 'model_checkpoint', 'selection_protocol'):
            if shard[key] != shards[0][key]:
                raise ValueError(f'shard provenance mismatch: {key}')
        all_rows.extend(validate_row(r, configs) for r in shard['rows'])
    if sorted(r['config_index'] for r in all_rows) != list(range(len(configs))):
        raise ValueError('full config coverage duplicated or incomplete')
    top = sorted(all_rows, key=lambda r: (-r['base_pair_f1'], -r['base_pair_precision'], r['config_index']))[:12]
    source = {**shards[0], 'rows': all_rows, 'configs': len(all_rows)}
    source_path = output / 'merged_source.json'
    write_or_verify(source_path, source)
    write_or_verify(output / 'top12_pairwise.json', {'rows': top, 'novel_used_for_selection': False})
    annotation = Path(source['selection_annotation'])
    if sha(annotation) != source['selection_annotation_sha256']:
        raise ValueError('selection annotation changed')
    prediction_root = output / 'official_subset/predictions'
    summary_root = output / 'official_subset/evaluation/association_only/teta'
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    available = {int(x[0]): x[1].strip() for line in devices if len(x := line.split(',')) == 3 and int(x[2]) >= 12000}
    gpus = [available[int(i)] for i in args.gpus.split(',') if int(i) in available][:3]
    if not gpus:
        raise RuntimeError('no requested GPU with 12GB free for Top12')
    def run_partition(slot):
        for row in top[slot::len(gpus)]:
            name = f"config_{row['config_index']:04d}"
            row_path = output / 'rows' / (name + '.json')
            write_or_verify(row_path, row)
            prediction = prediction_root / (name + '.json')
            cmd = [OVTR, str(Path(__file__).resolve()), 'candidate', '--source', str(source_path),
                '--row', str(row_path), '--prediction', str(prediction)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[slot], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(REPO))
            if not prediction.exists():
                with (row_path.parent / (name + '.log')).open('x') as log:
                    subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                read(prediction)  # Completed preserved child must have valid JSON.
            evaluated_input = summary_root.parent / 'predictions' / name / 'data/tao_track.json'
            evaluated_input.parent.mkdir(parents=True, exist_ok=True)
            if evaluated_input.exists():
                if sha(evaluated_input) != sha(prediction):
                    raise ValueError('existing evaluation input differs')
            else:
                os.link(prediction, evaluated_input)
            evaluation_path = output / 'rows' / (name + '_evaluation.json')
            if evaluation_path.exists():
                result = read(evaluation_path)
                if sha(result['summary']) != result['summary_hash']:
                    raise ValueError('completed evaluation summary changed')
            else:
                result = evaluate(evaluated_input, annotation, summary_root, name)
                write_new(evaluation_path, result)
            print(json.dumps({'status': 'TOP12_CANDIDATE_EVALUATED', 'config_index': row['config_index']}), flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_partition, range(len(gpus))))
    rows, selected = recover_rows(top, annotation, prediction_root, summary_root)
    write_new(base / 'selected.json', selected)
    write_new(base / 'recovered_active_sweep.json', {**source, 'status': 'COMPLETED',
        'official_subset': {'rows': rows, 'selected': selected}, 'best_baseline': selected})
    print(json.dumps({'status': 'FINAL_SELECTED', 'config_index': selected['config_index']}), flush=True)
    manifest_path = Path('/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final/vov_test_native_postfilter_official_v8/cache_v1/manifest.json')
    manifest = read(manifest_path)
    full_annotation = Path(manifest['annotation'])
    if sha(full_annotation) != manifest['annotation_hash']:
        raise ValueError('full source annotation mismatch')
    full = base / 'full_test'
    native_root = manifest_path.parent.parent
    cmd = [OVTR, str(REPO / 'tools/v9_active_recover_selection.py'), '--frontend', 'vovtrack',
        '--source', str(source_path), '--materialize-selected', str(base / 'selected.json'),
        '--output', str(full), '--annotation', str(full_annotation), '--calls-root', str(native_root / 'recorder_calls_v1'),
        '--baseline-prediction', str(native_root / 'stream_v1/tao_track.json'), '--external-root', EXTERNAL]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(REPO))
    with (output / 'full_materialize.log').open('x') as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = evaluate(full / 'prediction.json', full_annotation, full / 'evaluation/association_only', 'v9_3_vovtrack_full_test')
    write_new(full / 'evaluation/evaluation.json', {'status': 'COMPLETED', 'results': {'association_only': record},
        'materialization': read(full / 'prediction.meta.json')})
    print(json.dumps({'status': 'FULL_TEST_COMPLETED'}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['supervise', 'candidate'])
    p.add_argument('--source')
    p.add_argument('--row')
    p.add_argument('--prediction')
    p.add_argument('--gpus', default='2,3,6')
    p.add_argument('--resume-inflight', help='Preserve recorded live children and reuse their completed predictions')
    args = p.parse_args()
    if args.mode == 'candidate':
        candidate(args)
    else:
        try:
            supervise(args)
        except Exception as exc:
            write_new(ROOT / 'v9_3/active/vovtrack/closeout/failure.json', {'status': 'FAILED',
                'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__':
    main()
