#!/usr/bin/env python3
"""One-shot locked Phase A winner materialization and official full Test evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from v9_dual_recover_selection import read, sha, write_new


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--lane', choices=['dual', 'covtrack'], required=True)
    p.add_argument('--wait-pid', type=int, help='Attach to an already running materializer; never restart it')
    p.add_argument('--cores', type=int, default=1)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path('/data2/usr_for_deadline/tempotrack_v9_relocated_20260910')
    output = root / 'v9_3' / ('dual' if args.lane == 'dual' else 'active/covtrack')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'closeout.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    full = output / 'full_test'
    if args.lane == 'dual':
        source = read(root / 'dual/masa_test_v92.json')
        annotation = Path(source['annotation'])
        expected_annotation_hash = source['annotation_hash']
    else:
        source = read(root / 'active/covtrack/test/v92_merged/active_sweep.json')
        subset = read(source['selection_annotation'])['tempotrack_v9_subset']
        annotation = Path(subset['source_annotation'])
        expected_annotation_hash = subset['source_sha256']
    if sha(annotation) != expected_annotation_hash:
        raise ValueError('full annotation hash differs from frozen selection source')
    py = '/home/lwr/anaconda3/envs/masaenv/bin/python'
    if args.lane == 'dual':
        command = [py, str(repo / 'tools/v9_dual_recover_selection.py'), '--source',
            str(root / 'dual/masa_test_v92.json'), '--output', str(output), '--materialize']
    else:
        command = ['/home/lwr/anaconda3/envs/ovtr/bin/python', str(repo / 'tools/v9_active_recover_selection.py'),
            '--frontend', 'covtrack', '--source', str(root / 'active/covtrack/test/v92_merged/active_sweep.json'),
            '--materialize-selected', str(output / 'selected.json'), '--output', str(full),
            '--annotation', str(annotation), '--calls-root', str(root / 'covtrack/test/native_recorder_v5/calls'),
            '--baseline-prediction', str(root / 'covtrack/test/native_recorder_v5/stream/tao_track.json'),
            '--external-root', '/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONPATH=str(repo) + ':/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI',
        LD_PRELOAD='/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0')
    if args.wait_pid:
        process_path = Path('/proc') / str(args.wait_pid)
        identity = (process_path / 'stat').read_text().split(') ', 1)[1].split()[19] if process_path.exists() else None
        while process_path.exists():
            try:
                stat = (process_path / 'stat').read_text().split(') ', 1)[1].split()
            except FileNotFoundError:
                break
            if stat[0] == 'Z' or stat[19] != identity:
                break
            time.sleep(15)
        if not (full / 'prediction.meta.json').exists():
            write_new(output / 'closeout_result.json', {'status': 'FAILED', 'stage': 'materialize',
                'materializer_pid': args.wait_pid, 'log': str(output / 'materialize.log')})
            return 2
    stages = [] if args.wait_pid else [('materialize', command)]
    name = 'v9_3_' + args.lane + '_full_test'
    evaluation_root = full / 'evaluation/association_only'
    # Direct official evaluation consumes one immutable prediction.  Do not
    # call evaluate_v6: it eagerly constructs both association and TCC copies.
    stages.append(('evaluate', [py, str(repo / 'tools/eval_ovmot_teta.py'),
        '--gt', str(annotation), '--pred', str(full / 'prediction.json'), '--out', str(evaluation_root),
        '--name', name, '--cores', str(args.cores)]))
    for stage, cmd in stages:
        print(json.dumps({'stage': stage, 'command': cmd, 'pid': os.getpid()}), flush=True)
        with (output / (stage + '.log')).open('x') as log:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            write_new(output / 'closeout_result.json', {'status': 'FAILED', 'stage': stage,
                'returncode': proc.returncode, 'log': str(output / (stage + '.log'))})
            return proc.returncode
    sys.path.insert(0, str(repo))
    from tempotrack_research.evaluation.teta_parser import parse_teta_summary
    categories = read(annotation)['categories']
    protocol = SimpleNamespace(benchmark_categories=tuple(categories),
        base_ids=frozenset(int(c['id']) for c in categories if c.get('frequency') != 'r'),
        novel_ids=frozenset(int(c['id']) for c in categories if c.get('frequency') == 'r'),
        content_hash=lambda: sha(annotation))
    summary = evaluation_root / name / 'teta_summary_results.pth'
    parsed = parse_teta_summary(summary, category_protocol=protocol)
    evaluation = full / 'evaluation/evaluation.json'
    doc = {'status': 'COMPLETED', 'artifact': 'v9_3_official_association_only_evaluation',
        'annotation': str(annotation), 'annotation_hash': sha(annotation),
        'source_prediction': str(full / 'prediction.json'), 'source_prediction_hash': sha(full / 'prediction.json'),
        'materialization': read(full / 'prediction.meta.json'),
        'provenance': {'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'evaluator_script': str(repo / 'tools/eval_ovmot_teta.py'),
            'evaluator_script_hash': sha(repo / 'tools/eval_ovmot_teta.py'),
            'closeout_script_hash': sha(__file__),
            'environment': {'python': sys.executable, 'python_version': sys.version,
                **{k: env.get(k) for k in ('PYTHONPATH', 'LD_PRELOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
            'cores': args.cores},
        'results': {'association_only': {'parsed': parsed, 'summary': str(summary), 'summary_hash': sha(summary)}},
        'tcc': 'NOT_REQUESTED_ASSOCIATION_ONLY_CLOSEOUT'}
    write_new(evaluation, doc)
    write_new(output / 'closeout_result.json', {'status': 'COMPLETED', 'evaluation': str(evaluation),
        'evaluation_hash': sha(evaluation), 'metrics': {k: v['parsed'] for k, v in doc['results'].items()}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
