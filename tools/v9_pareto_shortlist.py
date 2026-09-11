#!/usr/bin/env python3
"""V9.3 bounded, Base-only shortlist from immutable corrected V9.2 sweeps.

No search is performed. prepare validates and streams existing sweep shards;
candidate evaluates one index; finish freezes the Base winner and runs Test.
All outputs are exclusive-create or hash-verified resumptions under v9_3.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path('/data2/usr_for_deadline/tempotrack_v9_relocated_20260910')
FIELDS = ('candidate_top_k', 'memory_capacity', 'query_observations', 'top_r',
          'min_dormant_gap', 'max_gap', 'reliability_multiplier', 'threshold', 'margin_threshold')
REASONS = ('A_precision95_recall', 'B_precision90_recall', 'C_precision85_f1',
           'D_best_f1', 'E_max_precision', 'F_precision80_recall',
           'G_lowest_false_merge', 'H_raw_control')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    with open(path) as f:
        return json.load(f)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) == value:
            return
        raise FileExistsError(f'refusing overwrite: {path}')
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify(path, expected):
    actual = sha(path)
    if not expected or actual != expected:
        raise ValueError(f'hash mismatch: {path}: {actual} != {expected}')
    return {'path': str(path), 'sha256': actual, 'status': 'COMPLETE'}


def identity(row):
    c = row['config']
    return (row['checkpoint_hash'] or '', str(c['checkpoint_step']), *(c[k] for k in FIELDS))


def validate_sweep(manifest):
    if manifest['schema_version'] != 10 or manifest['protocol'] != 'TEST_BASE_ADAPTED' or manifest['support_cache_semantics'] != 'raw support only; K/min_gap/max_gap applied per structural config':
        raise ValueError('uncorrected sweep semantics or non-Base selection protocol')


def validate_row(row, frontend):
    if (row['protocol'], row['split'], row['frontend'], row['selection_status']) != ('TEST_BASE_ADAPTED', 'test', frontend, 'VALID'):
        raise ValueError('invalid or non-Base sweep row')
    if any(not math.isfinite(float(v)) for v in row['selection_metrics'].values()):
        raise ValueError('nonfinite proxy metric')
    if any(not math.isfinite(float(row['config'][k])) for k in FIELDS):
        raise ValueError('nonfinite config value')


def ranks(row, raw_source=False):
    m = row['selection_metrics']
    p, r, f, n = m['precision'], m['recall'], m['f1'], m['false_merge']
    if m['accepted'] <= 0:
        return {}
    recall, f1 = (r, f, p, -n), (f, p, r, -n)
    result = {}
    raw = raw_source or float(row['config']['reliability_multiplier']) == 0
    if not raw_source:
        if p >= .95: result[REASONS[0]] = recall
        if p >= .90: result[REASONS[1]] = recall
        if p >= .85: result[REASONS[2]] = f1
        result[REASONS[3]] = f1
        result[REASONS[4]] = (p, r, f, -n)
        if p >= .80: result[REASONS[5]] = recall
        result[REASONS[6]] = (-n, p, r, f)
    if raw:
        # Prefer a valid primary control, otherwise best existing raw F1.
        result[REASONS[7]] = (p >= .95, r if p >= .95 else f, p, r, -n)
    return result


def offer(buckets, row, raw_source=False):
    for reason, rank in ranks(row, raw_source).items():
        bucket = buckets.setdefault(reason, [])
        if len(bucket) == 2 and rank < bucket[-1][0]:
            continue
        key = identity(row)
        if any(key == item[1] for item in bucket):
            continue
        bucket.append((rank, key, row))
        bucket.sort(key=lambda item: (tuple(-v for v in item[0]), item[1]))
        del bucket[2:]


def scan(task):
    path, frontend, raw_source = task
    manifest = read(path)
    validate_sweep(manifest)
    source = Path(manifest['all_rows_jsonl'])
    buckets, count, h = {}, 0, hashlib.sha256()
    checkpoints = {}
    with source.open('rb') as f:
        for line in f:
            h.update(line)
            if not line.strip(): continue
            row = json.loads(line)
            validate_row(row, frontend)
            if row.get('checkpoint'):
                checkpoints[row['checkpoint']] = row['checkpoint_hash']
            offer(buckets, row, raw_source)
            count += 1
    if h.hexdigest() != manifest['all_rows_hash'] or count != manifest['evaluated_rows']:
        raise ValueError(f'JSONL hash/count mismatch: {source}')
    for path_cp, digest in checkpoints.items(): verify(path_cp, digest)
    result = {'path': str(source), 'sha256': h.hexdigest(), 'row_count': count,
              'buckets': {k: [v[2] for v in b] for k, b in buckets.items()}, 'raw_source': raw_source}
    print(json.dumps({'scanned': str(source), 'rows': count}), flush=True)
    return result


def prepare(frontend, workers):
    out = ROOT / 'v9_3/pareto' / frontend
    if (out / 'candidates.json').exists():
        print(json.dumps({'already_prepared': str(out / 'candidates.json')})); return
    learned = ROOT / 'v9_2/sweeps' / frontend / ('test_base_adapted_learned' if frontend == 'vovtrack' else 'test_base_adapted') / 'merged.json'
    sources = [(learned, False)]
    if frontend == 'vovtrack':
        sources.append((learned.parent.parent / 'test_base_adapted_raw/merged.json', True))
    provenance, tasks = [], []
    event_hash = None
    for source, raw in sources:
        m = read(source)
        validate_sweep(m)
        if m['evaluated_structural_configs'] != m['structural_configs_total']:
            raise ValueError('incomplete structural coverage')
        if event_hash is not None and event_hash != m['event_cache_hash']:
            raise ValueError('raw/learned cache mismatch')
        event_hash = m['event_cache_hash']
        event = Path(m['event_cache']) / 'metadata.json'
        provenance.append(verify(event, event_hash))
        provenance.append(verify(m['search_space'], m['search_space_hash']))
        provenance.append({'path': str(source), 'sha256': sha(source)})
        indices = []
        for part in m['merged_parts']:
            provenance.append(verify(part['path'], part['sha256']))
            p = read(part['path'])
            indices.append(p['structural_shard_index'])
            for k in ('event_cache_hash', 'search_space_hash', 'support_cache_semantics', 'protocol', 'structural_shard_count'):
                if p[k] != m[k]: raise ValueError(f'shard provenance mismatch: {k}')
            tasks.append((part['path'], frontend, raw))
        if sorted(indices) != list(range(m['structural_shard_count'])):
            raise ValueError('missing or duplicate shards')
    metadata = read(event)
    if metadata['schema_version'] != 10 or metadata['frontend'] != frontend or metadata['split'] != 'test':
        raise ValueError('invalid event metadata')
    for key in ('manifest', 'frontend_prediction', 'annotation'):
        provenance.append(verify(metadata[key], metadata[key + '_hash']))
    for key, path in metadata['arrays'].items():
        provenance.append(verify(path, metadata['array_hashes'][key]))
    provenance.append(verify(metadata['rows_path'], metadata['rows_hash']))
    buckets, scans = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(scan, tasks):
            for rows in result.pop('buckets').values():
                for row in rows: offer(buckets, row, result['raw_source'])
            scans.append(result)
    candidates, by_key = [], {}
    for rank_index in (0, 1):
        for reason in REASONS:
            bucket = buckets.get(reason, [])
            if len(bucket) <= rank_index: continue
            row = bucket[rank_index][2]
            key = identity(row)
            if key in by_key:
                reasons = by_key[key]['reasons']
                if reason not in reasons: reasons.append(reason)
                continue
            if len(candidates) >= 12: continue
            entry = {**row, 'config_index': len(candidates), 'reasons': [reason]}
            candidates.append(entry); by_key[key] = entry
    if not candidates or not any(REASONS[7] in r['reasons'] for r in candidates):
        raise ValueError('missing candidates or raw control')
    annotation = read(metadata['annotation'])
    subset_path = REPO / 'outputs/tempotrack_v9/active/vov_cov_test_128.json'
    subset = read(subset_path)
    expected = sorted(annotation['videos'], key=lambda v: hashlib.sha256(str(int(v['id'])).encode()).hexdigest())[:128]
    ids = {int(v['id']) for v in expected}
    if len(ids) != 128 or subset['videos'] != expected:
        raise ValueError('selection is not the exact deterministic 128 Test subset')
    expected_images = [x for x in annotation['images'] if int(x['video_id']) in ids]
    image_ids = {int(x['id']) for x in expected_images}
    if subset['images'] != expected_images or subset['categories'] != annotation['categories'] or subset['annotations'] != [x for x in annotation['annotations'] if int(x['image_id']) in image_ids]:
        raise ValueError('subset annotation content mismatch')
    provenance.append({'path': str(subset_path), 'sha256': sha(subset_path)})
    native = read(metadata['manifest'])
    native['shards'] = [s for s in native['shards'] if int(s['video_id']) in ids]
    if {int(s['video_id']) for s in native['shards']} != ids:
        raise ValueError('native cache does not cover exact subset')
    native['v9_3_subset_source'] = {'path': metadata['manifest'], 'sha256': metadata['manifest_hash'], 'video_ids': sorted(ids)}
    write(out / 'subset_manifest.json', native)
    import ijson
    with open(metadata['frontend_prediction'], 'rb') as f:
        rows = [r for r in ijson.items(f, 'item', use_float=True) if int(r['video_id']) in ids]
    write(out / 'baseline_prediction.json', rows)
    document = {'artifact': 'v9_3_pareto_candidates', 'protocol': 'TEST_BASE_ADAPTED', 'frontend': frontend,
                'selection_base_only': True, 'novel_used_for_selection': False,
                'no_new_grid': True, 'max_candidates': 12, 'candidates': candidates,
                'category_absent': [r for r in REASONS if r not in buckets],
                'raw_control_semantics': 'existing raw sweep or lambda_zero existing learned row; effective checkpoint disabled',
                'source_provenance': provenance, 'scanned_shards': scans,
                'inputs': {k: metadata[k] for k in ('manifest', 'frontend_prediction', 'annotation')},
                'input_hashes': {k: metadata[k + '_hash'] for k in ('manifest', 'frontend_prediction', 'annotation')},
                'subset_annotation': str(subset_path), 'subset_annotation_hash': sha(subset_path),
                'subset_video_ids': sorted(ids), 'subset_manifest_hash': sha(out / 'subset_manifest.json'),
                'baseline_prediction_hash': sha(out / 'baseline_prediction.json'),
                'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}
    write(out / 'candidates.json', document)
    print(json.dumps({'prepared': str(out / 'candidates.json'), 'count': len(candidates)}), flush=True)


def memory_gate(required_gib=20):
    available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    if available < required_gib * 1024**3:
        raise RuntimeError(f'memory admission denied: need {required_gib} GiB available to preserve 12 GiB reserve')


def evaluate(annotation, prediction, out, name, cores):
    from tempotrack_research.evaluation.teta_parser import parse_teta_summary, inspect_installed_teta
    out = Path(out)
    complete = out / 'evaluation.json'
    receipt = {'annotation': str(annotation), 'annotation_hash': sha(annotation), 'prediction': str(prediction), 'prediction_hash': sha(prediction), 'name': name, 'evaluator_script_hash': sha(REPO / 'tools/eval_ovmot_teta.py')}
    if complete.exists():
        result = read(complete)
        verify(prediction, result['prediction_hash']); verify(annotation, result['annotation_hash'])
        verify(result['summary'], result['summary_hash'])
        return result
    summary = out / name / 'teta_summary_results.pth'
    recovered = summary.exists()
    if recovered:
        if (out / 'input_receipt.json').exists():
            if read(out / 'input_receipt.json') != receipt: raise ValueError('evaluation input receipt mismatch')
        else:
            # Initial launches predate receipts; bind their completed summaries
            # to verified source predictions and exact paths in the official log.
            log = (out / 'evaluator.log').read_text()
            if f'GT:   {annotation}' not in log or f'Pred: {prediction}' not in log:
                raise ValueError('cannot recover summary without exact input receipt')
            if name != 'baseline':
                inv = read(Path(prediction).parent / 'invariance.json')
                verify(prediction, inv['prediction_hash'])
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'partial evaluation preserved; use fresh directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'input_receipt.json', receipt)
    if not recovered:
        memory_gate()
        command = [sys.executable, str(REPO / 'tools/eval_ovmot_teta.py'), '--gt', str(annotation), '--pred', str(prediction), '--out', str(out), '--name', name, '--cores', str(cores)]
        with (out / 'evaluator.log').open('x') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    cats = read(annotation)['categories']
    protocol = SimpleNamespace(benchmark_categories=tuple(cats), base_ids=frozenset(int(c['id']) for c in cats if c.get('frequency') != 'r'), novel_ids=frozenset(int(c['id']) for c in cats if c.get('frequency') == 'r'), content_hash=lambda: sha(annotation))
    parsed = parse_teta_summary(summary, category_protocol=protocol)
    if not parsed.get('base') or parsed['base'].get('AssocA') is None: raise ValueError('missing Base AssocA')
    result = {'status': 'COMPLETED', 'protocol': 'association_only', 'annotation': str(annotation), 'annotation_hash': sha(annotation), 'prediction': str(prediction), 'prediction_hash': sha(prediction), 'summary': str(summary), 'summary_hash': sha(summary), 'parsed': parsed, 'evaluator': inspect_installed_teta(), 'evaluator_script_hash': sha(REPO / 'tools/eval_ovmot_teta.py')}
    write(complete, result)
    return result


def materialize_checked(doc, row, out, full, device):
    from tempotrack_research.orchestration.v9_parameter_search import materialize
    base = ROOT / 'v9_3/pareto' / doc['frontend']
    manifest = Path(doc['inputs']['manifest']) if full else base / 'subset_manifest.json'
    prediction = Path(doc['inputs']['frontend_prediction']) if full else base / 'baseline_prediction.json'
    annotation = Path(doc['inputs']['annotation']) if full else Path(doc['subset_annotation'])
    verify(manifest, doc['input_hashes']['manifest'] if full else doc['subset_manifest_hash'])
    verify(prediction, doc['input_hashes']['frontend_prediction'] if full else doc['baseline_prediction_hash'])
    verify(annotation, doc['input_hashes']['annotation'] if full else doc['subset_annotation_hash'])
    selected = out / 'selected_config.json'
    write(selected, row)
    if row.get('checkpoint'):
        import torch
        cp = Path(row['checkpoint'])
        verify(cp, row['checkpoint_hash'])
        training = cp.parent / 'train_result.json'
        episodes = cp.parent.parent.parent / 'episodes/build_data.json'
        train, build = read(training), read(episodes)
        if not train.get('base_only_supervision') or not build.get('base_only_supervision') or train['status'] != 'COMPLETED' or build['status'] != 'COMPLETED':
            raise ValueError('checkpoint lacks completed Base-only training provenance')
        state = torch.load(cp, map_location='cpu')
        if str(state['step']) != str(row['config']['checkpoint_step']) or any(state[k] != train[k] for k in ('input_hash', 'config_hash', 'algorithm_revision')):
            raise ValueError('checkpoint step/training provenance mismatch')
        checks = [verify(build['output'], build['episodes_hash']), verify(build['annotation'], build['annotation_hash'])]
        write(out / 'training_provenance.json', {'base_only_supervision': True, 'checkpoint': verify(cp, row['checkpoint_hash']), 'training': {'path': str(training), 'sha256': sha(training)}, 'episodes_manifest': {'path': str(episodes), 'sha256': sha(episodes)}, 'verified_inputs': checks})
        del state
    if (out / 'invariance.json').exists():
        inv = read(out / 'invariance.json')
        verify(out / 'prediction.json', inv['prediction_hash'])
        return out / 'prediction.json'
    if (out / 'prediction.json').exists(): raise FileExistsError('partial prediction preserved')
    memory_gate(44 if full else 20)
    # Local progress instrumentation only; native loading and engine semantics
    # remain the repository implementation. Restore even on a failed video.
    from tempotrack_research import v6_cli
    load_frames = v6_cli._frames_for_shard
    progress = {'videos_started': 0}
    def logged_frames(shard):
        progress['videos_started'] += 1
        print(json.dumps({'materialize_video': shard['video_id'], **progress,
                          'frontend': doc['frontend'], 'config_index': row['config_index'],
                          'full_test': full}), flush=True)
        return load_frames(shard)
    v6_cli._frames_for_shard = logged_frames
    try:
        materialize(frontend=doc['frontend'], manifest=manifest, frontend_prediction=prediction,
                    annotation=annotation, checkpoint=row.get('checkpoint'), selected_config=selected,
                    output=out, device=device)
    finally:
        v6_cli._frames_for_shard = load_frames
    # Independent exact UID/content/count gate, beyond production's selected-UID check.
    import ijson
    with prediction.open('rb') as f:
        original = {r['observation_uid']: r for r in ijson.items(f, 'item', use_float=True)}
    count, changed = 0, 0
    with (out / 'prediction.json').open('rb') as f:
        for r in ijson.items(f, 'item', use_float=True):
            old = original.pop(r['observation_uid'])
            changed += old.pop('track_id') != r.pop('track_id')
            if old != r: raise ValueError('non-track field changed')
            count += 1
    if original: raise ValueError('observations lost')
    write(out / 'invariance.json', {'only_track_id_changed': True, 'record_count': count, 'changed_observations': changed, 'prediction_hash': sha(out / 'prediction.json'), 'source_hash': sha(prediction), 'config_hash': sha(selected)})
    return out / 'prediction.json'


def candidate(frontend, index, device, cores):
    base = ROOT / 'v9_3/pareto' / frontend
    doc = read(base / 'candidates.json')
    if index == -1:
        prediction, name = base / 'baseline_prediction.json', 'baseline'
        verify(prediction, doc['baseline_prediction_hash'])
    else:
        row = doc['candidates'][index]
        name = f'candidate_{index:02d}'
        prediction = materialize_checked(doc, row, base / name, False, device)
    result = evaluate(Path(doc['subset_annotation']), prediction, base / name / 'evaluation', name, cores)
    print(json.dumps({'completed': name, 'base': result['parsed']['base'], 'novel': result['parsed']['novel']}), flush=True)


def official_rank(row):
    b, m = row['official']['parsed']['base'], row['selection_metrics']
    return (b['AssocA'], b['TETA'], m['precision'], m['f1'], -m['false_merge'], -row['config_index'])


def finish(frontend, device, cores):
    base = ROOT / 'v9_3/pareto' / frontend
    doc = read(base / 'candidates.json')
    rows = []
    for row in doc['candidates']:
        official = read(base / f"candidate_{row['config_index']:02d}" / 'evaluation/evaluation.json')
        verify(official['prediction'], official['prediction_hash']); verify(official['summary'], official['summary_hash'])
        if official['annotation_hash'] != doc['subset_annotation_hash']: raise ValueError('selection subset mismatch')
        rows.append({**row, 'official': official})
    baseline = read(base / 'baseline/evaluation/evaluation.json')
    winner = max(rows, key=official_rank)
    gain = winner['official']['parsed']['base']['AssocA'] - baseline['parsed']['base']['AssocA']
    decision = 'PROXY_GATE_MISMATCH' if gain > 0 and REASONS[0] in doc['category_absent'] else ('CURRENT_PSMR_SCORER_NO_OFFICIAL_GAIN' if gain <= 0 else 'OFFICIAL_BASE_GAIN')
    selected = {'protocol': 'TEST_BASE_ADAPTED', 'selection_base_only': True, 'novel_used_for_selection': False, 'tie_break': ['Base AssocA', 'Base TETA', 'proxy precision', 'proxy F1', 'lower false_merge', 'lower config_index'], 'baseline': baseline, 'rows': rows, 'final_selected': winner, 'base_gain': gain, 'decision': decision, 'candidates_hash': sha(base / 'candidates.json')}
    write(base / 'selected.json', selected)
    prediction = materialize_checked(doc, winner, base / 'full_test', True, device)
    official = evaluate(Path(doc['inputs']['annotation']), prediction, base / 'full_test/evaluation', frontend + '_pareto_test', cores)
    write(base / 'result.json', {'selection': selected, 'full_test': official})
    print(json.dumps({'frontend': frontend, 'winner': winner['config_index'], 'decision': decision, 'subset_base_gain': gain, 'full_test_base': official['parsed']['base'], 'full_test_novel': official['parsed']['novel']}), flush=True)


def run_lane(gpus, cores, adopt_pids=''):
    """Dispatch already prepared candidates across explicitly reserved free GPUs."""
    import queue
    import threading
    gpu_ids = [int(g) for g in gpus.split(',')]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(g not in (3, 4, 5) for g in gpu_ids):
        raise ValueError('Pareto lane is restricted to distinct GPUs 3-5')
    if len(gpu_ids) > 2 or cores != 1:
        raise ValueError('coordinated Pareto budget: at most two jobs, one evaluator core per job')
    free = queue.Queue()
    bindings = {}
    adopted = []
    for pid_text in filter(None, adopt_pids.split(',')):
        pid = int(pid_text)
        proc = Path('/proc') / str(pid)
        if not proc.exists(): continue
        argv = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        if str(Path(__file__).resolve()) not in argv or 'candidate' not in argv:
            raise ValueError(f'PID {pid} is not this tool candidate worker')
        f, index = argv[argv.index('--frontend') + 1], int(argv[argv.index('--index') + 1])
        env = proc.joinpath('environ').read_bytes().split(b'\0')
        uuid = next(e.split(b'=', 1)[1].decode() for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES='))
        start_ticks = proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
        adopted.append({'pid': pid, 'frontend': f, 'index': index, 'uuid': uuid, 'start_ticks': start_ticks})
    for gpu in gpu_ids:
        uuid, free_mib = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        # Reserve 6 GiB for this worker and leave at least 12 GiB free.
        if int(free_mib.strip()) < 18 * 1024: raise RuntimeError(f'GPU {gpu} has insufficient free VRAM')
        bindings[gpu] = uuid.strip()
        if not any(a['uuid'] == bindings[gpu] for a in adopted): free.put(gpu)
    if len({a['uuid'] for a in adopted}) != len(adopted) or any(a['uuid'] not in bindings.values() for a in adopted):
        raise ValueError('adopted workers must occupy distinct assigned UUID slots')
    root = ROOT / 'v9_3/pareto'
    write(root / f'resource_reservation_{os.getpid()}.json', {'owner': 'v9_pareto_shortlist', 'pid': os.getpid(), 'gpu_uuids': bindings, 'adopted_workers': adopted, 'max_candidate_processes': len(gpu_ids), 'max_full_test_jobs': 1, 'cores_per_evaluation': cores, 'ram_reserve_gib': 12, 'full_job_admission_gib': 44, 'protected_pids': [31022, 22412]})
    full_lock = threading.Lock()
    def job(action, frontend, index=None):
        if action == 'finish': full_lock.acquire()
        gpu = free.get()
        try:
            memory_gate(44 if action == 'finish' else 20)
            free_mib = int(subprocess.check_output(['nvidia-smi', '-i', bindings[gpu], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free_mib < 18 * 1024: raise RuntimeError(f'GPU {gpu} free VRAM changed below admission budget')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=bindings[gpu], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', LD_PRELOAD='/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0')
            env['PYTHONPATH'] = str(REPO) + ':/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI'
            command = [sys.executable, str(Path(__file__).resolve()), action, '--frontend', frontend, '--device', 'cuda:0', '--cores', str(cores)]
            if index is not None: command += ['--index', str(index)]
            log = root / frontend / f'{action}_{index if index is not None else "winner"}_{time.time_ns()}.log'
            with log.open('x') as f:
                p = subprocess.Popen(command, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
                print(json.dumps({'started': action, 'frontend': frontend, 'index': index, 'gpu': gpu, 'gpu_uuid': bindings[gpu], 'pid': p.pid, 'log': str(log)}), flush=True)
                code = p.wait()
            if code: raise RuntimeError(f'job failed ({code}); preserved log: {log}')
            print(json.dumps({'completed': action, 'frontend': frontend, 'index': index}), flush=True)
        finally:
            free.put(gpu)
            if action == 'finish': full_lock.release()
    jobs = []
    docs = {f: read(root / f / 'candidates.json') for f in ('vovtrack', 'covtrack')}
    # Round-robin frontend/config scheduling, bounded by physical GPU slots.
    for index in range(-1, max(len(d['candidates']) for d in docs.values())):
        for frontend, doc in docs.items():
            if index < len(doc['candidates']): jobs.append((frontend, index))
    outcomes = {f: {'candidate_completed': [], 'errors': [], 'full_test': 'PENDING'} for f in docs}
    remaining = {f: len(d['candidates']) + 1 for f, d in docs.items()}
    # Durable successful artifacts, including externally recovered summaries,
    # are the source of truth rather than stale in-memory exception history.
    def completed(frontend, index):
        name = 'baseline' if index == -1 else f'candidate_{index:02d}'
        path = root / frontend / name / 'evaluation/evaluation.json'
        if not path.exists(): return False
        result = read(path)
        if result['status'] != 'COMPLETED' or result['annotation_hash'] != docs[frontend]['subset_annotation_hash']:
            raise ValueError('completed candidate status/subset mismatch')
        verify(result['summary'], result['summary_hash'])
        verify(result['prediction'], result['prediction_hash'])
        if index >= 0:
            selected = read(root / frontend / name / 'selected_config.json')
            if selected != docs[frontend]['candidates'][index]: raise ValueError('completed config mapping mismatch')
        return True
    def adopt(worker):
        pid, frontend, index = worker['pid'], worker['frontend'], worker['index']
        gpu = next(g for g, uuid in bindings.items() if uuid == worker['uuid'])
        print(json.dumps({'adopted': worker}), flush=True)
        try:
            while True:
                try: stat = Path(f'/proc/{pid}/stat').read_text()
                except FileNotFoundError: break
                fields = stat.split(') ', 1)[1].split()
                if fields[0] == 'Z' or fields[19] != worker['start_ticks']: break
                time.sleep(2)
            name = 'baseline' if index == -1 else f'candidate_{index:02d}'
            if index >= 0 and not (root / frontend / name / 'invariance.json').exists():
                raise RuntimeError(f'adopted materializer {pid} failed; partial outputs preserved')
        finally:
            free.put(gpu)
        # Existing prediction/invariance causes a verified reuse, never replay.
        job('candidate', frontend, index)
    from collections import deque
    pending = deque()
    adopted_keys = {(w['frontend'], w['index']) for w in adopted}
    for f, i in jobs:
        if (f, i) in adopted_keys: continue
        if completed(f, i):
            outcomes[f]['candidate_completed'].append(i); remaining[f] -= 1
        else: pending.append(('candidate', f, i))
    for f in docs:
        if remaining[f] == 0: pending.appendleft(('finish', f, None))
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        running = {pool.submit(adopt, w): ('candidate', w['frontend'], w['index']) for w in adopted}
        while pending or running:
            while pending and len(running) < len(gpu_ids):
                action, f, i = pending.popleft()
                running[pool.submit(job, action, f, i)] = (action, f, i)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                action, frontend, index = running.pop(future)
                try:
                    future.result()
                    if action == 'finish': outcomes[frontend]['full_test'] = 'COMPLETED'
                    else: outcomes[frontend]['candidate_completed'].append(index)
                except Exception as exc:
                    outcomes[frontend]['errors'].append({'stage': action, 'index': index, 'error': str(exc)})
                    if action == 'finish': outcomes[frontend]['full_test'] = 'FAILED'
                    print(json.dumps({'frontend': frontend, 'failed': action, 'index': index, 'error': str(exc)}), flush=True)
                if action == 'candidate':
                    remaining[frontend] -= 1
                    # Reconcile a wrapper failure already repaired by a
                    # separate hash-bound summary recovery before freezing.
                    if remaining[frontend] == 0 and outcomes[frontend]['errors']:
                        unresolved = []
                        for error in outcomes[frontend]['errors']:
                            if error['stage'] == 'candidate' and completed(frontend, error['index']):
                                outcomes[frontend]['candidate_completed'].append(error['index'])
                            else: unresolved.append(error)
                        outcomes[frontend]['errors'] = unresolved
                    if remaining[frontend] == 0 and not outcomes[frontend]['errors']:
                        pending.appendleft(('finish', frontend, None))
    write(root / f'run_outcomes_{os.getpid()}.json', outcomes)
    print(json.dumps({'pareto_outcomes': outcomes}), flush=True)


def validate_shortlist():
    """Small finite regression evidence; never launches inference or a grid."""
    import ast
    import copy
    from tempotrack_research.orchestration.v9_parameter_search import _resolve_materialize_checkpoint
    checks = []
    ast.parse(Path(__file__).read_text()); checks.append('python_syntax')
    root = ROOT / 'v9_3/pareto'
    docs = {f: read(root / f / 'candidates.json') for f in ('vovtrack', 'covtrack')}
    row = copy.deepcopy(docs['vovtrack']['candidates'][0])
    rank = ranks(row)
    row['application_metrics'] = {'precision': -1e20, 'recall': 1e20, 'f1': 1e20, 'novel_AssocA': 1e20}
    assert ranks(row) == rank; checks.append('proxy_selection_ignores_application_and_novel_metrics')
    row['official'] = {'parsed': {'base': {'AssocA': 40., 'TETA': 40.}, 'novel': {'AssocA': 1e20}}}
    other = copy.deepcopy(row); other['official']['parsed']['base']['AssocA'] += .01
    other['official']['parsed']['novel']['AssocA'] = -1e20
    assert official_rank(other) > official_rank(row); checks.append('official_selection_ignores_novel')
    for field, value in [('protocol', 'TEST_FULL_ORACLE'), ('selection_status', 'INVALID'), ('split', 'val')]:
        bad = copy.deepcopy(row); bad[field] = value
        try: validate_row(bad, 'vovtrack')
        except ValueError: pass
        else: raise AssertionError(f'accepted invalid {field}')
    checks.append('reject_oracle_invalid_status_and_wrong_split')
    bad = copy.deepcopy(row); bad['selection_metrics']['precision'] = float('nan')
    try: validate_row(bad, 'vovtrack')
    except ValueError: pass
    else: raise AssertionError('accepted NaN')
    checks.append('reject_nonfinite_proxy_metrics')
    bad = {'schema_version': 10, 'protocol': 'TEST_BASE_ADAPTED', 'support_cache_semantics': 'legacy masked cache'}
    try: validate_sweep(bad)
    except ValueError: pass
    else: raise AssertionError('accepted stale support semantics')
    checks.append('reject_uncorrected_support_cache_semantics')
    raw = next(c for c in docs['covtrack']['candidates'] if 'H_raw_control' in c['reasons'])
    b = {}; offer(b, raw); offer(b, copy.deepcopy(raw))
    assert all(len(v) == 1 for v in b.values())
    assert _resolve_materialize_checkpoint(raw, raw['config'], '/does/not/exist') is None
    checks.append('raw_lambda_zero_deduplicated_across_reasons_and_checkpoint_disabled')
    # Exact brief key includes source checkpoint hash/step, even for lambda=0.
    for frontend, d in docs.items():
        assert len(d['candidates']) <= 12 and len({identity(c) for c in d['candidates']}) == len(d['candidates'])
        assert len(d['subset_video_ids']) == 128
        for c in d['candidates']: validate_row(c, frontend)
    checks.append('both_real_shortlists_max12_exact_key_unique_and_exact128_subset')
    learned = next(c for c in docs['vovtrack']['candidates'] if c['config']['reliability_multiplier'] > 0)
    cp = _resolve_materialize_checkpoint(learned, learned['config'], '/does/not/exist')
    assert cp.resolve() == Path(learned['checkpoint']).resolve() and sha(cp) == learned['checkpoint_hash']
    bad = copy.deepcopy(learned); bad['checkpoint_hash'] = '0' * 64
    try: _resolve_materialize_checkpoint(bad, bad['config'], None)
    except ValueError: pass
    else: raise AssertionError('accepted wrong checkpoint hash')
    checks.append('selected_exact_checkpoint_overrides_argument_and_rejects_hash_mismatch')
    for frontend in docs:
        d = read(root / frontend / 'baseline/evaluation/evaluation.json')
        verify(d['summary'], d['summary_hash'])
        assert all(math.isfinite(v) for group in ('base', 'novel') for v in d['parsed'][group].values())
    checks.append('real_official_baselines_parsed_and_all_base_novel_metrics_finite')
    result = {'status': 'PASS', 'checks': checks, 'test_count': len(checks), 'new_grid': False,
              'tool': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
              'candidates': {f: sha(root / f / 'candidates.json') for f in docs},
              'checkpoint_resolution_source_hash': sha(REPO / 'tempotrack_research/orchestration/v9_parameter_search.py'),
              'parser_source_hash': sha(REPO / 'tempotrack_research/evaluation/teta_parser.py')}
    path = root / 'validation.json'; write(path, result)
    print(json.dumps({'status': 'PASS', 'tests': checks, 'evidence': str(path), 'sha256': sha(path), 'tool_sha256': sha(__file__)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'candidate', 'finish', 'run', 'validate'])
    p.add_argument('--frontend', choices=['vovtrack', 'covtrack'])
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--adopt-pids', default='', help='own live candidate PIDs to wait for and reuse after coordinator handoff')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--index', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cores', type=int, default=1)
    a = p.parse_args()
    if a.action == 'validate': validate_shortlist()
    elif a.action == 'run': run_lane(a.gpus, a.cores, a.adopt_pids)
    elif a.action == 'prepare': prepare(a.frontend, a.workers)
    elif a.action == 'candidate': candidate(a.frontend, a.index, a.device, a.cores)
    else: finish(a.frontend, a.device, a.cores)


if __name__ == '__main__': main()
