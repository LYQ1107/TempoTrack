#!/usr/bin/env python3
"""Boundary checkpoint and pidfd-validated retirement of the authorized VOV owner."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import time

from v9_active_recover_selection import recover_log, validate_row
from v9_dual_recover_selection import read, sha, write_new


def identity(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'uid': root.stat().st_uid, 'cwd': os.readlink(root / 'cwd'),
        'argv': [s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s],
        'start_ticks': fields[19]}


def exited(pid, start_ticks=None):
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return True
    return stat[0] == 'Z' or (start_ticks is not None and stat[19] != start_ticks)


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('no validated pidfd syscall mapping for this platform')
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(434, int(pid), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed; refusing PID-only signal')
    return descriptor


def pidfd_term(descriptor):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, int(descriptor), int(signal.SIGTERM), ctypes.c_void_p(), 0) < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--owner-pid', type=int, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from v9_active_tracker_search import _configs
    repo = Path(__file__).resolve().parents[1]
    root = Path('/data2/usr_for_deadline/tempotrack_v9_relocated_20260910')
    source_path = root / 'active/vovtrack/test/v92_shard_1/active_shard.json'
    source = read(source_path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'migration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = identity(args.owner_pid)
    expected_script = str(repo / 'tools/v9_active_tracker_search.py')
    assert owner['uid'] == os.getuid() and owner['cwd'] == str(repo), owner
    assert len(owner['argv']) > 2 and owner['argv'][1] == expected_script, owner
    options = dict(zip(owner['argv'][2::2], owner['argv'][3::2]))
    for key, value in {'--frontend': 'vovtrack', '--config-shard-count': '3', '--config-shard-index': '0',
        '--calls-root': source['calls_root'], '--selection-annotation': source['selection_annotation'],
        '--model-config': source['model_config'], '--model-checkpoint': source['model_checkpoint'],
        '--output': str(root / 'active/vovtrack/test/v92_shard_0')}.items():
        assert options.get(key) == value, (key, options.get(key), value)
    log = root / 'active/vovtrack/test/v92_shard_0.log'
    assert os.readlink(Path('/proc') / str(args.owner_pid) / 'fd/1') == str(log)
    # The descriptor pins process identity even if the numeric PID is reused.
    pidfd = pidfd_open(args.owner_pid)
    assert identity(args.owner_pid) == owner
    configs = _configs('vovtrack')
    expected = {i for i in range(len(configs)) if i % 3 == 0}
    initial, _ = recover_log([log], configs, expected)
    inputs = [source_path, Path(source['selection_annotation']), Path(source['equivalence_annotation']),
        Path(source['model_config']), Path(source['model_checkpoint']), Path(expected_script)]
    inputs += sorted(Path(source['calls_root']).glob('match_calls_*'))
    input_hashes = {str(path): sha(path) for path in inputs}
    write_new(output / 'owner_identity.json', {'status': 'AWAITING_NEXT_COMPLETED_BOUNDARY',
        'owner': owner, 'initial_completed': len(initial), 'input_hashes': input_hashes})
    print(json.dumps({'status': 'AWAITING_NEXT_COMPLETED_BOUNDARY', 'owner': owner,
        'initial_completed': len(initial)}), flush=True)
    while True:
        if identity(args.owner_pid) != owner:
            raise RuntimeError('owner identity changed; refusing all signals')
        rows, rejected = recover_log([log], configs, expected)
        if len(rows) > len(initial):
            break
        time.sleep(.5)
    # Freeze the actual bytes before retirement; durable writes precede signal.
    snapshot = output / 'shard0_boundary.log'
    with snapshot.open('xb') as stream:
        stream.write(log.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    rows, rejected = recover_log([snapshot], configs, expected)
    remaining = sorted(expected - {r['config_index'] for r in rows})
    checkpoint = {'status': 'BOUNDARY_CHECKPOINTED', 'owner': owner, 'rows': rows,
        'remaining': remaining, 'log_snapshot': str(snapshot), 'log_hash': sha(snapshot),
        'rejected_partial_lines': rejected, 'input_hashes': input_hashes}
    write_new(output / 'boundary_checkpoint.json', checkpoint)
    with (output / 'boundary_checkpoint.json').open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
    assert identity(args.owner_pid) == owner
    pidfd_term(pidfd)
    print(json.dumps({'status': 'VALIDATED_OWNER_TERM_SENT', 'completed': len(rows), 'remaining': remaining}), flush=True)
    for _ in range(120):
        if exited(args.owner_pid, owner['start_ticks']):
            break
        time.sleep(.5)
    else:
        raise RuntimeError('owner has not exited; no workers launched')
    os.close(pidfd)
    # Include any row flushed between the checkpoint and process exit.
    terminal = output / 'shard0_retired.log'
    with terminal.open('xb') as stream:
        stream.write(log.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    rows, rejected = recover_log([terminal], configs, expected)
    checkpoint.update(status='OWNER_RETIRED', rows=rows,
        remaining=sorted(expected - {r['config_index'] for r in rows}),
        final_log_snapshot=str(terminal), final_log_hash=sha(terminal))
    retired_path = output / 'owner_retired.json'
    write_new(retired_path, checkpoint)
    requested_gpus = [int(v) for v in args.gpus.split(',')]
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    eligible = {int(parts[0]): parts[1].strip() for line in devices
        if len(parts := line.split(',')) == 3 and int(parts[2]) >= 12000}
    gpus = [g for g in requested_gpus if g in eligible]
    if not gpus:
        raise RuntimeError('no requested idle GPU; durable remaining checkpoint preserved')
    workers = []
    for slot, gpu in enumerate(gpus):
        indices = checkpoint['remaining'][slot::len(gpus)]
        if not indices:
            continue
        worker_output = output / f'worker_gpu{gpu}'
        worker_output.mkdir()
        cmd = ['nohup', 'setsid', '/home/lwr/anaconda3/envs/ovtr/bin/python',
            str(repo / 'tools/v9_active_recovery_worker.py'), '--config-indices', ','.join(map(str, indices)),
            '--checkpoint', str(retired_path), '--source', str(source_path), '--output', str(worker_output),
            '--external-root', '/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=eligible[gpu], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(repo))
        with (worker_output / 'worker.log').open('x') as log_stream:
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log_stream, stderr=subprocess.STDOUT)
        workers.append({'pid': proc.pid, 'gpu': gpu, 'gpu_uuid': eligible[gpu], 'config_indices': indices, 'output': str(worker_output)})
    write_new(output / 'workers.json', {'status': 'RUNNING', 'workers': workers})
    print(json.dumps({'status': 'WORKERS_LAUNCHED', 'workers': workers}), flush=True)
    while not all((Path(w['output']) / 'worker_complete.json').exists() for w in workers):
        for worker in workers:
            if exited(worker['pid']) and not (Path(worker['output']) / 'worker_complete.json').exists():
                raise RuntimeError(f"worker failed: {worker}")
        time.sleep(15)
    merged = {r['config_index']: r for r in checkpoint['rows']}
    for path in sorted((output / 'completed').glob('config_*.json')):
        row = validate_row(read(path), configs)
        if row['config_index'] in merged:
            raise ValueError('duplicate completion at shard merge')
        merged[row['config_index']] = row
    if set(merged) != expected:
        raise ValueError('shard0 coverage incomplete')
    write_new(output.parent / 'active_shard.json', {**source, 'status': 'SHARD_COMPLETE',
        'artifact': 'v9_3_active_recovered_shard', 'config_shard_index': 0,
        'rows': [merged[i] for i in sorted(merged)], 'configs': len(merged),
        'migration_checkpoint_hash': sha(retired_path)})
    print(json.dumps({'status': 'SHARD_COMPLETE', 'configs': len(merged)}), flush=True)


if __name__ == '__main__':
    main()
