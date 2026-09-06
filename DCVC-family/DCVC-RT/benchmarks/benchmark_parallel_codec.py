#!/usr/bin/env python3
"""Alternate saved-baseline/candidate runs with a barrier for simultaneous model workers."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser(__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--frames', type=int, default=160)
p.add_argument('--pipeline', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
results = []
for workers in (1,3):
    for rep in range(a.repeats):
        order = ('baseline','candidate') if rep%2 == 0 else ('candidate','baseline')
        for variant in order:
            run = a.output/f'{workers}w_{rep}_{variant}'
            run.mkdir()
            go = run/'go'
            processes, logs = [], []
            try:
                for i in range(workers):
                    log = (run/f'worker_{i}.log').open('w')
                    logs.append(log)
                    cmd = [sys.executable,str(Path(__file__).with_name('benchmark_codec_pipeline.py' if a.pipeline else 'benchmark_synthetic_codec.py')),
                           '--fixture',str(a.fixture),'--output',str(run/f'worker_{i}'),
                           '--extension',str(getattr(a,variant)), '--speed-only',
                           '--frames',str(a.frames), '--go-file',str(go)]
                    processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
                deadline = time.monotonic()+180
                while not all((run/f'worker_{i}/ready').exists() for i in range(workers)):
                    if any(proc.poll() is not None for proc in processes):
                        raise RuntimeError(f'worker exited before barrier: {run}')
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'worker startup timeout: {run}')
                    time.sleep(.1)
                go.touch()
                for proc in processes:
                    if proc.wait(timeout=180) != 0:
                        raise RuntimeError(f'worker failed: {run}')
                data = [json.loads((run/f'worker_{i}/result.json').read_text()) for i in range(workers)]
                elapsed = max(x['encode_end_unix'] for x in data)-min(x['encode_start_unix'] for x in data)
                row = dict(workers=workers,repeat=rep,variant=variant,
                           per_worker_fps=[x['steady_encode_fps'] for x in data],
                           aggregate_fps=sum(x["frames"] for x in data)/elapsed,
                           start_spread_seconds=max(x['encode_start_unix'] for x in data)-min(x['encode_start_unix'] for x in data),
                           rss_kib=[x['max_rss_kib'] for x in data],
                           hashes=[x['stream_sha256'] for x in data])
                results.append(row)
                (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(row),flush=True)
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                for log in logs:
                    log.close()
assert len({h for r in results for h in r['hashes']}) == 1, 'worker streams differ'
for workers in (1,3):
    for variant in ('baseline','candidate'):
        selected = [r for r in results if r['workers']==workers and r['variant']==variant]
        print(f'{workers} workers {variant}: median aggregate={statistics.median(r["aggregate_fps"] for r in selected):.3f} fps; '
              f'median steady worker={statistics.median(v for r in selected for v in r["per_worker_fps"]):.3f} fps',flush=True)
