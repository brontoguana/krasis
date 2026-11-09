"""Check complete paired prefill evidence without changing numerical gates."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from compare_runs import states

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--accepted', type=Path, help='Prior accepted run containing numerical{history} trials')
p.add_argument('--quality-baseline', type=Path)
a = p.parse_args()
rows = []
for n in [3072, 16384, 65536]:
    left = a.run / 'teacher' / f'pre{n}proof0'
    right = a.run / 'teacher' / f'pre{n}proof1'
    files = ['first-logits.f32'] + [f'step-{i}-logits.f32' for i in range(63)]
    values = 0
    for name in files:
        x = np.fromfile(left / name, dtype='<f4')
        y = np.fromfile(right / name, dtype='<f4')
        assert x.size and x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        assert np.array_equal(x.view('u4'), y.view('u4')), (n, name, 'FP32 candidate diverged')
        values += x.size
    state = states(left / 'final-state', right / 'final-state')
    assert state['bit_exact'], (n, 'registered state diverged')
    if a.accepted:
        old = a.accepted / 'teacher' / f'numerical{n}'
        assert all((left / name).read_bytes() == (old / name).read_bytes() for name in files)
        assert states(old / 'final-state', left / 'final-state')['bit_exact']
    paired = []
    for mode, suffixes in [(0, ['0a', '0b']), (1, ['1a', '1b'])]:
        trials = [json.loads((a.run / f'pre{n}{suffix}-validation.json').read_text()) for suffix in suffixes]
        for v in trials:
            assert v['receipt']['prefill_attention_execution'] == mode
        paired.append({'mode': mode, 'wall_s': [v['wall_seconds'] for v in trials],
                       'median_s': statistics.median(v['wall_seconds'] for v in trials),
                       'decode_ms_per_forward': [1000 * v['receipt']['decode_elapsed_seconds'] / v['receipt']['completed_decode_steps'] for v in trials],
                       'median_decode_ms_per_forward': statistics.median(1000 * v['receipt']['decode_elapsed_seconds'] / v['receipt']['completed_decode_steps'] for v in trials)})
    rows.append(dict(history=n, logit_vectors=64, logit_values=values, all_bitwise=True,
                     state_bytes=state['bytes'], state_allocations=state['allocations'],
                     paired=paired, wall_improvement_pct=100 * (1 - paired[1]['median_s'] / paired[0]['median_s'])))
quality = None
if a.quality_baseline:
    baseline = json.loads((a.quality_baseline / 'quality/RESULT.json').read_text())
    candidate = json.loads((a.run / 'quality/RESULT.json').read_text())
    before = {v['label']: v for v in baseline['rows']}
    after = {v['label']: v for v in candidate['rows']}
    assert before.keys() == after.keys()
    quality = [{'label': k, 'baseline_pass': before[k]['pass'], 'candidate_pass': after[k]['pass']} for k in before]
    assert all(not r['baseline_pass'] or r['candidate_pass'] for r in quality), 'new frozen-oracle failure'
result = dict(scope='Exact relative prefill regression; no independent whole-model admission',
              rows=rows, quality=quality, thresholds_changed=False)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**result, 'rows': [{k: v for k, v in r.items() if k != 'paired'} for r in rows]}, indent=2))
