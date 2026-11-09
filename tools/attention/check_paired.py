"""Check complete native graph outputs against exact staged K6 equations."""
import argparse
import json
from pathlib import Path

import numpy as np
from prepare_replay import oracle

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--wvc', action='store_true')
a = p.parse_args()
rows = []
for path in sorted((a.run / 'teacher').glob('*/mla-paired-graph/layer-*/descriptor.json')):
    root = path.parent
    d = json.loads(path.read_text())
    assert d['heads'] == 64 and d['cache_dim'] == 512 and d['rope'] == 0
    count = np.fromfile(root / 'count.i32', dtype='<i4')
    sequence = np.fromfile(root / 'sequence.i32', dtype='<i4')
    assert count.size == sequence.size == 1
    assert 0 < count[0] <= d['capacity'] and sequence[0] == d['logical_tokens']
    oracle(root, d['context'], d['scale'])
    expected = np.fromfile(root / 'oracle.f64', dtype='<f8')
    outputs = {}
    for name in ['baseline', 'candidate']:
        value = np.fromfile(root / f'{name}-graph.f32', dtype='<f4')
        assert value.shape == expected.shape == (32768,)
        delta = value.astype('f8') - expected
        outputs[name] = dict(finite=bool(np.isfinite(value).all()),
                             max_abs=float(np.max(np.abs(delta))),
                             rms=float(np.sqrt(np.mean(delta * delta))),
                             over_2e_4=int(np.count_nonzero(np.abs(delta) > 2e-4)))
    row = dict(trial=root.parent.parent.name, layer=d['layer'],
               logical_tokens=d['logical_tokens'], values=expected.size,
               outputs=outputs)
    row['operator_pass'] = all(r['finite'] and r['over_2e_4'] == 0 for r in outputs.values())
    if a.wvc:
        projected = {}
        for name in ['baseline', 'candidate']:
            f = np.fromfile(root / f'wvc-{name}.f32', dtype='<f4')
            b = np.fromfile(root / f'wvc-{name}.bf16', dtype='<u2')
            assert f.size == b.size == d['heads'] * d['value_dim'] and np.isfinite(f).all()
            bits = f.view('<u4')
            rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')
            assert np.array_equal(rounded, b), 'pre-rounding replay does not reproduce the native BF16 boundary'
            projected[name] = (f, b)
        left, right = projected['baseline'], projected['candidate']
        changed = np.flatnonzero(left[1] != right[1])
        row['wvc'] = dict(values=left[0].size, differing_bf16=int(changed.size),
                          max_pre_round_abs=float(np.max(np.abs(left[0].astype('f8') - right[0]))))
        if changed.size:
            i = int(changed[0])
            values = [(np.array([v[1][i]], dtype='<u2').astype('<u4') << 16).view('<f4')[0]
                      for v in [left, right]]
            row['wvc']['first_boundary'] = dict(index=i,
                baseline_before=float(left[0][i]), candidate_before=float(right[0][i]),
                baseline_bf16=float(values[0]), candidate_bf16=float(values[1]),
                midpoint=float((float(values[0]) + float(values[1])) / 2),
                adjacent_bf16=abs(int(left[1][i])-int(right[1][i])) == 1,
                straddles_midpoint=min(float(left[0][i]),float(right[0][i])) <= (float(values[0])+float(values[1]))/2 <= max(float(left[0][i]),float(right[0][i])))
    rows.append(row)
assert rows, 'no paired graph evidence'
result = dict(threshold_abs=2e-4, cases=len(rows), values_per_implementation=sum(r['values'] for r in rows),
              all_operator_pass=all(r['operator_pass'] for r in rows), rows=rows)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))

assert result['all_operator_pass'], 'paired graph equations exceed the unchanged operator threshold; evidence retained'
