"""Build independent query-tile tail cases from the retained K6 equation fixture."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('fixture', type=Path, help='Existing c2051 independent fixture')
p.add_argument('out', type=Path)
a = p.parse_args()
d = json.loads((a.fixture / 'descriptor.json').read_text())
assert d['capacity'] == 2051 and d['rows'] == 8
for rows in [33, 49, 63]:
    root = a.out / f'r{rows}'
    root.mkdir(parents=True)
    index = np.arange(rows) % d['rows']
    for name, dtype, width in [('q-absorbed.f32', '<f4', 64 * 512),
                               ('selected.i32', '<i4', 2051), ('positions.i32', '<i4', 1),
                               ('baseline-output.f32', '<f4', 64 * 512)]:
        x = np.fromfile(a.fixture / ('fused-0-' + name), dtype=dtype).reshape(d['rows'], width)
        x[index].tofile(root / ('fused-0-' + name))
    expected = np.fromfile(a.fixture / 'oracle.f64', dtype='<f8').reshape(8, 64 * 512)
    expected[index].tofile(root / 'oracle.f64')
    os.link(a.fixture / 'fused-0-ckv-cache.bin', root / 'fused-0-ckv-cache.bin')
    (root / 'descriptor.json').write_text(json.dumps({**d, 'rows': rows,
        'derived_from': 'eight real-operand masked-selection fixtures', 'query_tile_tail': rows % 32}, indent=2) + '\n')
