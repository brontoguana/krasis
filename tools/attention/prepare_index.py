"""Prepare actual captured index scores and the unchanged native selection kernels."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('capture', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir()
s = (a.source / 'src/cuda/decode_kernels.cu').read_text()
start = s.index('__device__ inline bool dsa_topk_precedes(')
end = s.index('\n}', s.index('extern "C" __global__ void dsa_topk_linear_merge_runs_g(')) + 2
(a.output / 'index_kernels.cuh').write_text(
    '#include <cuda_runtime.h>\n#include <cub/block/block_radix_sort.cuh>\n'
    '#include <cstdint>\n#include <climits>\n' + s[start:end])
layout = json.loads((a.capture / 'indexer-score-layout.json').read_text())
scores = np.fromfile(a.capture / 'indexer-scores.f32', dtype='<f4')
assert scores.size == layout['copied_causal_scores'] and np.isfinite(scores).all()
scores.tofile(a.output / 'scores.f32')
(a.output / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n')
