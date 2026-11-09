"""Independent host arithmetic/range proof for the offline K6 scale screen."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('fixtures', type=Path)
p.add_argument('prior_runs', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
# BF16 bits, decoded without runtime pack/unpack helpers; exact binary integers.
u = np.arange(65536, dtype=np.uint32)
finite = (u & 0x7f80) != 0x7f80
scales = (u[finite] << 16).view(np.float32)
codes = np.arange(-32, 32, dtype=np.float32)
# TF32 preserves the top ten fractional bits; all integer-code low bits vanish.
assert np.all((codes.view(np.uint32) & 0x1fff) == 0)
exact = scales.astype(np.float64)[:, None] * codes.astype(np.float64)
with np.errstate(over='ignore'):
    native = scales[:, None] * codes
in_range = np.isfinite(native)
assert np.all(native[in_range].astype(np.float64) == exact[in_range])
# Scale/code multiplication is exact in FP32 whenever its result is finite.
# Arbitrary finite BF16 is not enough to guarantee that result is finite.
probabilities = np.array([0., np.nextafter(np.float32(0), np.float32(1)),
                         np.finfo(np.float32).tiny, 2.**-20, 0.1, 0.5, 1.], dtype=np.float32)
positive = scales[scales >= 0]
with np.errstate(over='ignore', under='ignore', invalid='ignore'):
    ps = probabilities[:, None] * positive[None, :]
assert np.isfinite(ps).all()
# Explicit witnesses: p*s can underflow although p*(s*code) remains representable;
# original s*code can overflow although (p*s)*code is finite.
s = np.float32(2.**-133)
pv = np.float32(2.**-20)
old_under = np.float32(pv * np.float32(s * np.float32(31)))
new_under = np.float32(np.float32(pv * s) * np.float32(31))
with np.errstate(over='ignore'):
    smax = np.float32(positive.max())
    old_over = np.float32(np.float32(.01) * np.float32(smax * np.float32(31)))
    new_over = np.float32(np.float32(np.float32(.01) * smax) * np.float32(31))
assert old_under != new_under and np.isinf(old_over) and np.isfinite(new_over)
rows = []
for fixture in sorted(a.fixtures.iterdir()):
    raw = np.fromfile(fixture/'fused-0-ckv-cache.bin', dtype='u1').reshape(-1, 32, 14)
    # Little-endian BF16 scale at the start of each 14-byte block.
    scale_bits = raw[:, :, 0].astype('u4') | (raw[:, :, 1].astype('u4') << 8)
    ss = (scale_bits << 16).view('f4')
    probs = np.fromfile(a.prior_runs/('run-'+fixture.name)/'probs.f32', dtype='<f4')
    assert np.isfinite(ss).all() and (ss >= 0).all()
    assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
    positive_s, positive_p = ss[ss > 0], probs[probs > 0]
    minimum = float(positive_s.min()) * float(positive_p.min()) if len(positive_s) and len(positive_p) else 0.
    maximum = float(ss.max()) * float(probs.max())
    assert minimum == 0. or minimum >= np.finfo('f4').tiny
    assert float(ss.max()) * 32 < np.finfo('f4').max
    assert maximum * 32 < np.finfo('f4').max
    rows.append(dict(fixture=fixture.name,scale_min=float(ss.min()),scale_max=float(ss.max()),
                     probability_min_positive=float(positive_p.min()) if len(positive_p) else 0.,
                     probability_max=float(probs.max()),conservative_nonzero_scaled_probability_min=minimum,
                     scaled_probability_max_bound=maximum))
result = dict(all_passed=True,finite_bf16_patterns=len(scales),codes=64,
              finite_exact_scale_code_products=int(in_range.sum()),
              overflow_scale_code_products=int((~in_range).sum()),
              code_tf32_residual_zero=True,
              subnormal_witness=dict(original=float(old_under),factored=float(new_under)),
              overflow_witness=dict(original='infinity',factored=float(new_over)),
              fixtures=rows,
              contract='Host FP32 RN multiplication boundaries, FP64 exact scale-code products. Codes exact TF32. Factored probabilities require high/residual TF32; subnormals/overflow exclude universal equivalence. Actual retained ranges are normal and finite. No GPU/model correctness claim.')
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(all_passed=True,fixtures=len(rows),finite_exact_products=int(in_range.sum()),overflow_products=int((~in_range).sum()))))
