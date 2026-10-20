//! Exact integer superaccumulator for finite BF16 dot products, direct BF16 RNE.
//! Units and exponent limits below derive from IEEE BF16, not a model or GPU.
const FRACTION_BITS: i32 = 7;
const BIAS: i32 = 127;
const UNIT_EXP: i32 = 2 * (1 - BIAS - FRACTION_BITS); // least subnormal squared: 2^-266
const PRODUCT_BITS: usize =
    (2 * (254 - BIAS - FRACTION_BITS) - UNIT_EXP + 2 * (FRACTION_BITS + 1)) as usize;
fn decode(bits: u16) -> Result<(bool, u32, i32), String> {
    let e = ((bits >> 7) & 255) as i32;
    let f = (bits & 127) as u32;
    if e == 255 {
        return Err("exact reference requires finite BF16 operands".into());
    }
    Ok((
        bits & 0x8000 != 0,
        if e == 0 { f } else { 128 + f },
        e.max(1) - BIAS - FRACTION_BITS,
    ))
}
fn add(v: &mut [u64], coefficient: u32, shift: usize) -> Result<(), String> {
    let mut at = shift / 64;
    let mut carry = (coefficient as u128) << (shift % 64);
    while carry != 0 {
        let slot = v.get_mut(at).ok_or("exact accumulator extent overflow")?;
        let (next, overflow) = slot.overflowing_add(carry as u64);
        *slot = next;
        carry = (carry >> 64) + u128::from(overflow);
        at += 1;
    }
    Ok(())
}
fn rounded(m: &[u64], shift: usize) -> Result<u16, String> {
    let at = shift / 64;
    let offset = shift % 64;
    let mut q = m.get(at).copied().unwrap_or(0) >> offset;
    if offset != 0 {
        q |= m.get(at + 1).copied().unwrap_or(0) << (64 - offset);
    }
    if q > 256 {
        return Err("exact rounding quotient exceeds BF16 significand".into());
    }
    if shift == 0 {
        return Ok(q as u16);
    }
    let round_at = (shift - 1) / 64;
    let round_bit = (shift - 1) % 64;
    let round = m.get(round_at).copied().unwrap_or(0) & (1u64 << round_bit) != 0;
    let sticky = m.iter().take(round_at).any(|v| *v != 0)
        || (round_bit > 0
            && (m.get(round_at).copied().unwrap_or(0) & ((1u64 << round_bit) - 1)) != 0);
    Ok((q + u64::from(round && (sticky || q & 1 != 0))) as u16)
}
pub(super) fn dot(x: &[u16], w: &[u16]) -> Result<u16, String> {
    if x.is_empty() || x.len() != w.len() {
        return Err("invalid exact dot extent".into());
    }
    let count_bits = (usize::BITS - (x.len() - 1).leading_zeros()) as usize;
    let limbs = PRODUCT_BITS
        .checked_add(count_bits)
        .ok_or("exact accumulator length overflow")?
        .div_ceil(64);
    let mut pos = vec![0u64; limbs];
    let mut neg = vec![0u64; limbs];
    for (&x, &w) in x.iter().zip(w) {
        let (sx, mx, ex) = decode(x)?;
        let (sw, mw, ew) = decode(w)?;
        let shift =
            usize::try_from(ex + ew - UNIT_EXP).map_err(|_| "negative exact product shift")?;
        add(if sx ^ sw { &mut neg } else { &mut pos }, mx * mw, shift)?;
    }
    let cmp = pos.iter().rev().cmp(neg.iter().rev());
    if cmp == std::cmp::Ordering::Equal {
        return Ok(0);
    }
    let negative = cmp == std::cmp::Ordering::Less;
    let (mut mag, sub) = if negative { (neg, pos) } else { (pos, neg) };
    let mut borrow = false;
    for (v, s) in mag.iter_mut().zip(sub) {
        let (t, b1) = v.overflowing_sub(s);
        let (t, b2) = t.overflowing_sub(u64::from(borrow));
        *v = t;
        borrow = b1 || b2;
    }
    if borrow {
        return Err("exact magnitude subtraction underflow".into());
    }
    let top = mag
        .iter()
        .rposition(|v| *v != 0)
        .ok_or("missing nonzero exact magnitude")?;
    let bit = top * 64 + (63 - mag[top].leading_zeros() as usize);
    let mut exponent = i32::try_from(bit).map_err(|_| "exact exponent overflow")? + UNIT_EXP;
    let sign = if negative { 0x8000 } else { 0 };
    if exponent < 1 - BIAS {
        let q = rounded(
            &mag,
            usize::try_from(1 - BIAS - FRACTION_BITS - UNIT_EXP)
                .map_err(|_| "exact subnormal shift")?,
        )?;
        return Ok(sign | q);
    }
    let mut sig = rounded(&mag, bit - FRACTION_BITS as usize)?;
    if sig == 256 {
        sig = 128;
        exponent += 1;
    }
    if exponent > BIAS {
        return Ok(sign | 0x7f80);
    }
    Ok(sign | (((exponent + BIAS) as u16) << 7) | (sig - 128))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_halfway_and_double_rounding_counterexample() {
        let one = 0x3f80;
        let half_bf16_ulp = 0x3b80;
        let tiny = ((127 - 30) << 7) as u16;
        assert_eq!(dot(&[one, half_bf16_ulp], &[one, one]).unwrap(), one);
        assert_eq!(
            dot(&[one, half_bf16_ulp, tiny], &[one, one, one]).unwrap(),
            one + 1
        );
        assert_eq!(
            dot(&[one, half_bf16_ulp, tiny], &[one | 0x8000; 3]).unwrap(),
            (one + 1) | 0x8000
        );
        assert_eq!(dot(&[one, 0x3c40], &[one, one]).unwrap(), one + 2);
    }
    #[test]
    fn exact_subnormal_cancel_and_overflow() {
        assert_eq!(dot(&[1], &[0x3f00]).unwrap(), 0);
        assert_eq!(dot(&[1, 1], &[0x3f00, 0x3580]).unwrap(), 1);
        assert_eq!(dot(&[0x3f80, 0xbf80], &[0x3f80; 2]).unwrap(), 0);
        assert_eq!(dot(&[0x7f7f; 2], &[0x3f80; 2]).unwrap(), 0x7f80);
        assert!(dot(&[0x7f80], &[0x3f80]).is_err());
        assert!(dot(&[], &[]).is_err());
    }
}
