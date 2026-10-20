//! Offline numerical-reference conversion. Preserve every F64 sticky bit.
//! This is not model sampling, a runtime precision choice or a fallback.
pub fn bf16_from_f64_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        return sign | 0x7f80 | if fraction == 0 { 0 } else { 0x40 };
    }
    let mantissa = fraction | if exp == 0 { 0 } else { 1u64 << 52 };
    if mantissa == 0 {
        return sign;
    }
    let exponent = if exp == 0 { -1022 } else { exp - 1023 };
    fn rounded_shift(value: u64, shift: u32) -> u64 {
        if shift > 64 {
            return 0;
        }
        if shift == 64 {
            return u64::from(value > 1u64 << 63);
        }
        let quotient = value >> shift;
        let remainder = value & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        quotient + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0))
    }
    if exponent < -126 {
        return sign | rounded_shift(mantissa, (-exponent - 81) as u32) as u16;
    }
    let mut significand = rounded_shift(mantissa, 45);
    let mut biased = exponent + 127;
    if significand == 256 {
        significand = 128;
        biased += 1;
    }
    if biased >= 255 {
        return sign | 0x7f80;
    }
    sign | ((biased as u16) << 7) | (significand as u16 - 128)
}
