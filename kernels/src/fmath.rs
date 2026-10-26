pub fn sqrt_f32(x: f32) -> f32 {
    if !(x > 0.0) {
        return 0.0;
    }
    let mut guess = f32::from_bits(0x1FBD1DF5 + (x.to_bits() >> 1));
    let mut i = 0;
    while i < 4 {
        guess = 0.5 * (guess + x / guess);
        i += 1;
    }
    guess
}

pub fn floor_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let truncated = x as i32;
    let truncated_f = truncated as f32;
    if x < 0.0 && truncated_f != x {
        (truncated.wrapping_sub(1)) as f32
    } else {
        truncated_f
    }
}

pub fn exp_f32(x: f32) -> f32 {
    if x < -80.0 {
        return 0.0;
    }
    if x > 80.0 {
        return 1.0e35;
    }
    const LN2: f32 = 0.6931471805599453;
    const INV_LN2: f32 = 1.4426950408889634;
    let n = floor_f32(x * INV_LN2 + 0.5);
    let f = x - n * LN2;
    let p = 1.0
        + f * (1.0
            + f * (0.5
                + f * (1.0 / 6.0
                    + f * (1.0 / 24.0
                        + f * (1.0 / 120.0 + f * (1.0 / 720.0 + f * (1.0 / 5040.0 + f / 40320.0)))))));
    let bits = ((n as i32 + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

pub fn ln_f32(x: f32) -> f32 {
    if !(x > 0.0) {
        return -80.0;
    }
    const LN2: f32 = 0.6931471805599453;
    let bits = x.to_bits();
    let mut exp = ((bits >> 23) as i32) - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > 1.41421356 {
        mantissa *= 0.5;
        exp += 1;
    }
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let t2 = t * t;
    let mut acc = 0.0f32;
    let mut power = t;
    let mut k = 1.0f32;
    let mut i = 0;
    while i < 9 {
        acc += power / k;
        power *= t2;
        k += 2.0;
        i += 1;
    }
    2.0 * acc + (exp as f32) * LN2
}

pub fn pow_f32(base: f32, exponent: f32) -> f32 {
    if base <= 0.0 {
        return 0.0;
    }
    if exponent == 0.0 {
        return 1.0;
    }
    exp_f32(ln_f32(base) * exponent)
}

pub fn srgb_to_linear(value: f32) -> f32 {
    let value = clamp01(value);
    if value <= 0.04045 {
        value / 12.92
    } else {
        pow_f32((value + 0.055) / 1.055, 2.4)
    }
}

pub fn linear_to_srgb(value: f32) -> f32 {
    let value = clamp01(value);
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * pow_f32(value, 1.0 / 2.4) - 0.055
    }
}

pub fn clamp01(value: f32) -> f32 {
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

pub fn encode_display(linear: f32, black: f32, inv_span: f32, inv_gamma: f32) -> f32 {
    let mut t = (linear - black) * inv_span;
    t = clamp01(t);
    if inv_gamma != 1.0 {
        t = pow_f32(t, inv_gamma);
    }
    clamp01(linear_to_srgb(t))
}
