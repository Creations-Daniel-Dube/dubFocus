use dubfocus_kernels::fmath::{linear_to_srgb, srgb_to_linear};

pub fn to_linear(samples: &[f32], treat_srgb: bool) -> Vec<f32> {
    let mut values: Vec<f32> = if treat_srgb {
        samples.iter().copied().map(|value| srgb_to_linear(value.clamp(0.0, 1.0))).collect()
    } else {
        samples.iter().copied().map(|value| if value < 0.0 { 0.0 } else { value }).collect()
    };
    let max = values.iter().copied().fold(0.0f32, f32::max);
    if max > 1.01 {
        let scale = 1.0 / max;
        for value in &mut values {
            *value *= scale;
        }
    }
    values
}

pub fn luminance(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let pixels = samples.len() / channels;
    let mut luma = Vec::with_capacity(pixels);
    for pixel in 0..pixels {
        let offset = pixel * channels;
        let red = samples[offset];
        let green = samples[offset + 1];
        let blue = samples[offset + 2];
        luma.push(0.2126 * red + 0.7152 * green + 0.0722 * blue);
    }
    luma
}

pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

pub fn level_linear(value: f32, black: f32, white: f32, gamma: f32) -> f32 {
    let span = (white - black).max(1.0e-4);
    let t = ((value - black) / span).clamp(0.0, 1.0);
    if (gamma - 1.0).abs() < 1.0e-4 {
        t
    } else {
        t.powf(1.0 / gamma)
    }
}

pub fn encode_u8(value: f32) -> u8 {
    let scaled = linear_to_srgb(value) * 255.0 + 0.5;
    scaled.clamp(0.0, 255.0) as u8
}

pub fn encode_u16(value: f32) -> u16 {
    let scaled = linear_to_srgb(value) * 65535.0 + 0.5;
    scaled.clamp(0.0, 65535.0) as u16
}

pub fn downscale(samples: &[f32], width: usize, height: usize, channels: usize, target_w: usize, target_h: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; target_w * target_h * channels];
    for y in 0..target_h {
        let y0 = y * height / target_h;
        let y1 = ((y + 1) * height / target_h).max(y0 + 1).min(height);
        for x in 0..target_w {
            let x0 = x * width / target_w;
            let x1 = ((x + 1) * width / target_w).max(x0 + 1).min(width);
            let mut acc = [0.0f32; 4];
            let mut count = 0.0f32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let pixel = (sy * width + sx) * channels;
                    for channel in 0..channels {
                        acc[channel] += samples[pixel + channel];
                    }
                    count += 1.0;
                }
            }
            let pixel = (y * target_w + x) * channels;
            for channel in 0..channels {
                dst[pixel + channel] = acc[channel] / count;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip_stays_close() {
        for step in 0..32 {
            let encoded = step as f32 / 31.0;
            let linear = srgb_to_linear(encoded);
            let back = dubfocus_kernels::fmath::linear_to_srgb(linear);
            assert!((back - encoded).abs() < 2.0e-3, "{encoded} -> {back}");
        }
    }

    #[test]
    fn rec709_luma_of_white_is_one() {
        let y = luminance(&[1.0, 1.0, 1.0], 3);
        assert!((y[0] - 1.0).abs() < 1.0e-5);
    }
}
