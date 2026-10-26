#![allow(dead_code)]

use std::f32::consts::PI;

use crate::psf::{psf_taps, PsfKind};

pub fn contrast(image: &[f32], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    let mut count = 0.0f32;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = image[y * width + x];
            acc += (image[y * width + x + 1] - center).abs();
            count += 1.0;
        }
    }
    acc / count
}

pub fn blur(image: &[f32], width: usize, height: usize, kind: PsfKind, radius: f32) -> Vec<f32> {
    convolve(image, width, height, &psf_taps(kind, radius))
}

pub fn rl_once(object: &[f32], observed: &[f32], width: usize, height: usize, kind: PsfKind, radius: f32) -> Vec<f32> {
    let taps = psf_taps(kind, radius);
    let reblur = convolve(object, width, height, &taps);
    let ratio: Vec<f32> = observed
        .iter()
        .zip(&reblur)
        .map(|(value, blurred)| value / (blurred + 1.0e-6))
        .collect();
    let correction = convolve(&ratio, width, height, &taps);
    object
        .iter()
        .zip(&correction)
        .map(|(value, factor)| (value * factor).clamp(0.0, 16.0))
        .collect()
}

pub fn wiener(image: &[f32], width: usize, height: usize, kind: PsfKind, radius: f32, k: f32) -> Vec<f32> {
    let signal = dft(image, width, height, false);
    let mut psf = vec![0.0f32; width * height];
    for tap in psf_taps(kind, radius) {
        let x = tap.x.rem_euclid(width as i32) as usize;
        let y = tap.y.rem_euclid(height as i32) as usize;
        psf[y * width + x] += tap.weight;
    }
    let kernel = dft(&psf, width, height, false);
    let mut filtered = Vec::with_capacity(signal.len());
    for (bin, h) in signal.iter().zip(&kernel) {
        let denom = h.re * h.re + h.im * h.im + k;
        let scale_re = h.re / denom;
        let scale_im = -h.im / denom;
        filtered.push(Complex {
            re: bin.re * scale_re - bin.im * scale_im,
            im: bin.re * scale_im + bin.im * scale_re,
        });
    }
    idft(&filtered, width, height)
        .into_iter()
        .map(|value| value.clamp(0.0, 16.0))
        .collect()
}

fn convolve(image: &[f32], width: usize, height: usize, taps: &[crate::psf::Tap]) -> Vec<f32> {
    let mut dst = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for tap in taps {
                let sx = reflect((x as i32 - tap.x) as isize, width);
                let sy = reflect((y as i32 - tap.y) as isize, height);
                acc += tap.weight * image[sy * width + sx];
            }
            dst[y * width + x] = acc;
        }
    }
    dst
}

fn reflect(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = length as isize * 2;
    let mut index = index % period;
    if index < 0 {
        index += period;
    }
    if index >= length as isize {
        index = period - index - 1;
    }
    index as usize
}

struct Complex {
    re: f32,
    im: f32,
}

fn dft(src: &[f32], width: usize, height: usize, inverse: bool) -> Vec<Complex> {
    let mut out = Vec::with_capacity(width * height);
    let sign = if inverse { 1.0 } else { -1.0 };
    let norm = if inverse { 1.0 / (width * height) as f32 } else { 1.0 };
    for oy in 0..height {
        for ox in 0..width {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for iy in 0..height {
                for ix in 0..width {
                    let angle = sign * 2.0 * PI * ((ox * ix) as f32 / width as f32 + (oy * iy) as f32 / height as f32);
                    let (sin, cos) = angle.sin_cos();
                    let value = src[iy * width + ix];
                    re += value * cos;
                    im += value * sin;
                }
            }
            out.push(Complex { re: re * norm, im: im * norm });
        }
    }
    out
}

fn idft(src: &[Complex], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    let norm = 1.0 / (width * height) as f32;
    for iy in 0..height {
        for ix in 0..width {
            let mut re = 0.0f32;
            for oy in 0..height {
                for ox in 0..width {
                    let bin = &src[oy * width + ox];
                    let angle = 2.0 * PI * ((ox * ix) as f32 / width as f32 + (oy * iy) as f32 / height as f32);
                    let (sin, cos) = angle.sin_cos();
                    re += bin.re * cos - bin.im * sin;
                }
            }
            out[iy * width + ix] = re * norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(width: usize, height: usize) -> Vec<f32> {
        let mut image = vec![0.15f32; width * height];
        for y in 0..height {
            for x in width / 2..width {
                image[y * width + x] = 0.85;
            }
        }
        image
    }

    #[test]
    fn one_rl_iteration_raises_the_edge() {
        let sharp = step(24, 16);
        let blurred = blur(&sharp, 24, 16, PsfKind::Gaussian, 1.1);
        let before = contrast(&blurred, 24, 16);
        let once = rl_once(&blurred, &blurred, 24, 16, PsfKind::Gaussian, 1.1);
        assert!(contrast(&once, 24, 16) > before);
    }

    #[test]
    fn wiener_raises_the_edge() {
        let sharp = step(16, 12);
        let blurred = blur(&sharp, 16, 12, PsfKind::Gaussian, 1.0);
        let before = contrast(&blurred, 16, 12);
        let restored = wiener(&blurred, 16, 12, PsfKind::Gaussian, 1.0, 1.0e-3);
        assert!(contrast(&restored, 16, 12) > before * 1.05, "{before} -> {}", contrast(&restored, 16, 12));
    }
}
