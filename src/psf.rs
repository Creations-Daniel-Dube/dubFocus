#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsfKind {
    Disk,
    Gaussian,
}

#[derive(Clone, Copy, Debug)]
pub struct Tap {
    pub x: i32,
    pub y: i32,
    pub weight: f32,
}

pub fn support_radius(kind: PsfKind, radius: f32) -> i32 {
    let radius = radius.max(0.2);
    let reach = match kind {
        PsfKind::Disk => radius + 1.0,
        PsfKind::Gaussian => 4.0 * radius + 1.0,
    };
    (reach.ceil() as i32).max(1)
}

pub fn good_fft_size(min: i32) -> i32 {
    let mut n = min.max(2);
    loop {
        let mut value = n;
        for factor in [2, 3, 5] {
            while value % factor == 0 {
                value /= factor;
            }
        }
        if value == 1 {
            return n;
        }
        n += 1;
    }
}

pub fn geometry(width: i32, height: i32, kind: PsfKind, radius: f32) -> (i32, i32, i32, i32) {
    let pad = support_radius(kind, radius);
    let fft_w = good_fft_size(width + pad * 2);
    let fft_h = good_fft_size(height + pad * 2);
    (fft_w, fft_h, pad, pad)
}

pub fn psf_taps(kind: PsfKind, radius: f32) -> Vec<Tap> {
    let radius = radius.max(0.2);
    let reach = support_radius(kind, radius);
    let mut taps = Vec::new();
    let mut sum = 0.0f32;
    for y in -reach..=reach {
        for x in -reach..=reach {
            let weight = match kind {
                PsfKind::Disk => disk_weight(x as f32, y as f32, radius),
                PsfKind::Gaussian => gaussian_weight(x as f32, y as f32, radius),
            };
            if weight > 0.0 {
                taps.push(Tap { x, y, weight });
                sum += weight;
            }
        }
    }
    if sum > 0.0 {
        for tap in &mut taps {
            tap.weight /= sum;
        }
    }
    taps
}

pub fn wrapped_lists(taps: &[Tap], fft_w: i32, fft_h: i32) -> (Vec<i32>, Vec<f32>) {
    let mut index = Vec::with_capacity(taps.len());
    let mut weight = Vec::with_capacity(taps.len());
    for tap in taps {
        let x = tap.x.rem_euclid(fft_w);
        let y = tap.y.rem_euclid(fft_h);
        index.push(y * fft_w + x);
        weight.push(tap.weight);
    }
    (index, weight)
}

fn disk_weight(x: f32, y: f32, radius: f32) -> f32 {
    let distance = (x * x + y * y).sqrt();
    let t = (radius + 0.5 - distance) / 0.5;
    t.clamp(0.0, 1.0)
}

fn gaussian_weight(x: f32, y: f32, sigma: f32) -> f32 {
    let sigma = sigma.max(0.2);
    (-0.5 * (x * x + y * y) / (sigma * sigma)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_are_normalized() {
        for kind in [PsfKind::Disk, PsfKind::Gaussian] {
            let sum: f32 = psf_taps(kind, 1.4).iter().map(|tap| tap.weight).sum();
            assert!((sum - 1.0).abs() < 1.0e-4, "{sum}");
        }
    }

    #[test]
    fn fft_size_uses_small_factors() {
        let n = good_fft_size(1584 + 8);
        assert!(n >= 1592);
        let mut value = n;
        for factor in [2, 3, 5] {
            while value % factor == 0 {
                value /= factor;
            }
        }
        assert_eq!(value, 1);
    }
}
