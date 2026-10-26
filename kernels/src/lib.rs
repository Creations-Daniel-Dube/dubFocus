#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, stdarch_nvptx))]

pub mod fmath;
pub mod geom;

#[cfg(target_arch = "nvptx64")]
mod device {
    use crate::fmath::{encode_display, exp_f32, floor_f32, sqrt_f32};
    use crate::geom::{clamp_index, reflect_index};
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};

    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    struct C32 {
        x: f32,
        y: f32,
    }

    fn global_x() -> u32 {
        let thread = unsafe { _thread_idx_x() };
        let block = unsafe { _block_idx_x() };
        let width = unsafe { _block_dim_x() };
        block.wrapping_mul(width).wrapping_add(thread)
    }

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    unsafe fn load_f(ptr: *const f32, index: i32) -> f32 {
        unsafe { *ptr.add(index as usize) }
    }

    unsafe fn store_f(ptr: *mut f32, index: i32, value: f32) {
        unsafe { *ptr.add(index as usize) = value }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn copy_f32(src: *const f32, dst: *mut f32, n: i32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        unsafe { store_f(dst, gid, load_f(src, gid)) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn scale_relu(dst: *mut f32, n: i32, scale: f32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let mut value = unsafe { load_f(dst, gid) } * scale;
        if value < 0.0 || value.is_nan() {
            value = 0.0;
        } else if value > 16.0 {
            value = 16.0;
        }
        unsafe { store_f(dst, gid, value) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn scatter_f32(index: *const i32, value: *const f32, dst: *mut f32, n: i32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let at = unsafe { *index.add(gid as usize) };
        let sample = unsafe { load_f(value, gid) };
        unsafe { store_f(dst, at, sample) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn pad_reflect(
        src: *const f32,
        dst: *mut f32,
        src_w: i32,
        src_h: i32,
        dst_w: i32,
        dst_h: i32,
        origin_x: i32,
        origin_y: i32,
    ) {
        let gid = global_x() as i32;
        let n = dst_w.wrapping_mul(dst_h);
        if gid >= n || dst_w <= 0 {
            return;
        }
        let y = gid / dst_w;
        let x = gid - y * dst_w;
        let sx = reflect_index(x - origin_x, src_w);
        let sy = reflect_index(y - origin_y, src_h);
        unsafe { store_f(dst, gid, load_f(src, sy * src_w + sx)) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn crop_f32(
        src: *const f32,
        dst: *mut f32,
        src_w: i32,
        origin_x: i32,
        origin_y: i32,
        dst_w: i32,
        dst_h: i32,
    ) {
        let gid = global_x() as i32;
        let n = dst_w.wrapping_mul(dst_h);
        if gid >= n || dst_w <= 0 {
            return;
        }
        let y = gid / dst_w;
        let x = gid - y * dst_w;
        unsafe { store_f(dst, gid, load_f(src, (y + origin_y) * src_w + (x + origin_x))) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn wiener_divide(signal: *const C32, psf: *const C32, dst: *mut C32, n: i32, k: f32, max_gain: f32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let y = unsafe { *signal.add(gid as usize) };
        let h = unsafe { *psf.add(gid as usize) };
        let power = h.x * h.x + h.y * h.y;
        let mut denom = power + k;
        let mag = sqrt_f32(power);
        let floor = mag / max_gain.max(1.0);
        if denom < floor {
            denom = floor;
        }
        if denom < 1.0e-12 {
            denom = 1.0e-12;
        }
        let scale_re = h.x / denom;
        let scale_im = -h.y / denom;
        unsafe {
            *dst.add(gid as usize) = C32 {
                x: y.x * scale_re - y.y * scale_im,
                y: y.x * scale_im + y.y * scale_re,
            };
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn mul_complex(a: *const C32, b: *const C32, dst: *mut C32, n: i32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let left = unsafe { *a.add(gid as usize) };
        let right = unsafe { *b.add(gid as usize) };
        unsafe {
            *dst.add(gid as usize) = C32 {
                x: left.x * right.x - left.y * right.y,
                y: left.x * right.y + left.y * right.x,
            };
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn mul_conj(a: *const C32, b: *const C32, dst: *mut C32, n: i32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let left = unsafe { *a.add(gid as usize) };
        let right = unsafe { *b.add(gid as usize) };
        unsafe {
            *dst.add(gid as usize) = C32 {
                x: left.x * right.x + left.y * right.y,
                y: left.y * right.x - left.x * right.y,
            };
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn divide_eps(num: *const f32, den: *const f32, dst: *mut f32, n: i32, eps: f32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        unsafe { store_f(dst, gid, load_f(num, gid) / (load_f(den, gid) + eps)) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn rl_damped(
        obj: *mut f32,
        observed: *const f32,
        reblur: *const f32,
        correction: *const f32,
        n: i32,
        sigma: f32,
    ) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let current = unsafe { load_f(obj, gid) };
        let mut factor = unsafe { load_f(correction, gid) };
        if factor.is_nan() || factor < 0.0 {
            factor = 0.0;
        } else if factor > 8.0 {
            factor = 8.0;
        }
        let mut weight = 1.0f32;
        if sigma > 1.0e-8 {
            let residual = unsafe { load_f(observed, gid) - load_f(reblur, gid) };
            let u = residual / sigma;
            weight = 1.0 - exp_f32(-0.5 * u * u);
        }
        let mut next = current * (1.0 + (factor - 1.0) * weight);
        if next < 0.0 || next.is_nan() {
            next = 0.0;
        } else if next > 16.0 {
            next = 16.0;
        }
        unsafe { store_f(obj, gid, next) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn laplacian_reg(src: *const f32, dst: *mut f32, width: i32, height: i32, lambda: f32) {
        let gid = global_x() as i32;
        let n = width.wrapping_mul(height);
        if gid >= n || width <= 0 {
            return;
        }
        let y = gid / width;
        let x = gid - y * width;
        let mut acc = 0.0f32;
        let mut oy = -1i32;
        while oy <= 1 {
            let mut ox = -1i32;
            let sy = clamp_index(y + oy, height);
            while ox <= 1 {
                let sx = clamp_index(x + ox, width);
                acc += unsafe { load_f(src, sy * width + sx) };
                ox += 1;
            }
            oy += 1;
        }
        let blur = acc / 9.0;
        let value = unsafe { load_f(src, gid) };
        let mut denom = 1.0 + lambda * (value - blur) / (value + 1.0e-6);
        if denom < 0.05 || denom.is_nan() {
            denom = 0.05;
        }
        let mut next = value / denom;
        if next < 0.0 || next.is_nan() {
            next = 0.0;
        } else if next > 16.0 {
            next = 16.0;
        }
        unsafe { store_f(dst, gid, next) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn edge_mask(src: *const f32, dst: *mut f32, width: i32, height: i32) {
        let gid = global_x() as i32;
        let n = width.wrapping_mul(height);
        if gid >= n || width <= 0 {
            return;
        }
        let y = gid / width;
        let x = gid - y * width;
        let sample = |ox: i32, oy: i32| -> f32 {
            let sx = clamp_index(x + ox, width);
            let sy = clamp_index(y + oy, height);
            unsafe { load_f(src, sy * width + sx) }
        };
        let a = sample(-1, -1);
        let b = sample(0, -1);
        let c = sample(1, -1);
        let d = sample(-1, 0);
        let f = sample(1, 0);
        let g = sample(-1, 1);
        let h = sample(0, 1);
        let i = sample(1, 1);
        let gx = -a + c - 2.0 * d + 2.0 * f - g + i;
        let gy = -a - 2.0 * b - c + g + 2.0 * h + i;
        let mag = sqrt_f32(gx * gx + gy * gy);
        let edge0 = 0.02f32;
        let edge1 = 0.25f32;
        let mut t = (mag - edge0) / (edge1 - edge0);
        if t < 0.0 {
            t = 0.0;
        } else if t > 1.0 {
            t = 1.0;
        }
        unsafe { store_f(dst, gid, t * t * (3.0 - 2.0 * t)) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn box_blur3(src: *const f32, dst: *mut f32, width: i32, height: i32) {
        let gid = global_x() as i32;
        let n = width.wrapping_mul(height);
        if gid >= n || width <= 0 {
            return;
        }
        let y = gid / width;
        let x = gid - y * width;
        let mut acc = 0.0f32;
        let mut oy = -1i32;
        while oy <= 1 {
            let sy = clamp_index(y + oy, height);
            let mut ox = -1i32;
            while ox <= 1 {
                let sx = clamp_index(x + ox, width);
                acc += unsafe { load_f(src, sy * width + sx) };
                ox += 1;
            }
            oy += 1;
        }
        unsafe { store_f(dst, gid, acc / 9.0) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn lerp_mask(
        sharp: *const f32,
        smooth: *const f32,
        mask: *const f32,
        dst: *mut f32,
        n: i32,
        strength: f32,
    ) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let a = unsafe { load_f(sharp, gid) };
        let b = unsafe { load_f(smooth, gid) };
        let mut m = unsafe { load_f(mask, gid) } * strength;
        if m < 0.0 {
            m = 0.0;
        } else if m > 1.0 {
            m = 1.0;
        }
        unsafe { store_f(dst, gid, a + (b - a) * m) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn apply_chroma(
        src: *const f32,
        y_in: *const f32,
        y_out: *const f32,
        dst: *mut f32,
        n: i32,
        channels: i32,
    ) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        let restored = unsafe { load_f(y_out, gid) };
        if channels <= 1 {
            unsafe { store_f(dst, gid, restored) }
            return;
        }
        let base = unsafe { load_f(y_in, gid) };
        let scale = restored / (base + 1.0e-6);
        let pixel = gid * 3;
        let mut c = 0i32;
        while c < 3 {
            let mut value = unsafe { load_f(src, pixel + c) } * scale;
            if value < 0.0 || value.is_nan() {
                value = 0.0;
            }
            unsafe { store_f(dst, pixel + c, value) }
            c += 1;
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn shade_view(
        original: *const f32,
        processed: *const f32,
        rgba: *mut u8,
        img_w: i32,
        img_h: i32,
        channels: i32,
        view_w: i32,
        view_h: i32,
        origin_x: f32,
        origin_y: f32,
        scale: f32,
        wipe_x: f32,
        black: f32,
        inv_span: f32,
        inv_gamma: f32,
    ) {
        let gid = global_x() as i32;
        let n = view_w.wrapping_mul(view_h);
        if gid >= n || view_w <= 0 || img_w <= 0 || img_h <= 0 {
            return;
        }
        let sy = gid / view_w;
        let sx = gid - sy * view_w;
        let fx = origin_x + (sx as f32 + 0.5) * scale;
        let fy = origin_y + (sy as f32 + 0.5) * scale;
        let source = if (sx as f32) < wipe_x { original } else { processed };
        let (r, g, b) = sample(source, img_w, img_h, channels, fx, fy);
        let r = encode_display(r, black, inv_span, inv_gamma);
        let g = encode_display(g, black, inv_span, inv_gamma);
        let b = encode_display(b, black, inv_span, inv_gamma);
        let pixel = (gid as usize) * 4;
        unsafe {
            *rgba.add(pixel) = to_byte(r);
            *rgba.add(pixel + 1) = to_byte(g);
            *rgba.add(pixel + 2) = to_byte(b);
            *rgba.add(pixel + 3) = 255;
        }
    }

    fn to_byte(value: f32) -> u8 {
        let scaled = value * 255.0 + 0.5;
        if scaled <= 0.0 {
            0
        } else if scaled >= 255.0 {
            255
        } else {
            scaled as u8
        }
    }

    fn sample(src: *const f32, width: i32, height: i32, channels: i32, x: f32, y: f32) -> (f32, f32, f32) {
        let x0 = floor_f32(x) as i32;
        let y0 = floor_f32(y) as i32;
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let c00 = pixel_at(src, width, height, channels, x0, y0);
        let c10 = pixel_at(src, width, height, channels, x0 + 1, y0);
        let c01 = pixel_at(src, width, height, channels, x0, y0 + 1);
        let c11 = pixel_at(src, width, height, channels, x0 + 1, y0 + 1);
        let r = lerp(lerp(c00.0, c10.0, tx), lerp(c01.0, c11.0, tx), ty);
        let g = lerp(lerp(c00.1, c10.1, tx), lerp(c01.1, c11.1, tx), ty);
        let b = lerp(lerp(c00.2, c10.2, tx), lerp(c01.2, c11.2, tx), ty);
        (r, g, b)
    }

    fn pixel_at(src: *const f32, width: i32, height: i32, channels: i32, x: i32, y: i32) -> (f32, f32, f32) {
        let x = clamp_index(x, width);
        let y = clamp_index(y, height);
        let index = y * width + x;
        if channels <= 1 {
            let value = unsafe { load_f(src, index) };
            (value, value, value)
        } else {
            let pixel = index * 3;
            (
                unsafe { load_f(src, pixel) },
                unsafe { load_f(src, pixel + 1) },
                unsafe { load_f(src, pixel + 2) },
            )
        }
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}
