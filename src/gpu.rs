use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cudarc::cufft::CudaFft;
use cudarc::cufft::sys::{self as cufft_sys, float2};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::error::Error;
use crate::psf::{geometry, psf_taps, wrapped_lists, PsfKind};

macro_rules! klaunch {
    ($gpu:expr, $count:expr, $func:expr $(, $arg:expr)* $(,)?) => {{
        let count: u32 = $count;
        if count > 0 {
            unsafe {
                $gpu.stream
                    .launch_builder($func)
                    $(.arg($arg))*
                    .launch(LaunchConfig::for_num_elems(count))
                    .map_err(gpu_err)?;
            }
        }
    }};
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub total: usize,
    pub free: usize,
    pub major: i32,
    pub minor: i32,
}

impl GpuInfo {
    pub fn summary(&self) -> String {
        let arch = kernel_arch(self.major, self.minor).unwrap_or("non pris en charge");
        format!(
            "{} · compute {}.{} · {arch} · {:.1} Gio",
            self.name,
            self.major,
            self.minor,
            self.total as f64 / 1024.0 / 1024.0 / 1024.0
        )
    }
}

/// `sm_61` pour la GTX 1060 (6.1 et 6.2), `sm_70` à partir de compute 7.0.
/// La RTX 2060 est en 7.5 et la RTX 3060 en 8.6 : les deux restent sur `sm_70`.
pub fn kernel_arch(major: i32, minor: i32) -> Option<&'static str> {
    if (major, minor) >= (7, 0) {
        Some("sm_70")
    } else if (major, minor) >= (6, 1) {
        Some("sm_61")
    } else {
        None
    }
}

pub fn unsupported_message(major: i32, minor: i32) -> String {
    format!(
        "Carte non prise en charge (compute {major}.{minor}). dubFocus exige une GTX 1060 (6.1) ou une carte plus récente."
    )
}

pub fn probe_document(info: &GpuInfo) -> String {
    let kernels = kernel_arch(info.major, info.minor).unwrap_or("unsupported");
    format!(
        "{{\"ev\":\"probe\",\"name\":\"{}\",\"total\":{},\"free\":{},\"major\":{},\"minor\":{},\"kernels\":\"{kernels}\"}}",
        info.name.replace('"', "'"),
        info.total,
        info.free,
        info.major,
        info.minor
    )
}

fn ptx_for(arch: &str) -> Result<&'static str, Error> {
    let _stamp = env!("DUBFOCUS_PTX_STAMP");
    let _stamp_pascal = env!("DUBFOCUS_PTX_SM61_STAMP");
    match arch {
        "sm_70" => Ok(include_str!(env!("DUBFOCUS_PTX"))),
        "sm_61" => Ok(include_str!(env!("DUBFOCUS_PTX_SM61"))),
        _ => Err(Error::new(format!("module PTX inconnu : {arch}"))),
    }
}

pub fn probe_gpu() -> Result<GpuInfo, String> {
    let ctx = CudaContext::new(0).map_err(|err| err.to_string())?;
    device_info(&ctx)
}

fn device_info(ctx: &CudaContext) -> Result<GpuInfo, String> {
    let name = ctx.name().map_err(|err| err.to_string())?;
    let total = ctx.total_mem().map_err(|err| err.to_string())?;
    let (free, _) = ctx.mem_get_info().map_err(|err| err.to_string())?;
    let (major, minor) = ctx.compute_capability().map_err(|err| err.to_string())?;
    Ok(GpuInfo { name, total, free, major, minor })
}

struct Gpu {
    /// Module réellement chargé : `sm_70` ou `sm_61`.
    #[allow(dead_code)]
    arch: &'static str,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    copy_f32: CudaFunction,
    scale_relu: CudaFunction,
    scatter_f32: CudaFunction,
    pad_reflect: CudaFunction,
    crop_f32: CudaFunction,
    wiener_divide: CudaFunction,
    mul_complex: CudaFunction,
    mul_conj: CudaFunction,
    divide_eps: CudaFunction,
    rl_damped: CudaFunction,
    laplacian_reg: CudaFunction,
    edge_mask: CudaFunction,
    box_blur3: CudaFunction,
    lerp_mask: CudaFunction,
    apply_chroma: CudaFunction,
    shade_view: CudaFunction,
}

impl Gpu {
    fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gpu_err)?;
        let (major, minor) = ctx.compute_capability().map_err(gpu_err)?;
        let arch = kernel_arch(major, minor).ok_or_else(|| Error::new(unsupported_message(major, minor)))?;
        Self::load(ctx, arch)
    }

    /// Charge un module précis. La compute capability est lue avant `load_module`.
    /// Une carte récente peut exécuter le PTX `sm_61` ; le démarrage normal ne le fait pas.
    #[cfg(test)]
    fn load_arch(arch: &'static str) -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gpu_err)?;
        let (major, minor) = ctx.compute_capability().map_err(gpu_err)?;
        if (major, minor) < (6, 1) {
            return Err(Error::new(unsupported_message(major, minor)));
        }
        Self::load(ctx, arch)
    }

    fn load(ctx: Arc<CudaContext>, arch: &'static str) -> Result<Self, Error> {
        let ptx = ptx_for(arch)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(ptx)).map_err(|err| module_err(err, arch))?;
        let load = |name: &str| module.load_function(name).map_err(|err| Error::new(format!("noyau {name} : {err}")));
        Ok(Self {
            arch,
            stream,
            copy_f32: load("copy_f32")?,
            scale_relu: load("scale_relu")?,
            scatter_f32: load("scatter_f32")?,
            pad_reflect: load("pad_reflect")?,
            crop_f32: load("crop_f32")?,
            wiener_divide: load("wiener_divide")?,
            mul_complex: load("mul_complex")?,
            mul_conj: load("mul_conj")?,
            divide_eps: load("divide_eps")?,
            rl_damped: load("rl_damped")?,
            laplacian_reg: load("laplacian_reg")?,
            edge_mask: load("edge_mask")?,
            box_blur3: load("box_blur3")?,
            lerp_mask: load("lerp_mask")?,
            apply_chroma: load("apply_chroma")?,
            shade_view: load("shade_view")?,
            _module: module,
        })
    }
}

struct Picture {
    width: i32,
    height: i32,
    channels: i32,
    pixels: usize,
    original: CudaSlice<f32>,
    luma: CudaSlice<f32>,
    y_in: CudaSlice<f32>,
    y_out: CudaSlice<f32>,
    processed: CudaSlice<f32>,
    mean: f32,
}

struct Spectrum {
    fft_w: i32,
    fft_h: i32,
    origin_x: i32,
    origin_y: i32,
    real_n: usize,
    spec_n: usize,
    observed: CudaSlice<f32>,
    obj: CudaSlice<f32>,
    wiener: CudaSlice<f32>,
    reblur: CudaSlice<f32>,
    ratio: CudaSlice<f32>,
    correction: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    mask: CudaSlice<f32>,
    psf: CudaSlice<f32>,
    y_fft: CudaSlice<float2>,
    h_fft: CudaSlice<float2>,
    spec_a: CudaSlice<float2>,
    spec_b: CudaSlice<float2>,
    r2c: CudaFft,
    c2r: CudaFft,
    y_ready: bool,
    mask_ready: bool,
}

pub struct ViewRequest {
    pub view_w: i32,
    pub view_h: i32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub scale: f32,
    pub wipe_x: f32,
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
}

pub struct Session {
    gpu: Gpu,
    picture: Option<Picture>,
    spectrum: Option<Spectrum>,
    shade: Option<CudaSlice<u8>>,
    have_wiener: bool,
    have_rl: bool,
    dering: f32,
}

impl Session {
    pub fn new() -> Result<Self, Error> {
        Self::create(Gpu::new()?)
    }

    #[cfg(test)]
    pub fn arch(&self) -> &'static str {
        self.gpu.arch
    }

    #[cfg(test)]
    pub fn new_with_arch(arch: &'static str) -> Result<Self, Error> {
        Self::create(Gpu::load_arch(arch)?)
    }

    fn create(gpu: Gpu) -> Result<Self, Error> {
        Ok(Self {
            gpu,
            picture: None,
            spectrum: None,
            shade: None,
            have_wiener: false,
            have_rl: false,
            dering: 0.4,
        })
    }

    pub fn load(&mut self, interleaved: &[f32], luma: &[f32], width: i32, height: i32, channels: i32, mean: f32) -> Result<(), Error> {
        if width <= 0 || height <= 0 || width > 20_000 || height > 20_000 {
            return Err(Error::new("image trop grande ou vide"));
        }
        let pixels = width as usize * height as usize;
        if luma.len() != pixels || interleaved.len() != pixels * channels as usize {
            return Err(Error::new("tampon image incohérent"));
        }
        let stream = &self.gpu.stream;
        let original = stream.clone_htod(interleaved).map_err(gpu_err)?;
        let luma_buf = stream.clone_htod(luma).map_err(gpu_err)?;
        let y_in = stream.alloc_zeros::<f32>(pixels).map_err(gpu_err)?;
        let y_out = stream.alloc_zeros::<f32>(pixels).map_err(gpu_err)?;
        let processed = stream.alloc_zeros::<f32>(pixels * channels as usize).map_err(gpu_err)?;
        self.picture = Some(Picture {
            width,
            height,
            channels,
            pixels,
            original,
            luma: luma_buf,
            y_in,
            y_out,
            processed,
            mean,
        });
        self.spectrum = None;
        self.have_wiener = false;
        self.have_rl = false;
        Ok(())
    }

    pub fn wiener(&mut self, kind: PsfKind, radius: f32, k: f32) -> Result<(), Error> {
        self.prepare(kind, radius)?;
        self.ensure_signal()?;
        self.install_psf(kind, radius)?;
        {
            let gpu = &self.gpu;
            let spectrum = self.spectrum.as_mut().unwrap();
            let n = spectrum.spec_n as u32;
            let k = k.max(1.0e-6);
            let max_gain = 2.0f32;
            klaunch!(
                gpu,
                n,
                &gpu.wiener_divide,
                &spectrum.y_fft,
                &spectrum.h_fft,
                &mut spectrum.spec_a,
                &(spectrum.spec_n as i32),
                &k,
                &max_gain,
            );
            gpu.c2r_into(spectrum)?;
            let scale = 1.0 / (spectrum.fft_w as f32 * spectrum.fft_h as f32);
            let real_n = spectrum.real_n as u32;
            klaunch!(gpu, real_n, &gpu.scale_relu, &mut spectrum.wiener, &(spectrum.real_n as i32), &scale);
        }
        self.have_wiener = true;
        self.have_rl = false;
        self.publish_wiener()?;
        self.gpu.stream.synchronize().map_err(gpu_err)?;
        Ok(())
    }

    pub fn refine(
        &mut self,
        iterations: u32,
        k: f32,
        damping: f32,
        dering: f32,
        cancel: &AtomicBool,
        mut progress: impl FnMut(u32, u32),
    ) -> Result<bool, Error> {
        if !self.have_wiener {
            return Err(Error::new("aperçu Wiener absent"));
        }
        self.dering = dering.clamp(0.0, 1.0);
        let iterations = iterations.clamp(1, 80);
        {
            let gpu = &self.gpu;
            let spectrum = self.spectrum.as_mut().unwrap();
            let n = spectrum.real_n as u32;
            klaunch!(gpu, n, &gpu.copy_f32, &spectrum.observed, &mut spectrum.obj, &(spectrum.real_n as i32));
        }
        let sigma = if damping <= 0.0 { 0.0 } else { damping * self.picture.as_ref().unwrap().mean.max(1.0e-4) };
        let lambda = k.max(0.0);
        for step in 1..=iterations {
            if cancel.load(Ordering::Relaxed) {
                self.gpu.stream.synchronize().map_err(gpu_err)?;
                return Ok(false);
            }
            self.rl_step(sigma, lambda)?;
            progress(step, iterations);
        }
        self.have_rl = true;
        self.publish_rl()?;
        self.gpu.stream.synchronize().map_err(gpu_err)?;
        Ok(true)
    }

    pub fn set_dering(&mut self, dering: f32) -> Result<(), Error> {
        self.dering = dering.clamp(0.0, 1.0);
        if self.have_rl {
            self.publish_rl()?;
            self.gpu.stream.synchronize().map_err(gpu_err)?;
        }
        Ok(())
    }

    pub fn shade(&mut self, view: &ViewRequest) -> Result<Vec<u8>, Error> {
        if self.picture.is_none() {
            return Err(Error::new("aucune image"));
        }
        if view.view_w <= 0 || view.view_h <= 0 {
            return Err(Error::new("vue vide"));
        }
        let pixels_n = view.view_w as usize * view.view_h as usize;
        let bytes_n = pixels_n.saturating_mul(4);
        let needs = self.shade.as_ref().map(|buffer| buffer.len()) != Some(bytes_n);
        if needs {
            self.shade = Some(self.gpu.stream.alloc_zeros::<u8>(bytes_n).map_err(gpu_err)?);
        }
        let span = (view.white - view.black).max(1.0e-4);
        let inv_span = 1.0 / span;
        let inv_gamma = 1.0 / view.gamma.max(0.05);
        let gpu = &self.gpu;
        let picture = self.picture.as_ref().unwrap();
        let shade = self.shade.as_mut().unwrap();
        klaunch!(
            gpu,
            pixels_n as u32,
            &gpu.shade_view,
            &picture.original,
            &picture.processed,
            &mut *shade,
            &picture.width,
            &picture.height,
            &picture.channels,
            &view.view_w,
            &view.view_h,
            &view.origin_x,
            &view.origin_y,
            &view.scale,
            &view.wipe_x,
            &view.black,
            &inv_span,
            &inv_gamma,
        );
        let pixels = gpu.stream.clone_dtoh(shade).map_err(gpu_err)?;
        gpu.stream.synchronize().map_err(gpu_err)?;
        Ok(pixels)
    }

    pub fn download(&mut self) -> Result<Vec<f32>, Error> {
        let picture = self.picture.as_ref().ok_or_else(|| Error::new("aucune image"))?;
        let values = self.gpu.stream.clone_dtoh(&picture.processed).map_err(gpu_err)?;
        self.gpu.stream.synchronize().map_err(gpu_err)?;
        Ok(values)
    }

    #[cfg(test)]
    pub fn download_luma(&mut self) -> Result<Vec<f32>, Error> {
        let picture = self.picture.as_ref().ok_or_else(|| Error::new("aucune image"))?;
        let values = self.gpu.stream.clone_dtoh(&picture.y_out).map_err(gpu_err)?;
        self.gpu.stream.synchronize().map_err(gpu_err)?;
        Ok(values)
    }

    fn prepare(&mut self, kind: PsfKind, radius: f32) -> Result<(), Error> {
        let (width, height, pixels, fft_w, fft_h, origin_x, origin_y) = {
            let picture = self.picture.as_ref().ok_or_else(|| Error::new("aucune image"))?;
            let (fft_w, fft_h, origin_x, origin_y) = geometry(picture.width, picture.height, kind, radius);
            (picture.width, picture.height, picture.pixels, fft_w, fft_h, origin_x, origin_y)
        };
        let same = self.spectrum.as_ref().is_some_and(|spectrum| {
            spectrum.fft_w == fft_w && spectrum.fft_h == fft_h && spectrum.origin_x == origin_x && spectrum.origin_y == origin_y
        });
        if same {
            return Ok(());
        }
        let real_n = fft_w as usize * fft_h as usize;
        let spec_n = (fft_w as usize / 2 + 1) * fft_h as usize;
        if real_n > 1_000_000_000 {
            return Err(Error::new("FFT trop grande pour la mémoire"));
        }
        let stream = &self.gpu.stream;
        let mut spectrum = Spectrum {
            fft_w,
            fft_h,
            origin_x,
            origin_y,
            real_n,
            spec_n,
            observed: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            obj: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            wiener: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            reblur: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            ratio: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            correction: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            tmp: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            mask: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            psf: stream.alloc_zeros(real_n).map_err(gpu_err)?,
            y_fft: stream.alloc_zeros(spec_n).map_err(gpu_err)?,
            h_fft: stream.alloc_zeros(spec_n).map_err(gpu_err)?,
            spec_a: stream.alloc_zeros(spec_n).map_err(gpu_err)?,
            spec_b: stream.alloc_zeros(spec_n).map_err(gpu_err)?,
            r2c: CudaFft::plan_2d(fft_w, fft_h, cufft_sys::cufftType::CUFFT_R2C, stream.clone()).map_err(gpu_err)?,
            c2r: CudaFft::plan_2d(fft_w, fft_h, cufft_sys::cufftType::CUFFT_C2R, stream.clone()).map_err(gpu_err)?,
            y_ready: false,
            mask_ready: false,
        };
        {
            let gpu = &self.gpu;
            let luma = &self.picture.as_ref().unwrap().luma;
            let n = real_n as u32;
            klaunch!(
                gpu,
                n,
                &gpu.pad_reflect,
                luma,
                &mut spectrum.observed,
                &width,
                &height,
                &spectrum.fft_w,
                &spectrum.fft_h,
                &spectrum.origin_x,
                &spectrum.origin_y,
            );
        }
        {
            let gpu = &self.gpu;
            let y_in = &mut self.picture.as_mut().unwrap().y_in;
            klaunch!(
                gpu,
                pixels as u32,
                &gpu.crop_f32,
                &spectrum.observed,
                y_in,
                &spectrum.fft_w,
                &spectrum.origin_x,
                &spectrum.origin_y,
                &width,
                &height,
            );
        }
        self.spectrum = Some(spectrum);
        self.have_wiener = false;
        self.have_rl = false;
        Ok(())
    }

    fn ensure_signal(&mut self) -> Result<(), Error> {
        let ready = self.spectrum.as_ref().is_some_and(|spectrum| spectrum.y_ready);
        if ready {
            return Ok(());
        }
        let spectrum = self.spectrum.as_mut().unwrap();
        spectrum.r2c.exec_r2c(&spectrum.observed, &mut spectrum.y_fft).map_err(gpu_err)?;
        spectrum.y_ready = true;
        Ok(())
    }

    fn install_psf(&mut self, kind: PsfKind, radius: f32) -> Result<(), Error> {
        let (fft_w, fft_h) = {
            let spectrum = self.spectrum.as_ref().unwrap();
            (spectrum.fft_w, spectrum.fft_h)
        };
        let taps = psf_taps(kind, radius);
        let (index, weight) = wrapped_lists(&taps, fft_w, fft_h);
        let index_buf = self.gpu.stream.clone_htod(&index).map_err(gpu_err)?;
        let weight_buf = self.gpu.stream.clone_htod(&weight).map_err(gpu_err)?;
        {
            let gpu = &self.gpu;
            let spectrum = self.spectrum.as_mut().unwrap();
            gpu.stream.memset_zeros(&mut spectrum.psf).map_err(gpu_err)?;
            let n = index.len() as i32;
            klaunch!(gpu, n as u32, &gpu.scatter_f32, &index_buf, &weight_buf, &mut spectrum.psf, &n);
            spectrum.r2c.exec_r2c(&spectrum.psf, &mut spectrum.h_fft).map_err(gpu_err)?;
            gpu.stream.synchronize().map_err(gpu_err)?;
        }
        Ok(())
    }

    fn rl_step(&mut self, sigma: f32, lambda: f32) -> Result<(), Error> {
        let gpu = &self.gpu;
        let spectrum = self.spectrum.as_mut().unwrap();
        let real_n = spectrum.real_n as i32;
        let spec_n = spectrum.spec_n as i32;
        let scale = 1.0 / (spectrum.fft_w as f32 * spectrum.fft_h as f32);
        spectrum.r2c.exec_r2c(&spectrum.obj, &mut spectrum.spec_a).map_err(gpu_err)?;
        klaunch!(gpu, spec_n as u32, &gpu.mul_complex, &spectrum.spec_a, &spectrum.h_fft, &mut spectrum.spec_b, &spec_n);
        spectrum.c2r.exec_c2r(&mut spectrum.spec_b, &mut spectrum.reblur).map_err(gpu_err)?;
        klaunch!(gpu, real_n as u32, &gpu.scale_relu, &mut spectrum.reblur, &real_n, &scale);
        let eps = 1.0e-6f32;
        klaunch!(gpu, real_n as u32, &gpu.divide_eps, &spectrum.observed, &spectrum.reblur, &mut spectrum.ratio, &real_n, &eps);
        spectrum.r2c.exec_r2c(&spectrum.ratio, &mut spectrum.spec_a).map_err(gpu_err)?;
        klaunch!(gpu, spec_n as u32, &gpu.mul_conj, &spectrum.spec_a, &spectrum.h_fft, &mut spectrum.spec_b, &spec_n);
        spectrum.c2r.exec_c2r(&mut spectrum.spec_b, &mut spectrum.correction).map_err(gpu_err)?;
        klaunch!(gpu, real_n as u32, &gpu.scale_relu, &mut spectrum.correction, &real_n, &scale);
        klaunch!(
            gpu,
            real_n as u32,
            &gpu.rl_damped,
            &mut spectrum.obj,
            &spectrum.observed,
            &spectrum.reblur,
            &spectrum.correction,
            &real_n,
            &sigma,
        );
        klaunch!(gpu, real_n as u32, &gpu.laplacian_reg, &spectrum.obj, &mut spectrum.tmp, &spectrum.fft_w, &spectrum.fft_h, &lambda);
        klaunch!(gpu, real_n as u32, &gpu.copy_f32, &spectrum.tmp, &mut spectrum.obj, &real_n);
        Ok(())
    }

    fn publish_wiener(&mut self) -> Result<(), Error> {
        self.crop_from_wiener()?;
        self.recolor()
    }

    fn publish_rl(&mut self) -> Result<(), Error> {
        self.ensure_mask()?;
        {
            let gpu = &self.gpu;
            let spectrum = self.spectrum.as_mut().unwrap();
            let strength = self.dering;
            let n = spectrum.real_n as u32;
            klaunch!(
                gpu,
                n,
                &gpu.lerp_mask,
                &spectrum.obj,
                &spectrum.wiener,
                &spectrum.mask,
                &mut spectrum.tmp,
                &(spectrum.real_n as i32),
                &strength,
            );
        }
        self.crop_from_tmp()?;
        self.recolor()
    }

    fn ensure_mask(&mut self) -> Result<(), Error> {
        if self.spectrum.as_ref().is_some_and(|spectrum| spectrum.mask_ready) {
            return Ok(());
        }
        {
            let gpu = &self.gpu;
            let spectrum = self.spectrum.as_mut().unwrap();
            let n = spectrum.real_n as u32;
            klaunch!(gpu, n, &gpu.edge_mask, &spectrum.observed, &mut spectrum.mask, &spectrum.fft_w, &spectrum.fft_h);
            klaunch!(gpu, n, &gpu.box_blur3, &spectrum.mask, &mut spectrum.tmp, &spectrum.fft_w, &spectrum.fft_h);
            klaunch!(gpu, n, &gpu.copy_f32, &spectrum.tmp, &mut spectrum.mask, &(spectrum.real_n as i32));
            spectrum.mask_ready = true;
        }
        Ok(())
    }

    fn crop_from_wiener(&mut self) -> Result<(), Error> {
        let gpu = &self.gpu;
        let picture = self.picture.as_mut().unwrap();
        let spectrum = self.spectrum.as_ref().unwrap();
        klaunch!(
            gpu,
            picture.pixels as u32,
            &gpu.crop_f32,
            &spectrum.wiener,
            &mut picture.y_out,
            &spectrum.fft_w,
            &spectrum.origin_x,
            &spectrum.origin_y,
            &picture.width,
            &picture.height,
        );
        Ok(())
    }

    fn crop_from_tmp(&mut self) -> Result<(), Error> {
        let gpu = &self.gpu;
        let picture = self.picture.as_mut().unwrap();
        let spectrum = self.spectrum.as_ref().unwrap();
        klaunch!(
            gpu,
            picture.pixels as u32,
            &gpu.crop_f32,
            &spectrum.tmp,
            &mut picture.y_out,
            &spectrum.fft_w,
            &spectrum.origin_x,
            &spectrum.origin_y,
            &picture.width,
            &picture.height,
        );
        Ok(())
    }

    fn recolor(&mut self) -> Result<(), Error> {
        let gpu = &self.gpu;
        let picture = self.picture.as_mut().unwrap();
        let n = picture.pixels as i32;
        klaunch!(
            gpu,
            picture.pixels as u32,
            &gpu.apply_chroma,
            &picture.original,
            &picture.y_in,
            &picture.y_out,
            &mut picture.processed,
            &n,
            &picture.channels,
        );
        Ok(())
    }
}

impl Gpu {
    fn c2r_into(&self, spectrum: &mut Spectrum) -> Result<(), Error> {
        spectrum.c2r.exec_c2r(&mut spectrum.spec_a, &mut spectrum.wiener).map_err(gpu_err)
    }
}

fn gpu_err(err: impl ToString) -> Error {
    Error::new(format!("CUDA : {}", err.to_string()))
}

/// Une 1060 a besoin d'un pilote de la branche 580, le dernier qui expose CUDA 12 pour Pascal.
fn module_err(err: impl ToString, arch: &str) -> Error {
    let text = gpu_err(err);
    if arch == "sm_61" {
        Error::new(format!(
            "{text}. Une GTX 1060 a besoin d'un pilote de la branche 580, le dernier qui expose CUDA 12 pour Pascal."
        ))
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
        use crate::color::mean;
    use crate::reference::{blur, contrast};

    #[test]
    fn kernel_arch_splits_pascal_and_turing() {
        assert_eq!(kernel_arch(6, 0), None);
        assert_eq!(kernel_arch(6, 1), Some("sm_61"));
        assert_eq!(kernel_arch(6, 2), Some("sm_61"));
        assert_eq!(kernel_arch(7, 0), Some("sm_70"));
        assert_eq!(kernel_arch(7, 5), Some("sm_70"));
        assert_eq!(kernel_arch(8, 6), Some("sm_70"));
    }

    #[test]
    fn embedded_ptx_has_no_unresolved_extern() {
        for ptx in [include_str!(env!("DUBFOCUS_PTX")), include_str!(env!("DUBFOCUS_PTX_SM61"))] {
            assert!(!ptx.contains(".extern .func"), "{ptx}");
        }
        assert!(include_str!(env!("DUBFOCUS_PTX")).contains(".target sm_70"));
        assert!(!include_str!(env!("DUBFOCUS_PTX")).contains(".target sm_61"));
        assert!(include_str!(env!("DUBFOCUS_PTX_SM61")).contains(".target sm_61"));
        assert!(include_str!(env!("DUBFOCUS_PTX_SM61")).contains(".version 5.0"));
    }

    #[test]
    fn probe_document_lists_arch() {
        let pascal = GpuInfo { name: "GTX 1060".into(), total: 6 * 1024 * 1024 * 1024, free: 1024, major: 6, minor: 1 };
        let value: serde_json::Value = serde_json::from_str(&probe_document(&pascal)).unwrap();
        assert_eq!(value["major"], 6);
        assert_eq!(value["minor"], 1);
        assert_eq!(value["kernels"], "sm_61");
        assert_eq!(pascal.summary(), "GTX 1060 · compute 6.1 · sm_61 · 6.0 Gio");
        let recent = GpuInfo { name: "RTX 3060".into(), major: 8, minor: 6, ..pascal };
        let value: serde_json::Value = serde_json::from_str(&probe_document(&recent)).unwrap();
        assert_eq!(value["major"], 8);
        assert_eq!(value["minor"], 6);
        assert_eq!(value["kernels"], "sm_70");
        let old = GpuInfo { major: 5, minor: 2, ..pascal };
        let value: serde_json::Value = serde_json::from_str(&probe_document(&old)).unwrap();
        assert_eq!(value["kernels"], "unsupported");
    }

    #[test]
    fn recent_gpu_loads_sm70() {
        let info = match probe_gpu() {
            Ok(info) => info,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        let session = match Session::new() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        if (info.major, info.minor) >= (7, 0) {
            assert_eq!(kernel_arch(info.major, info.minor), Some("sm_70"));
            assert_eq!(session.arch(), "sm_70");
        }
    }

    #[test]
    fn gpu_restores_a_blurred_edge() {
        let session = match Session::new() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        restores_blurred_edge(session);
    }

    #[test]
    fn pascal_module_launches_on_newer_gpu() {
        let session = match Session::new_with_arch("sm_61") {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        assert_eq!(session.arch(), "sm_61");
        restores_blurred_edge(session);
    }

    fn restores_blurred_edge(mut session: Session) {
        let width = 64usize;
        let height = 48usize;
        let mut sharp = vec![0.2f32; width * height];
        for y in 0..height {
            for x in width / 2..width {
                sharp[y * width + x] = 0.8;
            }
        }
        let blurred = blur(&sharp, width, height, PsfKind::Gaussian, 1.2);
        let before = contrast(&blurred, width, height);
        session
            .load(&blurred, &blurred, width as i32, height as i32, 1, mean(&blurred))
            .unwrap();
        session.wiener(PsfKind::Gaussian, 1.2, 1.0e-4).unwrap();
        let preview = session.download_luma().unwrap();
        assert!(contrast(&preview, width, height) > before, "wiener {before}");
        let cancel = AtomicBool::new(false);
        session
            .refine(12, 1.0e-4, 0.0, 0.0, &cancel, |_, _| {})
            .unwrap();
        let restored = session.download_luma().unwrap();
        assert!(contrast(&restored, width, height) > before * 1.05, "{before} -> {}", contrast(&restored, width, height));
    }

    #[test]
    fn moon_demo_gets_sharper() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("demo").join("2026-05-22-224312-Lunar-RAW_stacked.png");
        if !path.is_file() {
            return;
        }
        let mut session = match Session::new() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        let image = crate::io::load_path(&path).unwrap();
        let linear = crate::color::to_linear(&image.samples, image.suggest_srgb());
        let luma = crate::color::luminance(&linear, image.channels);
        let width = image.width as usize;
        let height = image.height as usize;
        let before = contrast(&luma, width, height);
        session
            .load(&linear, &luma, image.width as i32, image.height as i32, image.channels as i32, mean(&luma))
            .unwrap();
        session.wiener(PsfKind::Gaussian, 1.1, 1.0e-3).unwrap();
        let cancel = AtomicBool::new(false);
        let finished = session.refine(8, 1.0e-3, 0.01, 0.35, &cancel, |_, _| {}).unwrap();
        assert!(finished);
        let restored = session.download_luma().unwrap();
        assert!(restored.iter().all(|value| value.is_finite()));
        let after = contrast(&restored, width, height);
        assert!(after > before, "contraste {before} -> {after}");
    }

    #[test]
    fn shade_shows_the_tiff() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("demo").join("2026-05-22-224312-Lunar-RAW_stacked.tiff");
        let mut session = match Session::new() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        let image = crate::io::load_path(&path).unwrap();
        let finite = image.samples.iter().copied().filter(|value| value.is_finite());
        let mut min = f32::MAX;
        let mut max = 0.0f32;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for value in finite {
            min = min.min(value);
            max = max.max(value);
            sum += value;
            count += 1.0;
        }
        eprintln!(
            "tiff {}x{} depth {} ch {} min {min} max {max} mean {}",
            image.width,
            image.height,
            image.bit_depth,
            image.channels,
            sum / count
        );
        let linear = crate::color::to_linear(&image.samples, false);
        let luma = crate::color::luminance(&linear, image.channels);
        session
            .load(&linear, &luma, image.width as i32, image.height as i32, image.channels as i32, mean(&luma))
            .unwrap();
        session.wiener(PsfKind::Disk, 1.3, 10f32.powf(-2.5)).unwrap();
        let processed = session.download().unwrap();
        let pmax = processed.iter().copied().fold(0.0f32, f32::max);
        assert!(pmax > 0.05, "image traitée vide");
        let pixels = session
            .shade(&ViewRequest {
                view_w: 320,
                view_h: 240,
                origin_x: 0.0,
                origin_y: 0.0,
                scale: image.width as f32 / 320.0,
                wipe_x: 160.0,
                black: 0.0,
                white: 1.0,
                gamma: 1.0,
            })
            .unwrap();
        assert_eq!(pixels.len(), 320 * 240 * 4);
        let lit = pixels.chunks(4).filter(|px| px[0] > 20 || px[1] > 20 || px[2] > 20).count();
        assert!(lit > 320 * 240 / 5, "prévisualisation trop sombre : {lit} pixels");
    }

    #[test]
    fn disk_preview_sharpens_without_a_weave() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("demo").join("2026-05-22-224312-Lunar-RAW_stacked.png");
        if !path.is_file() {
            return;
        }
        let mut session = match Session::new() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("GPU ignoré : {err}");
                return;
            }
        };
        let image = crate::io::load_path(&path).unwrap();
        let linear = crate::color::to_linear(&image.samples, true);
        let luma = crate::color::luminance(&linear, image.channels);
        let width = image.width as usize;
        let height = image.height as usize;
        session
            .load(&linear, &luma, image.width as i32, image.height as i32, image.channels as i32, mean(&luma))
            .unwrap();
        session.wiener(PsfKind::Disk, 2.3, 1.0e-3).unwrap();
        let cancel = AtomicBool::new(false);
        session.refine(20, 1.0e-3, 0.05, 0.25, &cancel, |_, _| {}).unwrap();
        let restored = session.download_luma().unwrap();
        let before = contrast(&luma, width, height);
        let after = contrast(&restored, width, height);
        assert!(after > before * 1.02, "contraste {before} -> {after}");
        let crop = |image: &[f32]| -> f32 {
            let x0 = width * 3 / 5;
            let y0 = height / 3;
            let mut acc = 0.0f32;
            let mut count = 0.0f32;
            for y in y0..y0 + 80 {
                for x in x0..x0 + 80 {
                    acc += (image[y * width + x] - image[y * width + x + 1]).abs();
                    count += 1.0;
                }
            }
            acc / count
        };
        assert!(crop(&restored) > crop(&luma) * 1.05, "le limbe ne gagne pas en netteté");
    }
}
