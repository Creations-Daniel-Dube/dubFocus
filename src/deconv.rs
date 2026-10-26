use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use crate::color::{downscale, luminance, mean, to_linear};
use crate::error::Error;
use crate::gpu::{Session, ViewRequest};
use crate::io::{self, ImageData, SaveKind};
use crate::psf::PsfKind;

const PREVIEW_PIXELS: usize = 6_000_000;
const PREVIEW_EDGE: u32 = 2048;

#[derive(Clone, Copy, Debug)]
pub struct ProcessParams {
    pub kind: PsfKind,
    pub radius: f32,
    pub log_k: f32,
    pub iterations: u32,
    pub damping: f32,
    pub dering: f32,
    pub treat_srgb: bool,
    pub preview: bool,
}

impl ProcessParams {
    pub fn k(self) -> f32 {
        10f32.powf(self.log_k).clamp(1.0e-6, 0.5)
    }
}

impl Default for ProcessParams {
    fn default() -> Self {
        Self {
            kind: PsfKind::Disk,
            radius: 1.2,
            log_k: -2.5,
            iterations: 20,
            damping: 0.02,
            dering: 0.4,
            treat_srgb: true,
            preview: false,
        }
    }
}

pub struct ShadeRequest {
    pub ticket: u64,
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

pub struct ExportRequest {
    pub ticket: u64,
    pub path: PathBuf,
    pub kind: SaveKind,
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
}

enum Cmd {
    Open { path: PathBuf, params: ProcessParams, ticket: u64 },
    Params { params: ProcessParams, ticket: u64 },
    Refine { params: ProcessParams, ticket: u64 },
    Dering { value: f32, ticket: u64 },
    Shade(ShadeRequest),
    Export(ExportRequest),
}

pub enum Msg {
    Ready {
        ticket: u64,
        width: u32,
        height: u32,
        channels: usize,
        bit_depth: u8,
        millis: u128,
        preview: bool,
        refined: bool,
        treat_srgb: bool,
    },
    Progress { ticket: u64, value: f32 },
    Shaded { ticket: u64, width: usize, height: usize, pixels: Vec<u8> },
    Saved { ticket: u64, path: String },
    Error { ticket: u64, message: String },
}

pub struct Engine {
    tx: Sender<Cmd>,
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
}

impl Engine {
    pub fn spawn() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (msg_tx, msg_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        std::thread::spawn(move || worker(cmd_rx, msg_tx, flag));
        Self { tx: cmd_tx, rx: msg_rx, cancel }
    }

    pub fn open(&self, path: PathBuf, params: ProcessParams, ticket: u64) {
        self.cancel.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Open { path, params, ticket });
    }

    pub fn params(&self, params: ProcessParams, ticket: u64) {
        self.cancel.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Params { params, ticket });
    }

    pub fn refine(&self, params: ProcessParams, ticket: u64) {
        self.cancel.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Refine { params, ticket });
    }

    pub fn dering(&self, value: f32, ticket: u64) {
        let _ = self.tx.send(Cmd::Dering { value, ticket });
    }

    pub fn shade(&self, request: ShadeRequest) {
        let _ = self.tx.send(Cmd::Shade(request));
    }

    pub fn export(&self, request: ExportRequest) {
        self.cancel.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Export(request));
    }

    pub fn poll(&self) -> Option<Msg> {
        self.rx.try_recv().ok()
    }
}

struct Job {
    raw: ImageData,
    generation: u64,
    params: ProcessParams,
    coord_scale: f32,
    refined: bool,
    loaded_key: Option<(bool, bool, u64)>,
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Msg>, cancel: Arc<AtomicBool>) {
    let mut session = match Session::new() {
        Ok(session) => session,
        Err(err) => {
            let _ = tx.send(Msg::Error { ticket: 0, message: err.message });
            while rx.recv().is_ok() {}
            return;
        }
    };
    let mut job: Option<Job> = None;
    let mut stash: Option<Cmd> = None;
    loop {
        let incoming = if let Some(cmd) = stash.take() {
            cmd
        } else {
            match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        };
        let cmd = coalesce(incoming, &rx, &mut stash);
        let outcome = match cmd {
            Cmd::Open { path, params, ticket } => open_path(&mut session, &tx, &cancel, &mut job, &path, params, ticket),
            Cmd::Params { params, ticket } => update_params(&mut session, &tx, &cancel, &mut job, params, ticket),
            Cmd::Refine { params, ticket } => refine(&mut session, &tx, &cancel, &mut job, params, ticket),
            Cmd::Dering { value, ticket } => retint(&mut session, &tx, &mut job, value, ticket),
            Cmd::Shade(request) => shade(&mut session, &tx, job.as_ref(), request),
            Cmd::Export(request) => export(&mut session, &tx, &cancel, &mut job, request),
        };
        if let Err((ticket, err)) = outcome {
            let _ = tx.send(Msg::Error { ticket, message: err.message });
        }
    }
}

fn coalesce(mut cmd: Cmd, rx: &Receiver<Cmd>, stash: &mut Option<Cmd>) -> Cmd {
    loop {
        match rx.try_recv() {
            Ok(next) => match (&mut cmd, next) {
                (Cmd::Params { params, ticket }, Cmd::Params { params: next, ticket: next_ticket }) => {
                    *params = next;
                    *ticket = next_ticket;
                }
                (Cmd::Dering { value, ticket }, Cmd::Dering { value: next, ticket: next_ticket }) => {
                    *value = next;
                    *ticket = next_ticket;
                }
                (Cmd::Shade(current), Cmd::Shade(next)) => *current = next,
                (_, next) => {
                    *stash = Some(next);
                    break;
                }
            },
            Err(_) => break,
        }
    }
    cmd
}

fn open_path(
    session: &mut Session,
    tx: &Sender<Msg>,
    cancel: &AtomicBool,
    job: &mut Option<Job>,
    path: &std::path::Path,
    mut params: ProcessParams,
    ticket: u64,
) -> Result<(), (u64, Error)> {
    let image = io::load_path(path).map_err(|err| (ticket, err))?;
    params.treat_srgb = image.suggest_srgb();
    let generation = job.as_ref().map(|job| job.generation.wrapping_add(1)).unwrap_or(1);
    *job = Some(Job {
        raw: image,
        generation,
        params,
        coord_scale: 1.0,
        refined: false,
        loaded_key: None,
    });
    restore(session, tx, cancel, job.as_mut().unwrap(), ticket, false)
}

fn update_params(
    session: &mut Session,
    tx: &Sender<Msg>,
    cancel: &AtomicBool,
    job: &mut Option<Job>,
    params: ProcessParams,
    ticket: u64,
) -> Result<(), (u64, Error)> {
    let Some(job) = job.as_mut() else {
        return Ok(());
    };
    job.params = params;
    job.refined = false;
    restore(session, tx, cancel, job, ticket, false)
}

fn refine(
    session: &mut Session,
    tx: &Sender<Msg>,
    cancel: &AtomicBool,
    job: &mut Option<Job>,
    params: ProcessParams,
    ticket: u64,
) -> Result<(), (u64, Error)> {
    let Some(job) = job.as_mut() else {
        return Ok(());
    };
    job.params = params;
    job.params.preview = false;
    if job.coord_scale != 1.0 {
        job.loaded_key = None;
        reload(session, job).map_err(|err| (ticket, err))?;
        session.wiener(job.params.kind, job.params.radius, job.params.k()).map_err(|err| (ticket, err))?;
    }
    cancel.store(false, Ordering::Relaxed);
    let started = Instant::now();
    let tx_progress = tx.clone();
    let finished = session
        .refine(job.params.iterations, job.params.k(), job.params.damping, job.params.dering, cancel, |done, total| {
            let _ = tx_progress.send(Msg::Progress { ticket, value: done as f32 / total as f32 });
        })
        .map_err(|err| (ticket, err))?;
    if finished {
        job.refined = true;
        let _ = tx.send(ready(job, ticket, started.elapsed().as_millis(), true));
    }
    Ok(())
}

fn retint(session: &mut Session, tx: &Sender<Msg>, job: &mut Option<Job>, value: f32, ticket: u64) -> Result<(), (u64, Error)> {
    let Some(job) = job.as_mut() else {
        return Ok(());
    };
    job.params.dering = value;
    if job.refined {
        session.set_dering(value).map_err(|err| (ticket, err))?;
        let _ = tx.send(ready(job, ticket, 0, true));
    }
    Ok(())
}

fn shade(session: &mut Session, tx: &Sender<Msg>, job: Option<&Job>, request: ShadeRequest) -> Result<(), (u64, Error)> {
    let Some(job) = job else {
        return Ok(());
    };
    let scale = job.coord_scale.max(1.0e-6);
    let pixels = session
        .shade(&ViewRequest {
            view_w: request.view_w,
            view_h: request.view_h,
            origin_x: request.origin_x / scale,
            origin_y: request.origin_y / scale,
            scale: request.scale / scale,
            wipe_x: request.wipe_x,
            black: request.black,
            white: request.white,
            gamma: request.gamma,
        })
        .map_err(|err| (request.ticket, err))?;
    let _ = tx.send(Msg::Shaded {
        ticket: request.ticket,
        width: request.view_w as usize,
        height: request.view_h as usize,
        pixels,
    });
    Ok(())
}

fn export(session: &mut Session, tx: &Sender<Msg>, cancel: &AtomicBool, job: &mut Option<Job>, request: ExportRequest) -> Result<(), (u64, Error)> {
    let Some(job) = job.as_mut() else {
        return Ok(());
    };
    job.params.preview = false;
    job.loaded_key = None;
    restore(session, tx, cancel, job, request.ticket, true)?;
    if !job.refined {
        return Ok(());
    }
    let linear = session.download().map_err(|err| (request.ticket, err))?;
    io::save_processed(
        &request.path,
        job.raw.width,
        job.raw.height,
        job.raw.channels,
        &linear,
        request.black,
        request.white,
        request.gamma,
        request.kind,
    )
    .map_err(|err| (request.ticket, err))?;
    let _ = tx.send(Msg::Saved { ticket: request.ticket, path: request.path.display().to_string() });
    let _ = tx.send(ready(job, request.ticket, 0, job.refined));
    Ok(())
}

fn restore(
    session: &mut Session,
    tx: &Sender<Msg>,
    cancel: &AtomicBool,
    job: &mut Job,
    ticket: u64,
    quiet: bool,
) -> Result<(), (u64, Error)> {
    cancel.store(false, Ordering::Relaxed);
    let started = Instant::now();
    reload(session, job).map_err(|err| (ticket, err))?;
    session.wiener(job.params.kind, job.params.radius, job.params.k()).map_err(|err| (ticket, err))?;
    let tx_progress = tx.clone();
    let finished = session
        .refine(job.params.iterations, job.params.k(), job.params.damping, job.params.dering, cancel, |done, total| {
            let _ = tx_progress.send(Msg::Progress { ticket, value: done as f32 / total as f32 });
        })
        .map_err(|err| (ticket, err))?;
    if !finished {
        job.refined = false;
        return Ok(());
    }
    job.refined = true;
    if !quiet {
        let _ = tx.send(ready(job, ticket, started.elapsed().as_millis(), true));
    }
    Ok(())
}

fn reload(session: &mut Session, job: &mut Job) -> Result<(), Error> {
    let key = (job.params.treat_srgb, job.params.preview, job.generation);
    if job.loaded_key == Some(key) {
        return Ok(());
    }
    let linear = to_linear(&job.raw.samples, job.params.treat_srgb);
    let (width, height, coord_scale) = working_size(job.raw.width, job.raw.height, job.params.preview);
    let working = if width == job.raw.width && height == job.raw.height {
        linear
    } else {
        downscale(&linear, job.raw.width as usize, job.raw.height as usize, job.raw.channels, width as usize, height as usize)
    };
    let luma = luminance(&working, job.raw.channels);
    let average = mean(&luma);
    session.load(&working, &luma, width as i32, height as i32, job.raw.channels as i32, average)?;
    job.coord_scale = coord_scale;
    job.loaded_key = Some(key);
    Ok(())
}

fn working_size(width: u32, height: u32, preview: bool) -> (u32, u32, f32) {
    let long = width.max(height);
    let pixels = width as usize * height as usize;
    if !preview || pixels <= PREVIEW_PIXELS || long <= PREVIEW_EDGE {
        return (width, height, 1.0);
    }
    let scale = PREVIEW_EDGE as f32 / long as f32;
    let target_w = ((width as f32) * scale).round().max(1.0) as u32;
    let target_h = ((height as f32) * scale).round().max(1.0) as u32;
    (target_w, target_h, width as f32 / target_w as f32)
}

fn ready(job: &Job, ticket: u64, millis: u128, refined: bool) -> Msg {
    Msg::Ready {
        ticket,
        width: job.raw.width,
        height: job.raw.height,
        channels: job.raw.channels,
        bit_depth: job.raw.bit_depth,
        millis,
        preview: job.coord_scale != 1.0,
        refined,
        treat_srgb: job.params.treat_srgb,
    }
}
