use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use crate::color::{downscale, luminance, mean, preview_rgba, to_linear};
use crate::error::Error;
use crate::gpu::{self, Session, ViewRequest};
use crate::io::{self, ImageData, SaveKind};
use crate::psf::PsfKind;

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
    Decoded {
        ticket: u64,
        width: u32,
        height: u32,
        channels: usize,
        bit_depth: u8,
        treat_srgb: bool,
        view_w: u32,
        view_h: u32,
        pixels: Vec<u8>,
    },
    Status { ticket: u64, text: String },
    Ready {
        ticket: u64,
        width: u32,
        height: u32,
        channels: usize,
        bit_depth: u8,
        millis: u128,
        preview: bool,
        memory_limited: bool,
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
    memory_limited: bool,
    loaded_key: Option<(bool, bool, u64)>,
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Msg>, cancel: Arc<AtomicBool>) {
    let mut session: Option<Session> = None;
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
            Cmd::Open { path, params, ticket } => open_command(&mut session, &tx, &cancel, &mut job, &path, params, ticket),
            Cmd::Params { params, ticket } => {
                match ready_session(&mut session, &tx, ticket) {
                    Ok(session) => update_params(session, &tx, &cancel, &mut job, params, ticket),
                    Err(err) => Err(err),
                }
            }
            Cmd::Refine { params, ticket } => {
                match ready_session(&mut session, &tx, ticket) {
                    Ok(session) => refine(session, &tx, &cancel, &mut job, params, ticket),
                    Err(err) => Err(err),
                }
            }
            Cmd::Dering { value, ticket } => {
                match ready_session(&mut session, &tx, ticket) {
                    Ok(session) => retint(session, &tx, &mut job, value, ticket),
                    Err(err) => Err(err),
                }
            }
            Cmd::Shade(request) => {
                let ticket = request.ticket;
                match ready_session(&mut session, &tx, ticket) {
                    Ok(session) => shade(session, &tx, job.as_ref(), request),
                    Err(err) => Err(err),
                }
            }
            Cmd::Export(request) => {
                let ticket = request.ticket;
                match ready_session(&mut session, &tx, ticket) {
                    Ok(session) => export(session, &tx, &cancel, &mut job, request),
                    Err(err) => Err(err),
                }
            }
        };
        if let Err((ticket, err)) = outcome {
            let _ = tx.send(Msg::Error { ticket, message: err.message });
        }
    }
}

fn ready_session<'a>(session: &'a mut Option<Session>, tx: &Sender<Msg>, ticket: u64) -> Result<&'a mut Session, (u64, Error)> {
    if session.is_none() {
        let _ = tx.send(Msg::Status { ticket, text: "Initialisation CUDA…".to_string() });
        match Session::new() {
            Ok(value) => *session = Some(value),
            Err(err) => return Err((ticket, err)),
        }
    }
    Ok(session.as_mut().unwrap())
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

fn open_command(
    session: &mut Option<Session>,
    tx: &Sender<Msg>,
    cancel: &AtomicBool,
    job: &mut Option<Job>,
    path: &std::path::Path,
    mut params: ProcessParams,
    ticket: u64,
) -> Result<(), (u64, Error)> {
    let image = io::load_path(path).map_err(|err| (ticket, err))?;
    params.treat_srgb = image.suggest_srgb();
    let (view_w, view_h, pixels) = preview_rgba(&image.samples, image.width, image.height, image.channels, params.treat_srgb);
    let _ = tx.send(Msg::Decoded {
        ticket,
        width: image.width,
        height: image.height,
        channels: image.channels,
        bit_depth: image.bit_depth,
        treat_srgb: params.treat_srgb,
        view_w,
        view_h,
        pixels,
    });
    let session = ready_session(session, tx, ticket)?;
    let generation = job.as_ref().map(|job| job.generation.wrapping_add(1)).unwrap_or(1);
    *job = Some(Job {
        raw: image,
        generation,
        params,
        coord_scale: 1.0,
        refined: false,
        memory_limited: false,
        loaded_key: None,
    });
    let _ = tx.send(Msg::Status { ticket, text: "Calcul sur le GPU…".to_string() });
    restore(session, tx, cancel, job.as_mut().unwrap(), ticket, true, false)
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
    restore(session, tx, cancel, job, ticket, true, false)
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
    if job.coord_scale != 1.0 || job.memory_limited {
        let saved = job.loaded_key;
        job.loaded_key = None;
        if let Err(err) = reload(session, job, false) {
            if session.has_picture() {
                job.loaded_key = saved;
            }
            return Err((ticket, err));
        }
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
    let saved = job.loaded_key;
    job.loaded_key = None;
    if let Err(err) = restore(session, tx, cancel, job, request.ticket, false, true) {
        if session.has_picture() {
            job.loaded_key = saved;
        }
        return Err(err);
    }
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
    allow_shrink: bool,
    quiet: bool,
) -> Result<(), (u64, Error)> {
    cancel.store(false, Ordering::Relaxed);
    let started = Instant::now();
    if !quiet {
        let _ = tx.send(Msg::Status { ticket, text: "Préparation GPU…".to_string() });
    }
    reload(session, job, allow_shrink).map_err(|err| (ticket, err))?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }
    if !quiet {
        let _ = tx.send(Msg::Status { ticket, text: "Filtre de Wiener…".to_string() });
    }
    session.wiener(job.params.kind, job.params.radius, job.params.k()).map_err(|err| (ticket, err))?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }
    if !quiet {
        let _ = tx.send(Msg::Status {
            ticket,
            text: if job.params.preview {
                "Richardson-Lucy, aperçu…".to_string()
            } else {
                "Richardson-Lucy…".to_string()
            },
        });
    }
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

fn reload(session: &mut Session, job: &mut Job, allow_shrink: bool) -> Result<(), Error> {
    let key = (job.params.treat_srgb, job.params.preview, job.generation);
    if allow_shrink && job.loaded_key == Some(key) {
        return Ok(());
    }
    let budget = session.memory_budget()?;
    let planned = if allow_shrink {
        gpu::plan_working_size_measured(job.raw.width, job.raw.height, job.raw.channels, job.params.preview, budget)?
    } else {
        let full = gpu::plan_working_size_measured(job.raw.width, job.raw.height, job.raw.channels, false, budget)?;
        if full.memory_limited || full.width != job.raw.width || full.height != job.raw.height {
            return Err(gpu::full_resolution_error(job.raw.width, job.raw.height, job.raw.channels, budget));
        }
        full
    };
    let linear = to_linear(&job.raw.samples, job.params.treat_srgb);
    let working = if planned.width == job.raw.width && planned.height == job.raw.height {
        linear
    } else {
        downscale(
            &linear,
            job.raw.width as usize,
            job.raw.height as usize,
            job.raw.channels,
            planned.width as usize,
            planned.height as usize,
        )
    };
    let luma = luminance(&working, job.raw.channels);
    let average = mean(&luma);
    job.loaded_key = None;
    session.load(&working, &luma, planned.width as i32, planned.height as i32, job.raw.channels as i32, average)?;
    job.coord_scale = planned.coord_scale;
    job.memory_limited = planned.memory_limited;
    job.loaded_key = Some(key);
    Ok(())
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
        memory_limited: job.memory_limited,
        refined,
        treat_srgb: job.params.treat_srgb,
    }
}
