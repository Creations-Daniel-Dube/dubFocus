use std::path::PathBuf;

use eframe::egui::{
    self, Align, Color32, ColorImage, Context, Frame, Layout, Rect, RichText, ScrollArea, Sense, Stroke, TextureHandle, TextureOptions, Vec2, pos2, vec2,
};
use rfd::FileDialog;

use crate::deconv::{Engine, ExportRequest, Msg, ProcessParams, ShadeRequest};
use crate::gpu::{kernel_arch, probe_gpu, unsupported_message};
use crate::io::SaveKind;
use crate::psf::PsfKind;
use crate::settings::Settings;

struct Meta {
    width: u32,
    height: u32,
    channels: usize,
    bit_depth: u8,
}

struct Camera {
    cx: f32,
    cy: f32,
    zoom: f32,
}

impl Camera {
    fn fit(&mut self, width: f32, height: f32, view_w: f32, view_h: f32) {
        if width <= 1.0 || height <= 1.0 || view_w <= 1.0 || view_h <= 1.0 {
            return;
        }
        self.zoom = (view_w / width).min(view_h / height).clamp(0.02, 40.0);
        self.cx = width * 0.5;
        self.cy = height * 0.5;
    }
}

pub struct App {
    settings: Settings,
    engine: Option<Engine>,
    gpu_line: String,
    warn_text: String,
    warn_open: bool,
    error: Option<String>,
    path: String,
    meta: Option<Meta>,
    params: ProcessParams,
    black: f32,
    white: f32,
    gamma: f32,
    wipe: f32,
    save_kind: SaveKind,
    camera: Camera,
    fit_next: bool,
    texture: Option<TextureHandle>,
    ticket: u64,
    shade_ticket: u64,
    shade_dirty: bool,
    shade_px: [i32; 2],
    shade_pending: bool,
    wiping: bool,
    busy: bool,
    progress: f32,
    refined: bool,
    status: String,
    settings_dirty: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(theme());
        install_font(&cc.egui_ctx);
        let settings = Settings::load();
        let gpu = probe_gpu();
        let (gpu_line, warn_text, start_engine) = match &gpu {
            Ok(info) => {
                let line = format!("GPU : {}", info.summary());
                match kernel_arch(info.major, info.minor) {
                    Some(_) => (line, String::new(), true),
                    None => (line, unsupported_message(info.major, info.minor), false),
                }
            }
            Err(err) => (
                format!("GPU indisponible : {err}"),
                "dubFocus a besoin d'un GPU NVIDIA et du pilote CUDA.".to_string(),
                false,
            ),
        };
        let warn_open = !start_engine;
        let engine = if start_engine { Some(Engine::spawn()) } else { None };
        let params = ProcessParams {
            kind: settings.kind(),
            radius: settings.radius.clamp(0.3, 8.0),
            log_k: settings.log_k.clamp(-4.0, -1.0),
            iterations: settings.iterations.clamp(1, 80),
            damping: settings.damping.clamp(0.0, 0.2),
            dering: settings.dering.clamp(0.0, 1.0),
            treat_srgb: true,
            preview: false,
        };
        Self {
            save_kind: settings.save_kind(),
            black: settings.black.clamp(0.0, 0.8),
            white: settings.white.clamp(0.2, 1.5),
            gamma: settings.gamma.clamp(0.2, 3.0),
            settings,
            engine,
            gpu_line,
            warn_text,
            warn_open,
            error: None,
            path: String::new(),
            meta: None,
            params,
            wipe: 0.5,
            camera: Camera { cx: 0.0, cy: 0.0, zoom: 1.0 },
            fit_next: false,
            texture: None,
            ticket: 0,
            shade_ticket: 0,
            shade_dirty: false,
            shade_px: [0, 0],
            shade_pending: false,
            wiping: false,
            busy: false,
            progress: 0.0,
            refined: false,
            status: "Ouvrez une image jpg, png ou tiff.".to_string(),
            settings_dirty: false,
        }
    }

    fn snapshot_params(&self, preview: bool) -> ProcessParams {
        let mut params = self.params;
        params.preview = preview;
        params.radius = params.radius.clamp(0.3, 8.0);
        params
    }

    fn mark_settings(&mut self) {
        self.settings.disk = self.params.kind == PsfKind::Disk;
        self.settings.radius = self.params.radius;
        self.settings.iterations = self.params.iterations;
        self.settings.log_k = self.params.log_k;
        self.settings.damping = self.params.damping;
        self.settings.dering = self.params.dering;
        self.settings.black = self.black;
        self.settings.white = self.white;
        self.settings.gamma = self.gamma;
        self.settings.set_save_kind(self.save_kind);
        self.settings_dirty = true;
    }

    fn remember_dir(&mut self, path: &std::path::Path) {
        if let Some(dir) = path.parent() {
            let text = dir.display().to_string();
            if !text.is_empty() {
                self.settings.last_dir = text;
                self.settings_dirty = true;
            }
        }
    }

    fn bump(&mut self) -> u64 {
        self.ticket = self.ticket.wrapping_add(1);
        self.ticket
    }

    fn send_process(&mut self, preview: bool) {
        if self.engine.is_none() || self.meta.is_none() {
            return;
        }
        let ticket = self.bump();
        self.busy = true;
        self.progress = 0.0;
        self.refined = false;
        self.status = if preview { "Richardson-Lucy, aperçu…".to_string() } else { "Richardson-Lucy…".to_string() };
        let params = self.snapshot_params(preview);
        self.engine.as_ref().unwrap().params(params, ticket);
        self.mark_settings();
    }

    fn open_dialog(&mut self) {
        if self.engine.is_none() {
            return;
        }
        let mut dialog = FileDialog::new().add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff"]);
        if !self.settings.last_dir.is_empty() {
            dialog = dialog.set_directory(&self.settings.last_dir);
        }
        let Some(path) = dialog.pick_file() else { return };
        self.remember_dir(&path);
        self.path = path.display().to_string();
        self.fit_next = true;
        self.busy = true;
        self.progress = 0.0;
        self.refined = false;
        self.texture = None;
        self.status = "Lecture…".to_string();
        let params = self.snapshot_params(false);
        let ticket = self.bump();
        self.engine.as_ref().unwrap().open(path, params, ticket);
    }

    fn save_dialog(&mut self) {
        if self.engine.is_none() || self.meta.is_none() {
            return;
        }
        let mut dialog = FileDialog::new().set_file_name(format!("dubfocus.{}", self.save_kind.extension()));
        if !self.settings.last_dir.is_empty() {
            dialog = dialog.set_directory(&self.settings.last_dir);
        }
        let Some(path) = dialog.save_file() else { return };
        let path = ensure_extension(path, self.save_kind.extension());
        self.remember_dir(&path);
        self.busy = true;
        self.progress = 0.0;
        self.status = "Export…".to_string();
        let ticket = self.bump();
        let kind = self.save_kind;
        let black = self.black;
        let white = self.white;
        let gamma = self.gamma;
        self.engine.as_ref().unwrap().export(ExportRequest { ticket, path, kind, black, white, gamma });
        self.mark_settings();
    }

    fn request_shade(&mut self, view_w: i32, view_h: i32, origin_x: f32, origin_y: f32, scale: f32) {
        if self.engine.is_none() || view_w < 2 || view_h < 2 {
            return;
        }
        self.shade_ticket = self.shade_ticket.wrapping_add(1);
        let request = ShadeRequest {
            ticket: self.shade_ticket,
            view_w,
            view_h,
            origin_x,
            origin_y,
            scale,
            wipe_x: self.wipe.clamp(0.0, 1.0) * view_w as f32,
            black: self.black,
            white: self.white.max(self.black + 0.01),
            gamma: self.gamma.max(0.05),
        };
        self.engine.as_ref().unwrap().shade(request);
        self.shade_px = [view_w, view_h];
        self.shade_dirty = false;
        self.shade_pending = true;
    }

    fn poll(&mut self, ctx: &Context) {
        let Some(engine) = &self.engine else { return };
        while let Some(msg) = engine.poll() {
            match msg {
                Msg::Ready { ticket, width, height, channels, bit_depth, millis, preview, refined, treat_srgb } => {
                    if ticket != self.ticket && ticket != 0 {
                        continue;
                    }
                    self.meta = Some(Meta { width, height, channels, bit_depth });
                    self.params.treat_srgb = treat_srgb;
                    self.busy = false;
                    self.progress = 0.0;
                    self.refined = refined;
                    self.shade_dirty = true;
                    self.status = if refined && preview {
                        format!("Aperçu réduit · Richardson-Lucy · {} it · {millis} ms", self.params.iterations)
                    } else if refined {
                        format!("Richardson-Lucy · {} itérations · {millis} ms", self.params.iterations)
                    } else if preview {
                        format!("Aperçu réduit · {millis} ms")
                    } else if millis == 0 {
                        "Image à jour.".to_string()
                    } else {
                        format!("Aperçu · {millis} ms")
                    };
                }
                Msg::Progress { ticket, value } => {
                    if ticket == self.ticket {
                        self.progress = value;
                        self.busy = true;
                        self.status = format!("Richardson-Lucy · {:.0} %", value * 100.0);
                    }
                }
                Msg::Shaded { ticket, width, height, pixels } => {
                    if ticket != self.shade_ticket || pixels.len() != width * height * 4 {
                        continue;
                    }
                    let image = ColorImage::from_rgba_unmultiplied([width, height], &pixels);
                    if let Some(texture) = &mut self.texture {
                        texture.set(image, TextureOptions::LINEAR);
                    } else {
                        self.texture = Some(ctx.load_texture("vue", image, TextureOptions::LINEAR));
                    }
                    self.shade_pending = false;
                }
                Msg::Saved { ticket, path } => {
                    if ticket == self.ticket {
                        self.busy = false;
                        self.progress = 0.0;
                        self.status = format!("Enregistré : {path}");
                        self.shade_dirty = true;
                    }
                }
                Msg::Error { ticket, message } => {
                    if ticket == 0 || ticket == self.ticket {
                        self.busy = false;
                        self.shade_pending = false;
                        self.progress = 0.0;
                        self.status = "Échec du traitement.".to_string();
                        self.error = Some(message);
                    }
                }
            }
        }
        if self.busy || self.shade_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let open = ui.add_enabled(self.engine.is_some() && !self.busy, egui::Button::new("Ouvrir"));
            if open.clicked() {
                self.open_dialog();
            }
            let save = ui.add_enabled(self.engine.is_some() && self.meta.is_some() && !self.busy, egui::Button::new("Enregistrer"));
            if save.clicked() {
                self.save_dialog();
            }
            ui.label(short_path(&self.path));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(&self.gpu_line);
            });
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ScrollArea::vertical().show(ui, |ui| {
            group(ui, "Image", |ui| {
                if let Some(meta) = &self.meta {
                    let color = if meta.channels == 1 { "mono" } else { "RVB" };
                    ui.label(format!("{} × {} · {} bits · {color}", meta.width, meta.height, meta.bit_depth));
                } else {
                    ui.label("Aucune image");
                }
                let mut srgb = self.params.treat_srgb;
                if ui.checkbox(&mut srgb, "Interpréter comme sRGB").changed() && self.meta.is_some() {
                    self.params.treat_srgb = srgb;
                    self.send_process(false);
                }
            });
            group(ui, "PSF", |ui| {
                let mut kind = self.params.kind;
                let disk = ui.radio_value(&mut kind, PsfKind::Disk, "Disque de défocalisation");
                let gauss = ui.radio_value(&mut kind, PsfKind::Gaussian, "Gaussienne");
                if (disk.changed() || gauss.changed()) && kind != self.params.kind {
                    self.params.kind = kind;
                    self.send_process(false);
                }
                ui.label(if self.params.kind == PsfKind::Disk { "Rayon du disque, en pixels" } else { "Sigma, en pixels" });
                ui.label("Visez la taille du flou. Un rayon trop grand crée des anneaux.");
                let radius = ui.add(egui::Slider::new(&mut self.params.radius, 0.3..=8.0).fixed_decimals(2));
                if radius.changed() || radius.drag_stopped() {
                    let preview = radius.dragged() && !radius.drag_stopped();
                    self.send_process(preview);
                }
            });
            group(ui, "Restauration", |ui| {
                ui.label("Itérations Richardson-Lucy");
                let iterations = ui.add(egui::Slider::new(&mut self.params.iterations, 1..=80));
                if iterations.changed() || iterations.drag_stopped() {
                    self.mark_settings();
                    let preview = iterations.dragged() && !iterations.drag_stopped();
                    self.send_process(preview);
                }
                ui.label(format!("Régularisation  K = {:.4}", self.params.k()));
                let regular = ui.add(egui::Slider::new(&mut self.params.log_k, -4.0..=-1.0).fixed_decimals(2));
                if regular.changed() || regular.drag_stopped() {
                    let preview = regular.dragged() && !regular.drag_stopped();
                    self.send_process(preview);
                }
                ui.label("Amortissement");
                let damping = ui.add(egui::Slider::new(&mut self.params.damping, 0.0..=0.2).fixed_decimals(3));
                if damping.changed() || damping.drag_stopped() {
                    self.mark_settings();
                    let preview = damping.dragged() && !damping.drag_stopped();
                    self.send_process(preview);
                }
                ui.label("Anti-ringing");
                let mut percent = self.params.dering * 100.0;
                let dering = ui.add(egui::Slider::new(&mut percent, 0.0..=100.0).fixed_decimals(0));
                if dering.changed() {
                    self.params.dering = percent / 100.0;
                    self.mark_settings();
                    if self.refined && self.engine.is_some() {
                        let value = self.params.dering;
                        let ticket = self.bump();
                        self.busy = true;
                        self.engine.as_ref().unwrap().dering(value, ticket);
                    }
                }
                let refine = ui.add_enabled(self.meta.is_some() && !self.busy, egui::Button::new("Raffiner"));
                if refine.clicked() && self.engine.is_some() {
                    let ticket = self.bump();
                    let params = self.snapshot_params(false);
                    self.busy = true;
                    self.progress = 0.0;
                    self.status = "Richardson-Lucy…".to_string();
                    self.engine.as_ref().unwrap().refine(params, ticket);
                    self.mark_settings();
                }
                if self.busy {
                    ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                }
            });
            group(ui, "Niveaux", |ui| {
                ui.label("Appliqués à l'original et au résultat.");
                let mut changed = false;
                ui.label("Noir");
                changed |= ui.add(egui::Slider::new(&mut self.black, 0.0..=0.8).fixed_decimals(3)).changed();
                ui.label("Blanc");
                changed |= ui.add(egui::Slider::new(&mut self.white, 0.2..=1.5).fixed_decimals(3)).changed();
                ui.label("Gamma");
                changed |= ui.add(egui::Slider::new(&mut self.gamma, 0.2..=3.0).fixed_decimals(2)).changed();
                if ui.button("Réinitialiser").clicked() {
                    self.black = 0.0;
                    self.white = 1.0;
                    self.gamma = 1.0;
                    changed = true;
                }
                if changed {
                    self.white = self.white.max(self.black + 0.01);
                    self.shade_dirty = true;
                    self.mark_settings();
                }
            });
            group(ui, "Balayage", |ui| {
                if ui.add(egui::Slider::new(&mut self.wipe, 0.0..=1.0).fixed_decimals(2)).changed() {
                    self.shade_dirty = true;
                }
                ui.label("À gauche l'original, à droite le résultat.");
            });
            group(ui, "Export", |ui| {
                egui::ComboBox::from_id_salt("export")
                    .selected_text(self.save_kind.label())
                    .show_ui(ui, |ui| {
                        for kind in SaveKind::all() {
                            if ui.selectable_value(&mut self.save_kind, kind, kind.label()).changed() {
                                self.mark_settings();
                            }
                        }
                    });
                ui.label("PNG, JPEG et TIFF 16 bits suivent l'écran. Le TIFF 32 bits reste linéaire, niveaux compris.");
            });
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.label(&self.status);
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available.max(vec2(1.0, 1.0)), Sense::click_and_drag());
        ui.painter().rect_filled(rect, 4.0, Color32::from_rgb(0x0e, 0x03, 0x03));
        let ppp = ui.ctx().pixels_per_point();
        let view_w = (rect.width() * ppp).round().max(1.0);
        let view_h = (rect.height() * ppp).round().max(1.0);
        if self.fit_next {
            if let Some(meta) = &self.meta {
                self.camera.fit(meta.width as f32, meta.height as f32, view_w, view_h);
                self.fit_next = false;
                self.shade_dirty = true;
            }
        }
        self.interact(ui, &response, rect, view_w, view_h, ppp);
        let scale = 1.0 / self.camera.zoom.max(0.02);
        let origin_x = self.camera.cx - view_w * scale * 0.5;
        let origin_y = self.camera.cy - view_h * scale * 0.5;
        let view = [view_w as i32, view_h as i32];
        if self.meta.is_some() && self.shade_px != view {
            self.shade_dirty = true;
        }
        if self.shade_dirty && self.meta.is_some() && !self.busy {
            self.request_shade(view[0], view[1], origin_x, origin_y, scale);
        }
        let painter = ui.painter().with_clip_rect(rect);
        let image_rect = if let Some(texture) = &self.texture {
            let [tex_w, tex_h] = texture.size();
            let dest = preview_dest(rect, tex_w as i32, tex_h as i32, ppp);
            painter.image(texture.id(), dest, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            dest
        } else {
            if self.meta.is_none() {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Ouvrez une image pour raffiner la mise au point",
                    egui::TextStyle::Button.resolve(ui.style()),
                    Color32::from_rgb(0xff, 0xe1, 0x4d),
                );
            }
            rect
        };
        let line_x = image_rect.min.x + self.wipe.clamp(0.0, 1.0) * image_rect.width();
        painter.vline(line_x, image_rect.y_range(), Stroke::new(2.0, Color32::from_rgb(0xff, 0x93, 0x20)));
    }

    fn interact(&mut self, ui: &mut egui::Ui, response: &egui::Response, rect: Rect, view_w: f32, view_h: f32, ppp: f32) {
        if response.double_clicked() {
            if let Some(meta) = &self.meta {
                self.camera.fit(meta.width as f32, meta.height as f32, view_w, view_h);
                self.shade_dirty = true;
            }
        }
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                let line_x = rect.min.x + self.wipe.clamp(0.0, 1.0) * rect.width();
                self.wiping = (pos.x - line_x).abs() < 8.0;
            }
        }
        if response.dragged() {
            if self.wiping {
                if let Some(pos) = response.interact_pointer_pos() {
                    self.wipe = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                    self.shade_dirty = true;
                }
            } else {
                let delta = response.drag_delta();
                self.camera.cx -= delta.x * ppp / self.camera.zoom;
                self.camera.cy -= delta.y * ppp / self.camera.zoom;
                self.shade_dirty = true;
            }
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(pos) = response.hover_pos() {
                    let sx = (pos.x - rect.min.x) * ppp;
                    let sy = (pos.y - rect.min.y) * ppp;
                    let scale = 1.0 / self.camera.zoom;
                    let img_x = self.camera.cx - view_w * scale * 0.5 + sx * scale;
                    let img_y = self.camera.cy - view_h * scale * 0.5 + sy * scale;
                    let factor = (scroll * 0.002).exp();
                    self.camera.zoom = (self.camera.zoom * factor).clamp(0.02, 40.0);
                    self.camera.cx = img_x + (view_w * 0.5 - sx) / self.camera.zoom;
                    self.camera.cy = img_y + (view_h * 0.5 - sy) / self.camera.zoom;
                    self.shade_dirty = true;
                }
            }
        }
    }

    fn dialogs(&mut self, ctx: &Context) {
        if self.warn_open {
            let mut open = self.warn_open;
            egui::Window::new("CUDA requis")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    if !self.warn_text.is_empty() {
                        ui.label(&self.warn_text);
                    }
                    ui.label(&self.gpu_line);
                    if ui.button("OK").clicked() {
                        open = false;
                    }
                });
            self.warn_open = open;
        }
        if self.error.is_some() {
            let message = self.error.clone().unwrap_or_default();
            let mut close = false;
            egui::Window::new("Traitement impossible")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(&message);
                    if ui.button("OK").clicked() {
                        close = true;
                    }
                });
            if close {
                self.error = None;
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::left("controls").exact_size(348.0).resizable(false).show(ui, |ui| self.controls(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::CentralPanel::default().show(ui, |ui| self.viewport(ui));
        self.dialogs(&ctx);
        if self.settings_dirty {
            self.settings.save();
            self.settings_dirty = false;
        }
    }

    fn on_exit(&mut self) {
        self.mark_settings();
        self.settings.save();
    }
}

fn app_icon() -> egui::IconData {
    let image = image::load_from_memory(include_bytes!("../images/dubFocus.png"))
        .expect("images/dubFocus.png")
        .into_rgba8();
    let (width, height) = image.dimensions();
    egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    }
}

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([980.0, 640.0])
            .with_title("dubFocus")
            .with_icon(app_icon()),
        centered: true,
        ..Default::default()
    };
    eframe::run_native("dubFocus", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

fn theme() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    let label = Color32::from_rgb(0xff, 0xe1, 0x4d);
    let control = Color32::from_rgb(0xff, 0x93, 0x20);
    visuals.window_fill = Color32::from_rgb(0x16, 0x05, 0x05);
    visuals.panel_fill = Color32::from_rgb(0x16, 0x05, 0x05);
    visuals.extreme_bg_color = Color32::from_rgb(0x0e, 0x03, 0x03);
    visuals.faint_bg_color = Color32::from_rgb(0x3d, 0x12, 0x10);
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(0x3d, 0x12, 0x10);
    visuals.widgets.noninteractive.fg_stroke.color = label;
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(0x62, 0x20, 0x18);
    visuals.widgets.inactive.fg_stroke.color = control;
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x7a, 0x28, 0x20);
    visuals.widgets.hovered.fg_stroke.color = control;
    visuals.widgets.active.bg_fill = Color32::from_rgb(0x4a, 0x14, 0x14);
    visuals.widgets.active.fg_stroke.color = control;
    visuals.widgets.open.bg_fill = Color32::from_rgb(0x62, 0x20, 0x18);
    visuals.widgets.open.fg_stroke.color = control;
    visuals.selection.bg_fill = Color32::from_rgb(0xc4, 0x3a, 0x18);
    visuals.selection.stroke.color = control;
    visuals
}

fn install_font(ctx: &egui::Context) {
    let candidates = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"];
    let Some(bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else {
        return;
    };
    ctx.add_font(egui::FontInsert::new(
        "stack-bold",
        egui::FontData::from_owned(bytes),
        vec![egui::InsertFontFamily {
            family: egui::FontFamily::Proportional,
            priority: egui::FontPriority::Highest,
        }],
    ));
}

fn group(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).strong());
    Frame::NONE
        .fill(Color32::from_rgb(0x3d, 0x12, 0x10))
        .stroke(Stroke::new(1.0, Color32::from_rgb(0x8e, 0x32, 0x24)))
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add(ui);
        });
}

fn short_path(path: &str) -> String {
    if path.is_empty() {
        return "Aucune image".to_string();
    }
    std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn ensure_extension(path: PathBuf, extension: &str) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension(extension)
    }
}

/// Place la texture à sa taille native, centrée. Le zoom reste uniforme :
/// un changement de fenêtre n'étire pas les pixels en attendant le nouveau rendu.
fn preview_dest(area: Rect, tex_w: i32, tex_h: i32, ppp: f32) -> Rect {
    if tex_w <= 0 || tex_h <= 0 || ppp <= 0.0 {
        return area;
    }
    let size = vec2(tex_w as f32 / ppp, tex_h as f32 / ppp);
    if (size.x - area.width()).abs() <= 0.5 && (size.y - area.height()).abs() <= 0.5 {
        area
    } else {
        Rect::from_center_size(area.center(), size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_garde_le_ratio_quand_la_vue_change() {
        let area = Rect::from_min_size(pos2(10.0, 20.0), vec2(400.0, 200.0));
        let dest = preview_dest(area, 200, 200, 1.0);
        assert!((dest.width() / dest.height() - 1.0).abs() < 1.0e-4);
        assert!((dest.center().x - area.center().x).abs() < 1.0e-3);
        assert!((dest.center().y - area.center().y).abs() < 1.0e-3);
        assert!((dest.width() - 200.0).abs() < 1.0e-3);
    }

    #[test]
    fn preview_remplit_la_vue_quand_la_taille_correspond() {
        let area = Rect::from_min_size(pos2(0.0, 0.0), vec2(320.0, 180.0));
        let dest = preview_dest(area, 640, 360, 2.0);
        assert_eq!(dest, area);
    }

    #[test]
    fn preview_ne_deforme_pas_un_cadre_portrait() {
        let area = Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 200.0));
        let dest = preview_dest(area, 100, 200, 1.0);
        assert!((dest.width() / dest.height() - 0.5).abs() < 1.0e-4);
    }
}
