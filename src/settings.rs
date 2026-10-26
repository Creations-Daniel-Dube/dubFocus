use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::io::SaveKind;
use crate::psf::PsfKind;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub last_dir: String,
    pub disk: bool,
    pub radius: f32,
    pub iterations: u32,
    pub log_k: f32,
    pub damping: f32,
    pub dering: f32,
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
    pub save_kind: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            last_dir: String::new(),
            disk: true,
            radius: 1.2,
            iterations: 20,
            log_k: -2.5,
            damping: 0.02,
            dering: 0.4,
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
            save_kind: "png16".to_string(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let Ok(text) = std::fs::read_to_string(settings_path()) else {
            return Self::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) {
        let path = settings_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    pub fn kind(&self) -> PsfKind {
        if self.disk { PsfKind::Disk } else { PsfKind::Gaussian }
    }

    pub fn save_kind(&self) -> SaveKind {
        match self.save_kind.as_str() {
            "png8" => SaveKind::Png8,
            "tiff16" => SaveKind::Tiff16,
            "tiff32" => SaveKind::Tiff32,
            "jpeg" => SaveKind::Jpeg,
            _ => SaveKind::Png16,
        }
    }

    pub fn set_save_kind(&mut self, kind: SaveKind) {
        self.save_kind = match kind {
            SaveKind::Png8 => "png8",
            SaveKind::Png16 => "png16",
            SaveKind::Tiff16 => "tiff16",
            SaveKind::Tiff32 => "tiff32",
            SaveKind::Jpeg => "jpeg",
        }
        .to_string();
    }
}

fn settings_path() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("APPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
        base.join("dubFocus").join("settings.json")
    } else {
        let base = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|home| PathBuf::from(home).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("dubFocus").join("settings.json")
    }
}
