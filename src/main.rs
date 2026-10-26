#![cfg_attr(windows, windows_subsystem = "windows")]

mod color;
mod deconv;
mod error;
mod gpu;
mod io;
mod psf;
mod reference;
mod settings;
mod ui;

fn main() -> eframe::Result {
    ui::run()
}
