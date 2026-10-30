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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("probe") {
        std::process::exit(print_probe());
    }
    ui::run()
}

fn print_probe() -> i32 {
    let (text, code) = match gpu::probe_gpu() {
        Ok(info) => (gpu::probe_document(&info), 0),
        Err(err) => (format!("{{\"ev\":\"probe\",\"error\":\"{}\"}}", err.replace('"', "'")), 1),
    };
    write_console(&text);
    code
}

fn write_console(text: &str) {
    use std::io::Write;
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn AttachConsole(pid: u32) -> i32;
            fn GetStdHandle(std: u32) -> isize;
        }
        unsafe { AttachConsole(u32::MAX) };
        let handle = unsafe { GetStdHandle(0xFFFF_FFF5) };
        if handle == 0 || handle == -1 {
            if let Ok(mut console) = std::fs::OpenOptions::new().write(true).open("CONOUT$") {
                let _ = writeln!(console, "{text}");
                let _ = console.flush();
                return;
            }
        }
    }
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "{text}");
    let _ = stdout.flush();
}
