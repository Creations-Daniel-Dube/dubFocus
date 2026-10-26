use std::path::Path;
use std::process::Command;

fn main() {
    embed_windows_icon();

    let manifest = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let kernel_root = manifest.join("kernels").join("src");
    rerun_tree(&kernel_root);

    let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    let ptx_path = out_dir.join("kernels.ptx");
    let source = kernel_root.join("lib.rs");
    let emit = format!("asm={}", ptx_path.display());

    let status = Command::new("rustup")
        .args([
            "run",
            "nightly",
            "rustc",
            "--edition",
            "2024",
            "--crate-name",
            "dubfocus_kernels",
            "--crate-type",
            "rlib",
            "--target",
            "nvptx64-nvidia-cuda",
            "-C",
            "panic=abort",
            "-C",
            "opt-level=3",
            "-C",
            "overflow-checks=off",
            "-C",
            "debug-assertions=off",
            "--emit",
            &emit,
        ])
        .arg(&source)
        .status()
        .expect("rustup nightly est requis pour compiler les noyaux GPU");

    if !status.success() {
        panic!("la compilation PTX des noyaux a échoué");
    }

    println!("cargo:rustc-env=DUBFOCUS_PTX={}", ptx_path.display());
}

fn embed_windows_icon() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }
    println!("cargo:rerun-if-changed=images/dubFocus.ico");
    let mut res = winresource::WindowsResource::new();
    res.set_icon("images/dubFocus.ico");
    res.compile().expect("intégration de images/dubFocus.ico dans l'exécutable");
}

fn rerun_tree(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
