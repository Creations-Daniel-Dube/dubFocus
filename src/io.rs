use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use image::{ImageBuffer, ImageReader, Luma, Rgb};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::TiffEncoder;
use tiff::encoder::colortype::{Gray16, Gray32Float, RGB16, RGB32Float};
use tiff::ColorType;

use crate::color::{encode_u16, encode_u8, level_linear};
use crate::error::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveKind {
    Png8,
    Png16,
    Tiff16,
    Tiff32,
    Jpeg,
}

impl SaveKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Png8 => "PNG 8 bits",
            Self::Png16 => "PNG 16 bits",
            Self::Tiff16 => "TIFF 16 bits",
            Self::Tiff32 => "TIFF 32 bits",
            Self::Jpeg => "JPEG",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Png8 | Self::Png16 => "png",
            Self::Tiff16 | Self::Tiff32 => "tif",
            Self::Jpeg => "jpg",
        }
    }

    pub fn all() -> [SaveKind; 5] {
        [Self::Png8, Self::Png16, Self::Tiff16, Self::Tiff32, Self::Jpeg]
    }
}

#[derive(Clone, Debug)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub bit_depth: u8,
    pub samples: Vec<f32>,
}

impl ImageData {
    pub fn suggest_srgb(&self) -> bool {
        self.bit_depth <= 8
    }
}

pub fn load_path(path: &Path) -> Result<ImageData, Error> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "tif" | "tiff" => load_tiff(path),
        "jpg" | "jpeg" | "png" => load_raster(path),
        _ => Err(Error::new("format non pris en charge (jpg, png ou tiff)")),
    }
}

pub fn save_processed(
    path: &Path,
    width: u32,
    height: u32,
    channels: usize,
    linear: &[f32],
    black: f32,
    white: f32,
    gamma: f32,
    kind: SaveKind,
) -> Result<(), Error> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| Error::new(format!("dossier de sortie : {err}")))?;
    }
    let leveled: Vec<f32> = linear.iter().copied().map(|value| level_linear(value, black, white, gamma)).collect();
    match kind {
        SaveKind::Png8 | SaveKind::Jpeg => save_u8(path, width, height, channels, &leveled, kind),
        SaveKind::Png16 | SaveKind::Tiff16 => save_u16(path, width, height, channels, &leveled, kind),
        SaveKind::Tiff32 => save_f32(path, width, height, channels, &leveled),
    }
}

fn load_raster(path: &Path) -> Result<ImageData, Error> {
    let reader = ImageReader::open(path)
        .map_err(|err| Error::new(format!("lecture : {err}")))?
        .with_guessed_format()
        .map_err(|err| Error::new(format!("format : {err}")))?;
    let image = reader.decode().map_err(|err| Error::new(format!("décodage : {err}")))?;
    match image {
        image::DynamicImage::ImageLuma8(buffer) => pack_u8(buffer.width(), buffer.height(), 1, buffer.as_raw()),
        image::DynamicImage::ImageRgb8(buffer) => pack_u8(buffer.width(), buffer.height(), 3, buffer.as_raw()),
        image::DynamicImage::ImageRgba8(buffer) => pack_u8(buffer.width(), buffer.height(), 4, buffer.as_raw()),
        image::DynamicImage::ImageLuma16(buffer) => pack_u16(buffer.width(), buffer.height(), 1, buffer.as_raw()),
        image::DynamicImage::ImageRgb16(buffer) => pack_u16(buffer.width(), buffer.height(), 3, buffer.as_raw()),
        image::DynamicImage::ImageRgba16(buffer) => pack_u16(buffer.width(), buffer.height(), 4, buffer.as_raw()),
        other => {
            let buffer = other.to_rgb8();
            pack_u8(buffer.width(), buffer.height(), 3, buffer.as_raw())
        }
    }
}

fn load_tiff(path: &Path) -> Result<ImageData, Error> {
    let file = File::open(path).map_err(|err| Error::new(format!("lecture : {err}")))?;
    let mut decoder = Decoder::new(BufReader::new(file)).map_err(|err| Error::new(format!("tiff : {err}")))?;
    let (width, height) = decoder.dimensions().map_err(|err| Error::new(format!("tiff : {err}")))?;
    let color = decoder.colortype().map_err(|err| Error::new(format!("tiff : {err}")))?;
    let decoded = decoder.read_image().map_err(|err| Error::new(format!("tiff : {err}")))?;
    let bit_depth = match color {
        ColorType::Gray(bits) | ColorType::RGB(bits) | ColorType::RGBA(bits) | ColorType::GrayA(bits) => bits,
        ColorType::Multiband { bit_depth, .. } => bit_depth,
        _ => return Err(Error::new("tiff : type colorimétrique non pris en charge")),
    };
    let pixels = width as usize * height as usize;
    match decoded {
        DecodingResult::U8(values) => finish_tiff(width, height, bit_depth, pixels, normalize_u8(&values)),
        DecodingResult::U16(values) => finish_tiff(width, height, 16, pixels, normalize_u16(&values)),
        DecodingResult::F32(values) => finish_tiff(width, height, 32, pixels, values),
        _ => Err(Error::new("tiff : profondeur non prise en charge")),
    }
}

fn finish_tiff(width: u32, height: u32, bit_depth: u8, pixels: usize, samples: Vec<f32>) -> Result<ImageData, Error> {
    if pixels == 0 || samples.len() % pixels != 0 {
        return Err(Error::new("tiff : dimensions incohérentes"));
    }
    let raw_channels = samples.len() / pixels;
    let channels = if raw_channels >= 3 { 3 } else { 1 };
    let packed = if raw_channels == channels {
        samples
    } else {
        let mut packed = Vec::with_capacity(pixels * channels);
        for pixel in 0..pixels {
            for channel in 0..channels {
                packed.push(samples[pixel * raw_channels + channel]);
            }
        }
        packed
    };
    Ok(ImageData {
        width,
        height,
        channels,
        bit_depth,
        samples: packed,
    })
}

fn pack_u8(width: u32, height: u32, channels: usize, raw: &[u8]) -> Result<ImageData, Error> {
    let stored = if channels >= 3 { 3 } else { 1 };
    Ok(ImageData {
        width,
        height,
        channels: stored,
        bit_depth: 8,
        samples: take_channels(&normalize_u8(raw), channels, stored),
    })
}

fn pack_u16(width: u32, height: u32, channels: usize, raw: &[u16]) -> Result<ImageData, Error> {
    let stored = if channels >= 3 { 3 } else { 1 };
    Ok(ImageData {
        width,
        height,
        channels: stored,
        bit_depth: 16,
        samples: take_channels(&normalize_u16(raw), channels, stored),
    })
}

fn take_channels(samples: &[f32], channels: usize, stored: usize) -> Vec<f32> {
    if channels == stored {
        return samples.to_vec();
    }
    let pixels = samples.len() / channels;
    let mut packed = Vec::with_capacity(pixels * stored);
    for pixel in 0..pixels {
        for channel in 0..stored {
            packed.push(samples[pixel * channels + channel]);
        }
    }
    packed
}

fn normalize_u8(values: &[u8]) -> Vec<f32> {
    values.iter().map(|value| *value as f32 / 255.0).collect()
}

fn normalize_u16(values: &[u16]) -> Vec<f32> {
    values.iter().map(|value| *value as f32 / 65535.0).collect()
}

fn save_u8(path: &Path, width: u32, height: u32, channels: usize, leveled: &[f32], kind: SaveKind) -> Result<(), Error> {
    let encoded: Vec<u8> = leveled.iter().copied().map(encode_u8).collect();
    if channels <= 1 {
        let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, encoded).ok_or_else(|| Error::new("encodage 8 bits"))?;
        if kind == SaveKind::Jpeg {
            buffer.save(path).map_err(|err| Error::new(format!("écriture : {err}")))?;
        } else {
            buffer.save(path).map_err(|err| Error::new(format!("écriture : {err}")))?;
        }
        return Ok(());
    }
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, encoded).ok_or_else(|| Error::new("encodage 8 bits"))?;
    if kind == SaveKind::Jpeg {
        let file = File::create(path).map_err(|err| Error::new(format!("écriture : {err}")))?;
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(BufWriter::new(file), 95);
        encoder
            .encode(buffer.as_raw(), width, height, image::ExtendedColorType::Rgb8)
            .map_err(|err| Error::new(format!("jpeg : {err}")))?;
        Ok(())
    } else {
        buffer.save(path).map_err(|err| Error::new(format!("écriture : {err}")))
    }
}

fn save_u16(path: &Path, width: u32, height: u32, channels: usize, leveled: &[f32], kind: SaveKind) -> Result<(), Error> {
    let encoded: Vec<u16> = leveled.iter().copied().map(encode_u16).collect();
    if kind == SaveKind::Png16 {
        if channels <= 1 {
            let buffer = ImageBuffer::<Luma<u16>, _>::from_raw(width, height, encoded).ok_or_else(|| Error::new("encodage 16 bits"))?;
            buffer.save(path).map_err(|err| Error::new(format!("écriture : {err}")))
        } else {
            let buffer = ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, encoded).ok_or_else(|| Error::new("encodage 16 bits"))?;
            buffer.save(path).map_err(|err| Error::new(format!("écriture : {err}")))
        }
    } else {
        write_tiff_u16(path, width, height, channels > 1, &encoded)
    }
}

fn save_f32(path: &Path, width: u32, height: u32, channels: usize, leveled: &[f32]) -> Result<(), Error> {
    let file = File::create(path).map_err(|err| Error::new(format!("écriture : {err}")))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file)).map_err(|err| Error::new(format!("tiff : {err}")))?;
    if channels > 1 {
        encoder
            .write_image::<RGB32Float>(width, height, leveled)
            .map_err(|err| Error::new(format!("tiff : {err}")))?;
    } else {
        encoder
            .write_image::<Gray32Float>(width, height, leveled)
            .map_err(|err| Error::new(format!("tiff : {err}")))?;
    }
    Ok(())
}

fn write_tiff_u16(path: &Path, width: u32, height: u32, rgb: bool, data: &[u16]) -> Result<(), Error> {
    let file = File::create(path).map_err(|err| Error::new(format!("écriture : {err}")))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file)).map_err(|err| Error::new(format!("tiff : {err}")))?;
    if rgb {
        encoder
            .write_image::<RGB16>(width, height, data)
            .map_err(|err| Error::new(format!("tiff : {err}")))?;
    } else {
        encoder
            .write_image::<Gray16>(width, height, data)
            .map_err(|err| Error::new(format!("tiff : {err}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png8_and_tiff32_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dubfocus-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("sample.png");
        let tiff = dir.join("sample.tif");
        let source = vec![0.0f32, 0.25, 0.5, 1.0, 0.0, 1.0];
        save_processed(&png, 2, 1, 3, &source, 0.0, 1.0, 1.0, SaveKind::Png8).unwrap();
        save_processed(&tiff, 2, 1, 3, &source, 0.0, 1.0, 1.0, SaveKind::Tiff32).unwrap();
        let loaded = load_path(&png).unwrap();
        assert_eq!((loaded.width, loaded.height, loaded.bit_depth, loaded.channels), (2, 1, 8, 3));
        let float = load_path(&tiff).unwrap();
        assert_eq!(float.bit_depth, 32);
        assert_eq!(float.channels, 3);
        assert!((float.samples[3] - 1.0).abs() < 1.0e-4);
        let bytes = std::fs::read(&tiff).unwrap();
        let mut decoder = Decoder::new(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(decoder.dimensions().unwrap(), (2, 1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
