pub mod model;
pub mod ninepatch;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, RgbaImage};

pub use ninepatch::{Insets, NinePatchCoords, PixelCoords};

static EMBEDDED_MODEL: &[u8] = include_bytes!("../ninepatch_model.safetensors");

/// Holds a loaded model ready for inference. Create once, predict many.
pub struct Slicer {
    model: model::NinePatchNet,
    device: Device,
}

impl Slicer {
    /// Load the embedded model with automatic device selection (Metal > CPU).
    pub fn new() -> Result<Self> {
        let device = Device::new_cuda(0)
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu);
        Self::with_device(device)
    }

    /// Load the embedded model on a specific device.
    pub fn with_device(device: Device) -> Result<Self> {
        let vb =
            VarBuilder::from_buffered_safetensors(EMBEDDED_MODEL.to_vec(), DType::F32, &device)?;
        let model = model::NinePatchNet::load(vb).context("failed to load model")?;
        Ok(Self { model, device })
    }

    /// Load the embedded model, forcing CPU.
    pub fn cpu() -> Result<Self> {
        Self::with_device(Device::Cpu)
    }

    /// Predict nine-patch slice coordinates for an image.
    pub fn predict(&self, img: &DynamicImage) -> Result<NinePatchCoords> {
        let input = preprocess(img, &self.device)?;
        let output = self.model.forward(&input)?;
        let raw: Vec<f32> = output.squeeze(0)?.to_vec1()?;

        Ok(NinePatchCoords {
            left: raw[0],
            right: raw[1],
            top: raw[2],
            bottom: raw[3],
        })
    }
}

/// Render a visualization overlay showing slice lines on the image.
pub fn visualize(img: &DynamicImage, coords: &NinePatchCoords) -> RgbaImage {
    ninepatch::visualize(img, coords)
}

/// Stretch an image using nine-patch coordinates to a new size.
pub fn stretch(
    img: &DynamicImage,
    coords: &NinePatchCoords,
    width: u32,
    height: u32,
) -> Result<RgbaImage> {
    ninepatch::stretch(img, coords, width, height)
}

fn preprocess(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
    let resized = rgb.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let data: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            p.0.iter()
                .enumerate()
                .map(|(c, &v)| (v as f32 / 255.0 - mean[c]) / std[c])
        })
        .collect();

    let tensor = Tensor::from_vec(data, (224, 224, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?;

    Ok(tensor)
}
