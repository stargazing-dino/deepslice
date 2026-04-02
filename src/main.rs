mod model;
mod ninepatch;

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use image::GenericImageView;

static EMBEDDED_MODEL: &[u8] = include_bytes!("../ninepatch_model.safetensors");

use crate::ninepatch::NinePatchCoords;

#[derive(Parser)]
#[command(name = "deepslice", about = "Auto nine-patch slicer powered by ML")]
struct Args {
    /// Input image path
    image: PathBuf,

    /// Stretch to WxH (e.g. 400x200)
    #[arg(long, value_parser = parse_dimensions)]
    stretch: Option<(u32, u32)>,

    /// Output path for stretched or visualization image
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Save visualization overlay showing slice lines
    #[arg(long)]
    visualize: bool,

    /// Output coordinates as JSON
    #[arg(long)]
    json: bool,

    /// Run on CPU (default: auto-detect)
    #[arg(long)]
    cpu: bool,
}

fn parse_dimensions(s: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return Err("expected WxH format (e.g. 400x200)".into());
    }
    let w = parts[0].parse().map_err(|_| "invalid width")?;
    let h = parts[1].parse().map_err(|_| "invalid height")?;
    Ok((w, h))
}

fn preprocess(img: &image::DynamicImage, device: &Device) -> Result<Tensor> {
    let rgb = image::DynamicImage::ImageRgb8(img.to_rgb8());
    let resized = rgb.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let data: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            p.0.iter().enumerate().map(|(c, &v)| {
                (v as f32 / 255.0 - mean[c]) / std[c]
            })
        })
        .collect();

    let tensor = Tensor::from_vec(data, (224, 224, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?;

    Ok(tensor)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    };

    // Load model
    let vb = VarBuilder::from_buffered_safetensors(EMBEDDED_MODEL.to_vec(), DType::F32, &device)?;
    let model = model::NinePatchNet::load(vb).context("failed to load model")?;

    // Load image
    let img = image::open(&args.image)
        .with_context(|| format!("failed to open {}", args.image.display()))?;
    let (w, h) = img.dimensions();

    // Run inference
    let input = preprocess(&img, &device)?;
    let output = model.forward(&input)?;
    let coords_raw: Vec<f32> = output.squeeze(0)?.to_vec1()?;

    let coords = NinePatchCoords {
        left: coords_raw[0],
        right: coords_raw[1],
        top: coords_raw[2],
        bottom: coords_raw[3],
    };

    let px = coords.to_pixels(w, h);

    if args.json {
        let out = serde_json::json!({
            "image": args.image.display().to_string(),
            "size": { "width": w, "height": h },
            "normalized": {
                "left": coords.left,
                "right": coords.right,
                "top": coords.top,
                "bottom": coords.bottom,
            },
            "pixels": {
                "left": px.left,
                "right": px.right,
                "top": px.top,
                "bottom": px.bottom,
            },
            "insets": px.insets(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("Image: {}x{}", w, h);
        println!();
        println!("Normalized:  L={:.4}  R={:.4}  T={:.4}  B={:.4}", coords.left, coords.right, coords.top, coords.bottom);
        println!("Pixels:      L={}  R={}  T={}  B={}", px.left, px.right, px.top, px.bottom);
        let insets = px.insets();
        println!("Insets:      L={}  R={}  T={}  B={}", insets.left, insets.right, insets.top, insets.bottom);
    }

    if args.visualize {
        let viz = ninepatch::visualize(&img, &coords);
        let out_path = args.output.clone().unwrap_or_else(|| {
            let stem = args.image.file_stem().unwrap().to_string_lossy();
            args.image.with_file_name(format!("{stem}_slices.png"))
        });
        viz.save(&out_path)?;
        if !args.json {
            println!("\nVisualization saved to {}", out_path.display());
        }
    }

    if let Some((new_w, new_h)) = args.stretch {
        let stretched = ninepatch::stretch(&img, &coords, new_w, new_h)?;
        let out_path = args.output.unwrap_or_else(|| {
            let stem = args.image.file_stem().unwrap().to_string_lossy();
            args.image.with_file_name(format!("{stem}_stretched.png"))
        });
        stretched.save(&out_path)?;
        if !args.json {
            println!("\nStretched {}x{} -> {}", new_w, new_h, out_path.display());
        }
    }

    Ok(())
}
