use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use image::GenericImageView;

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

fn main() -> Result<()> {
    let args = Args::parse();

    let slicer = if args.cpu {
        deepslice::Slicer::cpu()?
    } else {
        deepslice::Slicer::new()?
    };

    let img = image::open(&args.image)
        .with_context(|| format!("failed to open {}", args.image.display()))?;
    let (w, h) = img.dimensions();

    let coords = slicer.predict(&img)?;
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
        let viz = deepslice::visualize(&img, &coords);
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
        let stretched = deepslice::stretch(&img, &coords, new_w, new_h)?;
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
