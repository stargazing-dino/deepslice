# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "pillow", "numpy"]
# ///
"""
9-Patch Coordinate Predictor - Inference Script

Predicts stretchable region coordinates from UI element images.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model(model_path: Path, device: str) -> torch.jit.ScriptModule:
    """Load the TorchScript model."""
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize to 224x224
    img = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Add batch dimension
    return tensor.unsqueeze(0)


def predict(model: torch.jit.ScriptModule, image: Image.Image, device: str) -> dict:
    """Run inference and return coordinates."""
    tensor = preprocess(image).to(device)

    with torch.no_grad():
        output = model(tensor)

    # Output is [batch, 4] with values in [0, 1]
    coords = output[0].cpu().numpy()

    return {
        "stretch_left": float(coords[0]),
        "stretch_right": float(coords[1]),
        "stretch_top": float(coords[2]),
        "stretch_bottom": float(coords[3]),
    }


def coords_to_pixels(coords: dict, width: int, height: int) -> dict:
    """Convert normalized coords to pixel values."""
    return {
        "left": int(coords["stretch_left"] * width),
        "right": int(coords["stretch_right"] * width),
        "top": int(coords["stretch_top"] * height),
        "bottom": int(coords["stretch_bottom"] * height),
    }


def visualize(image: Image.Image, coords: dict, output_path: Path = None) -> Image.Image:
    """Draw 9-patch lines on the image."""
    from PIL import ImageDraw

    viz = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(viz)

    w, h = image.size
    px = coords_to_pixels(coords, w, h)

    # Draw vertical lines (left and right boundaries)
    draw.line([(px["left"], 0), (px["left"], h)], fill=(255, 0, 0, 200), width=2)
    draw.line([(px["right"], 0), (px["right"], h)], fill=(255, 0, 0, 200), width=2)

    # Draw horizontal lines (top and bottom boundaries)
    draw.line([(0, px["top"]), (w, px["top"])], fill=(0, 255, 0, 200), width=2)
    draw.line([(0, px["bottom"]), (w, px["bottom"])], fill=(0, 255, 0, 200), width=2)

    if output_path:
        viz.save(output_path)
        print(f"Saved visualization to {output_path}")

    return viz


def nine_patch_stretch(image: Image.Image, coords: dict, new_width: int, new_height: int) -> Image.Image:
    """Stretch image using 9-patch coordinates."""
    w, h = image.size
    px = coords_to_pixels(coords, w, h)

    left, right = px["left"], px["right"]
    top, bottom = px["top"], px["bottom"]

    # Calculate new stretch region sizes
    new_stretch_w = new_width - left - (w - right)
    new_stretch_h = new_height - top - (h - bottom)

    if new_stretch_w < 1 or new_stretch_h < 1:
        raise ValueError("Target size too small for 9-patch stretch")

    # Create output image
    result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

    # Extract the 9 regions from source
    regions = {
        "tl": image.crop((0, 0, left, top)),
        "t": image.crop((left, 0, right, top)),
        "tr": image.crop((right, 0, w, top)),
        "l": image.crop((0, top, left, bottom)),
        "c": image.crop((left, top, right, bottom)),
        "r": image.crop((right, top, w, bottom)),
        "bl": image.crop((0, bottom, left, h)),
        "b": image.crop((left, bottom, right, h)),
        "br": image.crop((right, bottom, w, h)),
    }

    # Calculate new positions
    new_right = left + new_stretch_w
    new_bottom = top + new_stretch_h

    # Paste corners (no stretching)
    result.paste(regions["tl"], (0, 0))
    result.paste(regions["tr"], (new_right, 0))
    result.paste(regions["bl"], (0, new_bottom))
    result.paste(regions["br"], (new_right, new_bottom))

    # Paste and stretch edges
    if regions["t"].size[0] > 0 and regions["t"].size[1] > 0:
        t_stretched = regions["t"].resize((new_stretch_w, top), Image.Resampling.LANCZOS)
        result.paste(t_stretched, (left, 0))

    if regions["b"].size[0] > 0 and regions["b"].size[1] > 0:
        b_stretched = regions["b"].resize((new_stretch_w, h - bottom), Image.Resampling.LANCZOS)
        result.paste(b_stretched, (left, new_bottom))

    if regions["l"].size[0] > 0 and regions["l"].size[1] > 0:
        l_stretched = regions["l"].resize((left, new_stretch_h), Image.Resampling.LANCZOS)
        result.paste(l_stretched, (0, top))

    if regions["r"].size[0] > 0 and regions["r"].size[1] > 0:
        r_stretched = regions["r"].resize((w - right, new_stretch_h), Image.Resampling.LANCZOS)
        result.paste(r_stretched, (new_right, top))

    # Paste and stretch center
    if regions["c"].size[0] > 0 and regions["c"].size[1] > 0:
        c_stretched = regions["c"].resize((new_stretch_w, new_stretch_h), Image.Resampling.LANCZOS)
        result.paste(c_stretched, (left, top))

    return result


def main():
    parser = argparse.ArgumentParser(description="9-Patch Coordinate Predictor")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--model", type=Path, default=Path("ninepatch_model.pt"), help="Model path")
    parser.add_argument("--visualize", action="store_true", help="Show visualization window")
    parser.add_argument("--vis-output", type=Path, metavar="PATH", help="Save visualization to file")
    parser.add_argument("--stretch", type=int, nargs=2, metavar=("W", "H"), help="Stretch to new dimensions")
    parser.add_argument("--output", "-o", type=Path, metavar="PATH", help="Output path for stretched image")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, args.device)

    # Load image
    print(f"Processing {args.image}...")
    image = Image.open(args.image).convert("RGBA")
    w, h = image.size

    # Predict
    coords = predict(model, image, args.device)
    px = coords_to_pixels(coords, w, h)

    print(f"\nImage size: {w}x{h}")
    print(f"\nNormalized coordinates:")
    print(f"  stretch_left:   {coords['stretch_left']:.4f}")
    print(f"  stretch_right:  {coords['stretch_right']:.4f}")
    print(f"  stretch_top:    {coords['stretch_top']:.4f}")
    print(f"  stretch_bottom: {coords['stretch_bottom']:.4f}")

    print(f"\nPixel coordinates:")
    print(f"  left:   {px['left']}")
    print(f"  right:  {px['right']}")
    print(f"  top:    {px['top']}")
    print(f"  bottom: {px['bottom']}")

    print(f"\nInsets (for game engines):")
    print(f"  left:   {px['left']}")
    print(f"  right:  {w - px['right']}")
    print(f"  top:    {px['top']}")
    print(f"  bottom: {h - px['bottom']}")

    # Visualization
    if args.visualize or args.vis_output:
        viz = visualize(image, coords, args.vis_output)
        if args.visualize:
            viz.show()

    # Stretch
    if args.stretch:
        new_w, new_h = args.stretch
        print(f"\nStretching to {new_w}x{new_h}...")
        stretched = nine_patch_stretch(image, coords, new_w, new_h)

        output_path = args.output or args.image.with_stem(f"{args.image.stem}_stretched")
        stretched.save(output_path)
        print(f"Saved stretched image to {output_path}")


if __name__ == "__main__":
    main()
