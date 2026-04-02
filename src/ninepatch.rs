use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};

#[derive(Debug, Clone)]
pub struct NinePatchCoords {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl NinePatchCoords {
    pub fn to_pixels(&self, width: u32, height: u32) -> PixelCoords {
        PixelCoords {
            left: (self.left * width as f32) as u32,
            right: (self.right * width as f32) as u32,
            top: (self.top * height as f32) as u32,
            bottom: (self.bottom * height as f32) as u32,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PixelCoords {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelCoords {
    pub fn insets(&self) -> Insets {
        Insets {
            left: self.left,
            right: self.width - self.right,
            top: self.top,
            bottom: self.height - self.bottom,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Insets {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

pub fn visualize(img: &DynamicImage, coords: &NinePatchCoords) -> RgbaImage {
    let mut viz = img.to_rgba8();
    let (w, h) = img.dimensions();
    let px = coords.to_pixels(w, h);

    let red = Rgba([255, 0, 0, 200]);
    let green = Rgba([0, 255, 0, 200]);

    // Vertical lines (left/right boundaries)
    for y in 0..h {
        if px.left < w { viz.put_pixel(px.left, y, red); }
        if px.left > 0 { viz.put_pixel(px.left.saturating_sub(1), y, red); }
        if px.right < w { viz.put_pixel(px.right, y, red); }
        if px.right > 0 { viz.put_pixel(px.right.saturating_sub(1), y, red); }
    }

    // Horizontal lines (top/bottom boundaries)
    for x in 0..w {
        if px.top < h { viz.put_pixel(x, px.top, green); }
        if px.top > 0 { viz.put_pixel(x, px.top.saturating_sub(1), green); }
        if px.bottom < h { viz.put_pixel(x, px.bottom, green); }
        if px.bottom > 0 { viz.put_pixel(x, px.bottom.saturating_sub(1), green); }
    }

    viz
}

pub fn stretch(img: &DynamicImage, coords: &NinePatchCoords, new_w: u32, new_h: u32) -> anyhow::Result<RgbaImage> {
    let (w, h) = img.dimensions();
    let px = coords.to_pixels(w, h);

    let left = px.left;
    let right = px.right;
    let top = px.top;
    let bottom = px.bottom;

    let corner_w = left + (w - right);
    let corner_h = top + (h - bottom);
    anyhow::ensure!(new_w > corner_w, "target width {} too small (need > {})", new_w, corner_w);
    anyhow::ensure!(new_h > corner_h, "target height {} too small (need > {})", new_h, corner_h);

    let stretch_w = new_w - corner_w;
    let stretch_h = new_h - corner_h;
    let new_right = left + stretch_w;
    let new_bottom = top + stretch_h;

    let mut result: RgbaImage = ImageBuffer::new(new_w, new_h);

    let crop = |x: u32, y: u32, cw: u32, ch: u32| -> RgbaImage {
        img.crop_imm(x, y, cw, ch).to_rgba8()
    };

    let resize = |src: &RgbaImage, tw: u32, th: u32| -> RgbaImage {
        image::imageops::resize(src, tw, th, image::imageops::FilterType::Lanczos3)
    };

    // Corners (no stretch)
    copy_region(&mut result, &crop(0, 0, left, top), 0, 0);
    copy_region(&mut result, &crop(right, 0, w - right, top), new_right, 0);
    copy_region(&mut result, &crop(0, bottom, left, h - bottom), 0, new_bottom);
    copy_region(&mut result, &crop(right, bottom, w - right, h - bottom), new_right, new_bottom);

    // Edges
    let src_stretch_w = right - left;
    let src_stretch_h = bottom - top;

    if src_stretch_w > 0 && top > 0 {
        let t = crop(left, 0, src_stretch_w, top);
        copy_region(&mut result, &resize(&t, stretch_w, top), left, 0);
    }
    if src_stretch_w > 0 && h - bottom > 0 {
        let b = crop(left, bottom, src_stretch_w, h - bottom);
        copy_region(&mut result, &resize(&b, stretch_w, h - bottom), left, new_bottom);
    }
    if src_stretch_h > 0 && left > 0 {
        let l = crop(0, top, left, src_stretch_h);
        copy_region(&mut result, &resize(&l, left, stretch_h), 0, top);
    }
    if src_stretch_h > 0 && w - right > 0 {
        let r = crop(right, top, w - right, src_stretch_h);
        copy_region(&mut result, &resize(&r, w - right, stretch_h), new_right, top);
    }

    // Center
    if src_stretch_w > 0 && src_stretch_h > 0 {
        let c = crop(left, top, src_stretch_w, src_stretch_h);
        copy_region(&mut result, &resize(&c, stretch_w, stretch_h), left, top);
    }

    Ok(result)
}

fn copy_region(dest: &mut RgbaImage, src: &RgbaImage, dx: u32, dy: u32) {
    for (x, y, pixel) in src.enumerate_pixels() {
        let tx = dx + x;
        let ty = dy + y;
        if tx < dest.width() && ty < dest.height() {
            dest.put_pixel(tx, ty, *pixel);
        }
    }
}
