"""
Synthetic 9-patch training data generator.

Generates UI elements (cards, pills, wide cards, header cards) with ground truth
9-patch stretch coordinates. Uses a curriculum of increasing complexity stages.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm


@dataclass
class NinePatchCoords:
    """Normalized 9-patch coordinates in [0,1] range."""
    stretch_left: float
    stretch_right: float
    stretch_top: float
    stretch_bottom: float

    def to_array(self) -> np.ndarray:
        return np.array([self.stretch_left, self.stretch_right,
                         self.stretch_top, self.stretch_bottom], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'NinePatchCoords':
        return cls(*arr.tolist())

    def to_pixels(self, width: int, height: int) -> 'NinePatchCoords':
        return NinePatchCoords(
            stretch_left=int(self.stretch_left * width),
            stretch_right=int(self.stretch_right * width),
            stretch_top=int(self.stretch_top * height),
            stretch_bottom=int(self.stretch_bottom * height),
        )

    def flip_horizontal(self) -> 'NinePatchCoords':
        return NinePatchCoords(
            stretch_left=1.0 - self.stretch_right,
            stretch_right=1.0 - self.stretch_left,
            stretch_top=self.stretch_top,
            stretch_bottom=self.stretch_bottom,
        )

    def flip_vertical(self) -> 'NinePatchCoords':
        return NinePatchCoords(
            stretch_left=self.stretch_left,
            stretch_right=self.stretch_right,
            stretch_top=1.0 - self.stretch_bottom,
            stretch_bottom=1.0 - self.stretch_top,
        )


class NinePatchGeneratorV2:
    """
    Generates synthetic UI elements with 9-patch ground truth.

    V2 Features:
    - Stacked card layers for depth effects
    - Gradient borders (linear, multi-stop)
    - Inner effects (highlight, shadow, glow)
    - Background variations (gradients, textures)
    - More element types (pills, wide cards, header cards)
    """

    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        self.output_size = output_size

    def generate_sample(self, curriculum_stage: int = 1) -> Tuple[np.ndarray, NinePatchCoords]:
        """Generate a synthetic UI element with ground truth coordinates."""
        element_type = self._select_element_type(curriculum_stage)

        if element_type == 'pill':
            return self._generate_pill(curriculum_stage)
        elif element_type == 'wide_card':
            return self._generate_wide_card(curriculum_stage)
        elif element_type == 'header_card':
            return self._generate_header_card(curriculum_stage)
        else:
            return self._generate_standard_card(curriculum_stage)

    def _select_element_type(self, stage: int) -> str:
        if stage <= 2:
            return 'standard'
        types = ['standard', 'standard', 'standard']
        if stage >= 3:
            types.extend(['pill', 'wide_card'])
        if stage >= 4:
            types.extend(['header_card', 'pill', 'wide_card'])
        return random.choice(types)

    def _generate_standard_card(self, stage: int) -> Tuple[np.ndarray, NinePatchCoords]:
        elem_width = random.randint(80, 180)
        elem_height = random.randint(40, 120)
        corner_radius = self._get_corner_radius(stage)

        num_stack_layers = 0
        stack_offset = 0
        if stage >= 3 and random.random() > 0.5:
            num_stack_layers = random.randint(1, 2 if stage == 3 else 3)
            stack_offset = random.randint(3, 6)

        shadow_pad = 30 if stage >= 2 else 20
        stack_pad = num_stack_layers * stack_offset
        pad_left = shadow_pad
        pad_top = shadow_pad
        pad_right = shadow_pad + stack_pad
        pad_bottom = shadow_pad + stack_pad

        canvas_w = elem_width + pad_left + pad_right
        canvas_h = elem_height + pad_top + pad_bottom

        img = self._create_background(canvas_w, canvas_h, stage)
        main_bbox = (pad_left, pad_top, pad_left + elem_width, pad_top + elem_height)

        if stage >= 2:
            img = self._add_shadow(img, main_bbox, corner_radius, stage)

        if num_stack_layers > 0:
            img = self._add_stacked_layers(
                img, main_bbox, corner_radius, num_stack_layers, stack_offset, stage
            )

        img = self._draw_card(img, main_bbox, corner_radius, stage)

        if stage >= 3:
            img = self._add_inner_effects(img, main_bbox, corner_radius, stage)

        if stage >= 2:
            img = self._add_noise(img, stage)

        coords = self._compute_coords(main_bbox, corner_radius, canvas_w, canvas_h)

        img_rgb = img.convert('RGB').resize(self.output_size, Image.LANCZOS)
        return np.array(img_rgb), coords

    def _generate_pill(self, stage: int) -> Tuple[np.ndarray, NinePatchCoords]:
        elem_width = random.randint(50, 100)
        elem_height = random.randint(24, 36)
        corner_radius = elem_height // 2

        pad = 25
        canvas_w = elem_width + pad * 2
        canvas_h = elem_height + pad * 2

        img = self._create_background(canvas_w, canvas_h, stage)
        bbox = (pad, pad, pad + elem_width, pad + elem_height)

        if stage >= 2:
            img = self._add_shadow(img, bbox, corner_radius, stage)

        img = self._draw_card(img, bbox, corner_radius, stage)

        if stage >= 3:
            img = self._add_inner_effects(img, bbox, corner_radius, stage)

        if stage >= 2:
            img = self._add_noise(img, stage)

        coords = self._compute_coords(bbox, corner_radius, canvas_w, canvas_h)
        img_rgb = img.convert('RGB').resize(self.output_size, Image.LANCZOS)
        return np.array(img_rgb), coords

    def _generate_wide_card(self, stage: int) -> Tuple[np.ndarray, NinePatchCoords]:
        elem_width = random.randint(180, 220)
        elem_height = random.randint(30, 50)
        corner_radius = self._get_corner_radius(stage)
        corner_radius = min(corner_radius, elem_height // 2)

        pad = 25
        canvas_w = elem_width + pad * 2
        canvas_h = elem_height + pad * 2

        img = self._create_background(canvas_w, canvas_h, stage)
        bbox = (pad, pad, pad + elem_width, pad + elem_height)

        if stage >= 2:
            img = self._add_shadow(img, bbox, corner_radius, stage)

        img = self._draw_card(img, bbox, corner_radius, stage)

        if stage >= 3:
            img = self._add_inner_effects(img, bbox, corner_radius, stage)

        if stage >= 2:
            img = self._add_noise(img, stage)

        coords = self._compute_coords(bbox, corner_radius, canvas_w, canvas_h)
        img_rgb = img.convert('RGB').resize(self.output_size, Image.LANCZOS)
        return np.array(img_rgb), coords

    def _generate_header_card(self, stage: int) -> Tuple[np.ndarray, NinePatchCoords]:
        elem_width = random.randint(100, 180)
        elem_height = random.randint(80, 120)
        header_height = random.randint(20, 35)
        corner_radius = self._get_corner_radius(stage)

        pad = 30
        canvas_w = elem_width + pad * 2
        canvas_h = elem_height + pad * 2

        img = self._create_background(canvas_w, canvas_h, stage)
        bbox = (pad, pad, pad + elem_width, pad + elem_height)

        if stage >= 2:
            img = self._add_shadow(img, bbox, corner_radius, stage)

        draw = ImageDraw.Draw(img)
        body_color = self._random_color(alpha=255)
        draw.rounded_rectangle(bbox, radius=corner_radius, fill=body_color)

        header_color = self._random_color(alpha=255)
        header_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        header_draw = ImageDraw.Draw(header_img)
        header_draw.rounded_rectangle(bbox, radius=corner_radius, fill=header_color)
        body_rect = (bbox[0], bbox[1] + header_height, bbox[2], bbox[3])
        header_draw.rectangle(body_rect, fill=body_color)
        img = Image.alpha_composite(img, header_img)

        if random.random() > 0.3:
            draw = ImageDraw.Draw(img)
            border_color = self._random_color(alpha=255)
            draw.rounded_rectangle(bbox, radius=corner_radius, outline=border_color, width=2)

        if stage >= 3:
            img = self._add_inner_effects(img, bbox, corner_radius, stage)

        if stage >= 2:
            img = self._add_noise(img, stage)

        coords = self._compute_coords(bbox, corner_radius, canvas_w, canvas_h)
        img_rgb = img.convert('RGB').resize(self.output_size, Image.LANCZOS)
        return np.array(img_rgb), coords

    def _add_stacked_layers(self, img: Image.Image, main_bbox: Tuple[int, int, int, int],
                            radius: int, num_layers: int, offset: int,
                            stage: int) -> Image.Image:
        x0, y0, x1, y1 = main_bbox

        for i in range(num_layers, 0, -1):
            layer_offset = offset * i
            layer_bbox = (
                x0 + layer_offset,
                y0 + layer_offset,
                x1 + layer_offset,
                y1 + layer_offset
            )

            darkness = 0.85 - (i * 0.1)
            base_color = self._random_color(alpha=255)
            layer_color = tuple(int(c * darkness) for c in base_color[:3]) + (255,)

            img = self._add_shadow(img, layer_bbox, radius, stage, alpha_mult=0.5)

            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle(layer_bbox, radius=radius, fill=layer_color)

            if random.random() > 0.5:
                border_color = tuple(int(c * 0.7) for c in layer_color[:3]) + (255,)
                draw.rounded_rectangle(layer_bbox, radius=radius,
                                       outline=border_color, width=1)

        return img

    def _draw_card(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                   radius: int, stage: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        fill_color = self._random_color(alpha=255)
        draw.rounded_rectangle(bbox, radius=radius, fill=fill_color)

        if random.random() > 0.3:
            if stage >= 3 and random.random() > 0.5:
                img = self._draw_gradient_border(img, bbox, radius, stage)
            else:
                border_color = self._random_color(alpha=255)
                border_width = random.randint(1, 3)
                draw.rounded_rectangle(bbox, radius=radius,
                                       outline=border_color, width=border_width)

        return img

    def _draw_gradient_border(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                              radius: int, stage: int) -> Image.Image:
        x0, y0, x1, y1 = bbox
        border_width = random.randint(2, 4)

        num_stops = random.randint(2, 4)
        colors = [self._random_vibrant_color() for _ in range(num_stops)]
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])

        border_mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(border_mask)
        mask_draw.rounded_rectangle(bbox, radius=radius, fill=255)
        inner_bbox = (x0 + border_width, y0 + border_width,
                      x1 - border_width, y1 - border_width)
        inner_radius = max(0, radius - border_width)
        mask_draw.rounded_rectangle(inner_bbox, radius=inner_radius, fill=0)

        gradient = self._create_gradient(img.size, colors, direction)
        img.paste(gradient, mask=border_mask)

        return img

    def _create_gradient(self, size: Tuple[int, int], colors: List[Tuple[int, int, int]],
                         direction: str) -> Image.Image:
        w, h = size
        gradient = Image.new('RGBA', size)

        if direction == 'horizontal':
            for x in range(w):
                t = x / max(1, w - 1)
                color = self._interpolate_colors(colors, t)
                ImageDraw.Draw(gradient).line([(x, 0), (x, h)], fill=color)
        elif direction == 'vertical':
            for y in range(h):
                t = y / max(1, h - 1)
                color = self._interpolate_colors(colors, t)
                ImageDraw.Draw(gradient).line([(0, y), (w, y)], fill=color)
        else:
            for x in range(w):
                for y in range(h):
                    t = (x + y) / max(1, w + h - 2)
                    color = self._interpolate_colors(colors, t)
                    gradient.putpixel((x, y), color)

        return gradient

    def _interpolate_colors(self, colors: List[Tuple[int, int, int]], t: float) -> Tuple[int, int, int, int]:
        n = len(colors)
        if n == 0:
            return (128, 128, 128, 255)
        if n == 1:
            return colors[0] + (255,)

        t = max(0, min(1, t))
        segment_size = 1.0 / (n - 1)
        segment = min(int(t / segment_size), n - 2)
        local_t = (t - segment * segment_size) / segment_size

        c1 = colors[segment]
        c2 = colors[segment + 1]

        r = int(c1[0] + (c2[0] - c1[0]) * local_t)
        g = int(c1[1] + (c2[1] - c1[1]) * local_t)
        b = int(c1[2] + (c2[2] - c1[2]) * local_t)

        return (r, g, b, 255)

    def _add_inner_effects(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                           radius: int, stage: int) -> Image.Image:
        effects = []
        if random.random() > 0.5:
            effects.append('highlight')
        if random.random() > 0.6:
            effects.append('inner_shadow')
        if random.random() > 0.7:
            effects.append('inner_glow')

        for effect in effects:
            if effect == 'highlight':
                img = self._add_top_highlight(img, bbox, radius)
            elif effect == 'inner_shadow':
                img = self._add_inner_shadow(img, bbox, radius)
            elif effect == 'inner_glow':
                img = self._add_inner_glow(img, bbox, radius)

        return img

    def _add_top_highlight(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                           radius: int) -> Image.Image:
        x0, y0, x1, y1 = bbox
        highlight_height = min(20, (y1 - y0) // 3)

        highlight = Image.new('RGBA', img.size, (0, 0, 0, 0))

        for y in range(highlight_height):
            alpha = int(40 * (1 - y / highlight_height))
            for x in range(x0, x1):
                if self._point_in_rounded_rect(x, y + y0, bbox, radius):
                    highlight.putpixel((x, y + y0), (255, 255, 255, alpha))

        return Image.alpha_composite(img, highlight)

    def _add_inner_shadow(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                          radius: int) -> Image.Image:
        x0, y0, x1, y1 = bbox
        shadow_size = random.randint(3, 8)

        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))

        for y in range(y0, y1):
            for x in range(x0, x1):
                if not self._point_in_rounded_rect(x, y, bbox, radius):
                    continue

                dist_left = x - x0
                dist_right = x1 - x
                dist_top = y - y0
                dist_bottom = y1 - y
                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_dist < shadow_size:
                    alpha = int(30 * (1 - min_dist / shadow_size))
                    shadow.putpixel((x, y), (0, 0, 0, alpha))

        return Image.alpha_composite(img, shadow)

    def _add_inner_glow(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                        radius: int) -> Image.Image:
        x0, y0, x1, y1 = bbox
        glow_size = random.randint(5, 12)
        glow_color = self._random_vibrant_color()

        glow = Image.new('RGBA', img.size, (0, 0, 0, 0))

        for y in range(y0, y1):
            for x in range(x0, x1):
                if not self._point_in_rounded_rect(x, y, bbox, radius):
                    continue

                dist_left = x - x0
                dist_right = x1 - x
                dist_top = y - y0
                dist_bottom = y1 - y
                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_dist < glow_size:
                    alpha = int(25 * (1 - min_dist / glow_size))
                    glow.putpixel((x, y), glow_color + (alpha,))

        return Image.alpha_composite(img, glow)

    def _point_in_rounded_rect(self, x: int, y: int, bbox: Tuple[int, int, int, int],
                                radius: int) -> bool:
        x0, y0, x1, y1 = bbox

        if not (x0 <= x < x1 and y0 <= y < y1):
            return False

        if radius <= 0:
            return True

        corners = [
            (x0 + radius, y0 + radius),
            (x1 - radius, y0 + radius),
            (x0 + radius, y1 - radius),
            (x1 - radius, y1 - radius),
        ]

        if x < x0 + radius and y < y0 + radius:
            cx, cy = corners[0]
            if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
                return False

        if x >= x1 - radius and y < y0 + radius:
            cx, cy = corners[1]
            if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
                return False

        if x < x0 + radius and y >= y1 - radius:
            cx, cy = corners[2]
            if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
                return False

        if x >= x1 - radius and y >= y1 - radius:
            cx, cy = corners[3]
            if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
                return False

        return True

    def _create_background(self, width: int, height: int, stage: int) -> Image.Image:
        if stage <= 2:
            v = random.randint(240, 255)
            return Image.new('RGBA', (width, height), (v, v, v, 255))

        bg_type = random.choice(['solid', 'solid', 'gradient', 'textured'])

        if bg_type == 'solid':
            color = self._random_color(alpha=255)
            return Image.new('RGBA', (width, height), color)

        elif bg_type == 'gradient':
            colors = [self._random_color()[:3] for _ in range(2)]
            direction = random.choice(['horizontal', 'vertical'])
            img = self._create_gradient((width, height), colors, direction)
            return img.convert('RGBA')

        else:
            base_color = self._random_color(alpha=255)
            img = Image.new('RGBA', (width, height), base_color)
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode='RGBA')

    def _compute_coords(self, bbox: Tuple[int, int, int, int], radius: int,
                        img_w: int, img_h: int) -> NinePatchCoords:
        x0, y0, x1, y1 = bbox
        elem_w = x1 - x0
        elem_h = y1 - y0

        min_stretch = 4
        max_margin_x = max(1, (elem_w - min_stretch) // 2)
        max_margin_y = max(1, (elem_h - min_stretch) // 2)

        margin_x = min(max(1, radius), max_margin_x)
        margin_y = min(max(1, radius), max_margin_y)

        return NinePatchCoords(
            stretch_left=(x0 + margin_x) / img_w,
            stretch_right=(x1 - margin_x) / img_w,
            stretch_top=(y0 + margin_y) / img_h,
            stretch_bottom=(y1 - margin_y) / img_h,
        )

    def _add_shadow(self, img: Image.Image, bbox: Tuple[int, int, int, int],
                    radius: int, stage: int, alpha_mult: float = 1.0) -> Image.Image:
        offset_x = random.randint(2, 6)
        offset_y = random.randint(2, 8)
        blur_radius = random.randint(6, 12) if stage >= 3 else 8
        shadow_alpha = int(random.randint(40, 80) * alpha_mult)

        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)

        shadow_bbox = (bbox[0] + offset_x, bbox[1] + offset_y,
                       bbox[2] + offset_x, bbox[3] + offset_y)
        shadow_draw.rounded_rectangle(shadow_bbox, radius=radius,
                                      fill=(0, 0, 0, shadow_alpha))

        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        shadow.paste(img, (0, 0), img)
        return shadow

    def _add_noise(self, img: Image.Image, stage: int) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise_std = min(8, (stage - 1) * 3)
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='RGBA')

    def _get_corner_radius(self, stage: int) -> int:
        if stage == 1:
            return random.choice([0, 4, 8, 12])
        elif stage == 2:
            return random.randint(0, 20)
        else:
            return random.randint(0, 35)

    def _random_color(self, alpha: int = 255) -> Tuple[int, int, int, int]:
        return (random.randint(40, 240), random.randint(40, 240),
                random.randint(40, 240), alpha)

    def _random_vibrant_color(self) -> Tuple[int, int, int]:
        hue = random.random()

        if hue < 1/6:
            r, g, b = 255, int(255 * hue * 6), 0
        elif hue < 2/6:
            r, g, b = int(255 * (2 - hue * 6)), 255, 0
        elif hue < 3/6:
            r, g, b = 0, 255, int(255 * (hue * 6 - 2))
        elif hue < 4/6:
            r, g, b = 0, int(255 * (4 - hue * 6)), 255
        elif hue < 5/6:
            r, g, b = int(255 * (hue * 6 - 4)), 0, 255
        else:
            r, g, b = 255, 0, int(255 * (6 - hue * 6))

        r = max(100, min(255, r + random.randint(-30, 30)))
        g = max(100, min(255, g + random.randint(-30, 30)))
        b = max(100, min(255, b + random.randint(-30, 30)))

        return (r, g, b)


def generate_and_cache_dataset(generator: NinePatchGeneratorV2,
                                n_samples: int,
                                stage: int,
                                cache_dir: Path,
                                split: str = 'train') -> Path:
    """Generate samples and cache to disk."""
    split_dir = cache_dir / f'stage{stage}' / split
    split_dir.mkdir(parents=True, exist_ok=True)

    meta_file = split_dir / 'metadata.json'
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        if meta.get('n_samples') == n_samples:
            print(f"Using cached {split} data for stage {stage} ({n_samples} samples)")
            return split_dir

    print(f"Generating {n_samples} {split} samples for stage {stage}...")

    coords_list = []
    for i in tqdm(range(n_samples)):
        img_arr, coords = generator.generate_sample(curriculum_stage=stage)
        img = Image.fromarray(img_arr)
        img.save(split_dir / f'{i:05d}.png')
        coords_list.append(coords.to_array().tolist())

    with open(meta_file, 'w') as f:
        json.dump({
            'n_samples': n_samples,
            'stage': stage,
            'coords': coords_list
        }, f)

    print(f"Cached to {split_dir}")
    return split_dir
