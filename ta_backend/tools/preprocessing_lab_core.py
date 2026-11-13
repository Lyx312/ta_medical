from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

DEFAULT_MAX_LONG_SIDE = 1024
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)
DEFAULT_UNSHARP_SIGMA = 1.0
DEFAULT_UNSHARP_AMOUNT = 0.15
DEFAULT_JPEG_QUALITY = 85
SUPPORTED_STAGES = ("resize", "clahe", "unsharp", "encode")
StageParamValue = int | float | tuple[int, int]


@dataclass(frozen=True)
class StageVariantSpec:
    stage: str
    params: dict[str, StageParamValue]


@dataclass(frozen=True)
class PreprocessingVariant:
    index: int
    stage: str
    params: dict[str, StageParamValue]
    image: Image.Image
    content: bytes
    extension: str
    mime_type: str
    filename: str

    @property
    def label(self) -> str:
        parts = [f"#{self.index}", f"stage={self.stage}"]
        parts.extend(f"{key}={_format_label_value(value)}" for key, value in self.params.items())
        return " | ".join(parts)

    def metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "stage": self.stage,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "byte_size": len(self.content),
            "width": self.image.width,
            "height": self.image.height,
            "params": _json_safe_params(self.params),
        }


def load_source_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as source_image:
        return ImageOps.exif_transpose(source_image).copy()


def build_stage_parameter_grid(
    stage: str,
    *,
    max_long_sides: Iterable[int] = (DEFAULT_MAX_LONG_SIDE,),
    clahe_clip_limits: Iterable[float] = (DEFAULT_CLAHE_CLIP_LIMIT,),
    clahe_tile_grid_sizes: Iterable[tuple[int, int]] = (DEFAULT_CLAHE_TILE_GRID_SIZE,),
    unsharp_sigmas: Iterable[float] = (DEFAULT_UNSHARP_SIGMA,),
    unsharp_amounts: Iterable[float] = (DEFAULT_UNSHARP_AMOUNT,),
    jpeg_qualities: Iterable[int] = (DEFAULT_JPEG_QUALITY,),
    max_variants: int = 100,
) -> list[StageVariantSpec]:
    normalized_stage = normalize_stage(stage)

    if normalized_stage == "resize":
        specs = [
            StageVariantSpec(stage=normalized_stage, params={"max_long_side": int(max_long_side)})
            for max_long_side in max_long_sides
        ]
    elif normalized_stage == "clahe":
        specs = [
            StageVariantSpec(
                stage=normalized_stage,
                params={
                    "clahe_clip_limit": float(clip_limit),
                    "clahe_tile_grid_size": _coerce_tile_grid_size(tile_grid_size),
                },
            )
            for clip_limit, tile_grid_size in product(clahe_clip_limits, clahe_tile_grid_sizes)
        ]
    elif normalized_stage == "unsharp":
        specs = [
            StageVariantSpec(
                stage=normalized_stage,
                params={"unsharp_sigma": float(sigma), "unsharp_amount": float(amount)},
            )
            for sigma, amount in product(unsharp_sigmas, unsharp_amounts)
        ]
    else:
        specs = [
            StageVariantSpec(stage=normalized_stage, params={"jpeg_quality": int(jpeg_quality)})
            for jpeg_quality in jpeg_qualities
        ]

    if not specs:
        raise ValueError(f"Stage {normalized_stage!r} has no parameter values to test.")
    if len(specs) > max_variants:
        raise ValueError(
            f"Parameter grid has {len(specs)} variants, above max_variants={max_variants}. "
            "Reduce the selected stage parameter list or increase the safety cap."
        )

    for spec in specs:
        _validate_stage_spec(spec)
    return specs


def process_stage_variants(source_image: Image.Image, variant_specs: Iterable[StageVariantSpec]) -> list[PreprocessingVariant]:
    variants: list[PreprocessingVariant] = []
    original = source_image.copy()

    for index, spec in enumerate(variant_specs, start=1):
        stage = normalize_stage(spec.stage)
        params = dict(spec.params)
        _validate_stage_spec(StageVariantSpec(stage=stage, params=params))

        if stage == "resize":
            processed = resize_if_needed(original, _int_param(params, "max_long_side"))
            content, extension, mime_type = encode_processed_image(processed, DEFAULT_JPEG_QUALITY)
        elif stage == "clahe":
            processed = apply_clahe_only(
                original,
                clip_limit=_float_param(params, "clahe_clip_limit"),
                tile_grid_size=_tile_grid_param(params, "clahe_tile_grid_size"),
            )
            content, extension, mime_type = encode_processed_image(processed, DEFAULT_JPEG_QUALITY)
        elif stage == "unsharp":
            processed = apply_unsharp_only(
                original,
                sigma=_float_param(params, "unsharp_sigma"),
                amount=_float_param(params, "unsharp_amount"),
            )
            content, extension, mime_type = encode_processed_image(processed, DEFAULT_JPEG_QUALITY)
        else:
            content, extension, mime_type = encode_jpeg_preview(original, _int_param(params, "jpeg_quality"))
            with Image.open(BytesIO(content)) as preview:
                processed = preview.copy()

        variants.append(
            PreprocessingVariant(
                index=index,
                stage=stage,
                params=params,
                image=processed,
                content=content,
                extension=extension,
                mime_type=mime_type,
                filename=f"variant_{index:03d}.{extension}",
            )
        )

    return variants


def normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized not in SUPPORTED_STAGES:
        stages = ", ".join(SUPPORTED_STAGES)
        raise ValueError(f"Unsupported stage {stage!r}. Supported stages: {stages}.")
    return normalized


def resize_if_needed(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side <= 0:
        raise ValueError("max_long_side must be greater than 0.")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_long_side:
        return image.copy()

    scale = max_long_side / longest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def apply_clahe_only(
    image: Image.Image,
    *,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID_SIZE,
) -> Image.Image:
    if clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be greater than 0.")
    if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError("clahe_tile_grid_size values must be greater than 0.")

    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    lightness_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    enhanced_lightness = clahe.apply(lightness_channel)
    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def apply_unsharp_only(
    image: Image.Image,
    *,
    sigma: float = DEFAULT_UNSHARP_SIGMA,
    amount: float = DEFAULT_UNSHARP_AMOUNT,
) -> Image.Image:
    if sigma <= 0:
        raise ValueError("unsharp_sigma must be greater than 0.")
    if amount < 0:
        raise ValueError("unsharp_amount must not be negative.")

    rgb_array = np.array(image.convert("RGB"))
    blurred = cv2.GaussianBlur(rgb_array, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharpened = cv2.addWeighted(rgb_array, 1.0 + float(amount), blurred, -float(amount), 0)
    return Image.fromarray(sharpened)


def encode_processed_image(image: Image.Image, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> tuple[bytes, str, str]:
    if has_alpha_channel(image):
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue(), "png", "image/png"
    return encode_jpeg_preview(image, jpeg_quality)


def encode_jpeg_preview(image: Image.Image, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> tuple[bytes, str, str]:
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")

    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return output.getvalue(), "jpg", "image/jpeg"


def create_contact_sheet(
    *,
    original_image: Image.Image,
    variants: Iterable[PreprocessingVariant],
    columns: int = 3,
    thumb_width: int = 360,
    padding: int = 18,
    label_height: int = 72,
    background: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    if columns <= 0:
        raise ValueError("columns must be greater than 0.")

    cells: list[tuple[str, Image.Image]] = [("original", original_image)]
    cells.extend((variant.label, variant.image) for variant in variants)

    thumb_heights = [_scaled_height(image, thumb_width) for _, image in cells]
    cell_width = thumb_width + padding * 2
    cell_height = max(thumb_heights, default=thumb_width) + label_height + padding * 2
    row_count = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height), background)
    draw = ImageDraw.Draw(sheet)

    for cell_index, (label, image) in enumerate(cells):
        row = cell_index // columns
        column = cell_index % columns
        x = column * cell_width + padding
        y = row * cell_height + padding
        thumbnail = _make_thumbnail(image, thumb_width)
        sheet.paste(thumbnail.convert("RGB"), (x, y))
        label_y = y + thumbnail.height + 8
        for line in _wrap_label(label, max_chars=42):
            draw.text((x, label_y), line, fill=(20, 20, 20))
            label_y += 14

    return sheet


def save_lab_outputs(
    *,
    output_dir: str | Path,
    input_path: str | Path,
    original_image: Image.Image,
    variants: Iterable[PreprocessingVariant],
    contact_sheet: Image.Image | None = None,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variant_list = list(variants)
    original_image.save(output_path / "original.png")
    for variant in variant_list:
        (output_path / variant.filename).write_bytes(variant.content)

    if contact_sheet is not None:
        contact_sheet.convert("RGB").save(output_path / "contact_sheet.jpg", format="JPEG", quality=90)

    selected_stage = variant_list[0].stage if variant_list else None
    metadata: dict[str, object] = {
        "input_path": str(input_path),
        "stage": selected_stage,
        "original_width": original_image.width,
        "original_height": original_image.height,
        "variant_count": len(variant_list),
        "variants": [variant.metadata() for variant in variant_list],
    }
    (output_path / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def has_alpha_channel(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)


def parse_tile_grid_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value)
    if not match:
        raise ValueError("Tile grid size must use WIDTHxHEIGHT format, for example 8x8.")
    return int(match.group(1)), int(match.group(2))


def _coerce_tile_grid_size(value: tuple[int, int]) -> tuple[int, int]:
    width, height = value
    return int(width), int(height)


def _validate_stage_spec(spec: StageVariantSpec) -> None:
    stage = normalize_stage(spec.stage)
    params = spec.params

    if stage == "resize":
        if _int_param(params, "max_long_side", default=0) <= 0:
            raise ValueError("max_long_side must be greater than 0.")
    elif stage == "clahe":
        tile_grid_size = _tile_grid_param(params, "clahe_tile_grid_size", default=(0, 0))
        if _float_param(params, "clahe_clip_limit", default=0) <= 0:
            raise ValueError("clahe_clip_limit must be greater than 0.")
        if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError("clahe_tile_grid_size must contain two positive integers.")
    elif stage == "unsharp":
        if _float_param(params, "unsharp_sigma", default=0) <= 0:
            raise ValueError("unsharp_sigma must be greater than 0.")
        if _float_param(params, "unsharp_amount", default=-1) < 0:
            raise ValueError("unsharp_amount must not be negative.")
    elif not 1 <= _int_param(params, "jpeg_quality", default=0) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")


def _int_param(params: Mapping[str, StageParamValue], key: str, *, default: int | None = None) -> int:
    value = params.get(key, default) if default is not None else params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number.")
    return int(value)


def _float_param(params: Mapping[str, StageParamValue], key: str, *, default: float | None = None) -> float:
    value = params.get(key, default) if default is not None else params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _tile_grid_param(
    params: Mapping[str, StageParamValue],
    key: str,
    *,
    default: tuple[int, int] | None = None,
) -> tuple[int, int]:
    value = params.get(key, default) if default is not None else params[key]
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{key} must contain two integers.")
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise ValueError(f"{key} must contain two integers.")
    return width, height


def _json_safe_params(params: dict[str, StageParamValue]) -> dict[str, object]:
    safe_params: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            safe_params[key] = list(value)
        else:
            safe_params[key] = value
    return safe_params


def _format_label_value(value: object) -> str:
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _scaled_height(image: Image.Image, target_width: int) -> int:
    width, height = image.size
    if width <= 0:
        return target_width
    return max(1, round(height * (target_width / width)))


def _make_thumbnail(image: Image.Image, target_width: int) -> Image.Image:
    thumbnail = image.copy()
    target_height = _scaled_height(thumbnail, target_width)
    thumbnail.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    return thumbnail


def _wrap_label(label: str, max_chars: int) -> list[str]:
    if len(label) <= max_chars:
        return [label]

    words = label.split(" | ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} | {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
