from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, status
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_LONG_SIDE = 1024
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
UNSHARP_SIGMA = 1.0
UNSHARP_AMOUNT = 0.15
JPEG_QUALITY = 85


@dataclass(frozen=True)
class ProcessedDocumentImage:
    filename: str
    mime_type: str
    content: bytes


def preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
    try:
        with Image.open(BytesIO(content)) as source_image:
            normalized_image = ImageOps.exif_transpose(source_image)
            resized_image = _resize_if_needed(normalized_image)
            processed_image = _apply_document_enhancement(resized_image)
            processed_bytes, processed_format, mime_type = _encode_processed_image(processed_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a readable image.",
        ) from exc

    processed_filename = _build_output_filename(filename, processed_format)
    return ProcessedDocumentImage(
        filename=processed_filename,
        mime_type=mime_type,
        content=processed_bytes,
    )


def _resize_if_needed(image: Image.Image) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= MAX_LONG_SIDE:
        return image.copy()

    scale = MAX_LONG_SIDE / float(long_side)
    resized_dimensions = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized_dimensions, Image.Resampling.LANCZOS)


def _apply_document_enhancement(image: Image.Image) -> Image.Image:
    has_alpha = _has_alpha_channel(image)
    working_image = image.convert("RGBA" if has_alpha else "RGB")
    alpha_channel = working_image.getchannel("A") if has_alpha else None
    rgb_image = working_image.convert("RGB")

    rgb_array = np.array(rgb_image)
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_array)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    enhanced_l_channel = clahe.apply(l_channel)
    contrast_enhanced = cv2.cvtColor(
        cv2.merge((enhanced_l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )

    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=UNSHARP_SIGMA)
    sharpened = cv2.addWeighted(
        contrast_enhanced,
        1.0 + UNSHARP_AMOUNT,
        blurred,
        -UNSHARP_AMOUNT,
        0,
    )
    sharpened_rgb_image = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8), "RGB")

    if not has_alpha or alpha_channel is None:
        return sharpened_rgb_image

    red_channel, green_channel, blue_channel = sharpened_rgb_image.split()
    return Image.merge("RGBA", (red_channel, green_channel, blue_channel, alpha_channel))


def _encode_processed_image(image: Image.Image) -> tuple[bytes, str, str]:
    output = BytesIO()
    if _has_alpha_channel(image):
        image.save(output, format="PNG", optimize=True)
        return output.getvalue(), "png", "image/png"

    image.convert("RGB").save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return output.getvalue(), "jpg", "image/jpeg"


def _build_output_filename(filename: str, processed_format: str) -> str:
    base_name = Path(filename).stem or "uploaded_file"
    return f"{base_name}.{processed_format}"


def _has_alpha_channel(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    return image.mode == "P" and "transparency" in image.info
