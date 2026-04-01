from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image

from app.services.image_preprocessing import MAX_LONG_SIDE, preprocess_document_image


def build_image_bytes(
    *,
    size: tuple[int, int],
    image_mode: str = "RGB",
    image_format: str = "JPEG",
    color: tuple[int, ...] = (240, 240, 240),
    exif_orientation: int | None = None,
) -> bytes:
    image = Image.new(image_mode, size, color=color)
    output = BytesIO()

    save_kwargs: dict[str, str | bytes] = {"format": image_format}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        save_kwargs["exif"] = exif.tobytes()

    image.save(output, **save_kwargs)
    return output.getvalue()


def test_preprocess_document_image_downsizes_long_side_to_1024() -> None:
    processed_image = preprocess_document_image(
        filename="large.jpg",
        content=build_image_bytes(size=(2400, 1200)),
    )

    with Image.open(BytesIO(processed_image.content)) as image:
        assert max(image.size) == MAX_LONG_SIDE
        assert image.size == (1024, 512)
    assert processed_image.mime_type == "image/jpeg"
    assert processed_image.filename.endswith(".jpg")


def test_preprocess_document_image_does_not_upscale_smaller_images() -> None:
    processed_image = preprocess_document_image(
        filename="small.jpg",
        content=build_image_bytes(size=(400, 200)),
    )

    with Image.open(BytesIO(processed_image.content)) as image:
        assert image.size == (400, 200)


def test_preprocess_document_image_applies_exif_autorotate() -> None:
    processed_image = preprocess_document_image(
        filename="rotated.jpg",
        content=build_image_bytes(size=(40, 80), exif_orientation=6),
    )

    with Image.open(BytesIO(processed_image.content)) as image:
        assert image.size == (80, 40)


def test_preprocess_document_image_preserves_alpha_as_png() -> None:
    processed_image = preprocess_document_image(
        filename="transparent.png",
        content=build_image_bytes(
            size=(600, 600),
            image_mode="RGBA",
            image_format="PNG",
            color=(255, 255, 255, 128),
        ),
    )

    with Image.open(BytesIO(processed_image.content)) as image:
        assert image.mode == "RGBA"
    assert processed_image.mime_type == "image/png"
    assert processed_image.filename.endswith(".png")


def test_preprocess_document_image_rejects_invalid_bytes() -> None:
    with pytest.raises(HTTPException) as exc_info:
        preprocess_document_image(
            filename="broken.png",
            content=b"not-an-image",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Uploaded file is not a readable image."
