from pathlib import Path

import pytest
from PIL import Image

from tools.preprocessing_lab_core import (
    StageVariantSpec,
    build_stage_parameter_grid,
    create_contact_sheet,
    parse_tile_grid_size,
    process_stage_variants,
    save_lab_outputs,
)


def test_parse_tile_grid_size_accepts_width_x_height() -> None:
    assert parse_tile_grid_size("8x8") == (8, 8)
    assert parse_tile_grid_size("16 x 12") == (16, 12)


def test_parse_tile_grid_size_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        parse_tile_grid_size("8,8")


def test_build_stage_parameter_grid_generates_resize_variants() -> None:
    grid = build_stage_parameter_grid("resize", max_long_sides=[512, 1024])

    assert grid == [
        StageVariantSpec(stage="resize", params={"max_long_side": 512}),
        StageVariantSpec(stage="resize", params={"max_long_side": 1024}),
    ]


def test_build_stage_parameter_grid_generates_only_selected_stage_cross_product() -> None:
    grid = build_stage_parameter_grid(
        "clahe",
        max_long_sides=[512, 1024],
        clahe_clip_limits=[1.0, 2.0],
        clahe_tile_grid_sizes=[(8, 8), (16, 16)],
        unsharp_sigmas=[1.0, 1.5],
        unsharp_amounts=[0.0, 0.15],
        jpeg_qualities=[85, 95],
    )

    assert len(grid) == 4
    assert {spec.stage for spec in grid} == {"clahe"}
    assert all("max_long_side" not in spec.params for spec in grid)
    assert all("unsharp_sigma" not in spec.params for spec in grid)
    assert all("jpeg_quality" not in spec.params for spec in grid)


def test_build_stage_parameter_grid_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError, match="Unsupported stage"):
        build_stage_parameter_grid("deskew")


@pytest.mark.parametrize(
    ("stage", "kwargs"),
    [
        ("resize", {"max_long_sides": [256, 512, 1024]}),
        ("clahe", {"clahe_clip_limits": [1.0, 2.0], "clahe_tile_grid_sizes": [(8, 8), (16, 16)]}),
        ("unsharp", {"unsharp_sigmas": [0.75, 1.0], "unsharp_amounts": [0.0, 0.15]}),
        ("encode", {"jpeg_qualities": [70, 85, 95]}),
    ],
)
def test_build_stage_parameter_grid_safety_cap_applies_per_stage(stage: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="above max_variants=2"):
        build_stage_parameter_grid(stage, max_variants=2, **kwargs)


def test_resize_stage_only_changes_dimensions() -> None:
    source_image = Image.new("RGB", (120, 80), color=(240, 240, 240))
    grid = build_stage_parameter_grid("resize", max_long_sides=[60])

    variants = process_stage_variants(source_image, grid)

    assert len(variants) == 1
    assert variants[0].stage == "resize"
    assert variants[0].image.size == (60, 40)
    assert variants[0].params == {"max_long_side": 60}
    assert variants[0].mime_type == "image/jpeg"
    assert variants[0].content


def test_clahe_stage_keeps_original_dimensions() -> None:
    source_image = Image.new("RGB", (120, 80), color=(180, 180, 180))
    grid = build_stage_parameter_grid("clahe", clahe_clip_limits=[1.0], clahe_tile_grid_sizes=[(8, 8)])

    variants = process_stage_variants(source_image, grid)

    assert variants[0].stage == "clahe"
    assert variants[0].image.size == source_image.size
    assert variants[0].params == {"clahe_clip_limit": 1.0, "clahe_tile_grid_size": (8, 8)}


def test_unsharp_stage_keeps_original_dimensions() -> None:
    source_image = Image.new("RGB", (120, 80), color=(180, 180, 180))
    grid = build_stage_parameter_grid("unsharp", unsharp_sigmas=[1.0], unsharp_amounts=[0.15])

    variants = process_stage_variants(source_image, grid)

    assert variants[0].stage == "unsharp"
    assert variants[0].image.size == source_image.size
    assert variants[0].params == {"unsharp_sigma": 1.0, "unsharp_amount": 0.15}


def test_encode_stage_reports_quality_and_file_size_difference() -> None:
    source_image = Image.new("RGB", (96, 64))
    pixels = source_image.load()
    for y in range(source_image.height):
        for x in range(source_image.width):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)

    grid = build_stage_parameter_grid("encode", jpeg_qualities=[30, 95])
    variants = process_stage_variants(source_image, grid)

    assert [variant.stage for variant in variants] == ["encode", "encode"]
    assert [variant.params["jpeg_quality"] for variant in variants] == [30, 95]
    assert variants[0].image.size == source_image.size
    assert variants[1].image.size == source_image.size
    assert variants[0].metadata()["byte_size"] != variants[1].metadata()["byte_size"]


def test_contact_sheet_smoke() -> None:
    source_image = Image.new("RGB", (120, 80), color=(240, 240, 240))
    variants = process_stage_variants(source_image, build_stage_parameter_grid("resize", max_long_sides=[60]))

    contact_sheet = create_contact_sheet(original_image=source_image, variants=variants, columns=2)

    assert contact_sheet.width > source_image.width
    assert contact_sheet.height > source_image.height


def test_save_lab_outputs_writes_images_and_stage_metadata(tmp_path: Path) -> None:
    source_image = Image.new("RGB", (90, 60), color=(230, 230, 230))
    variants = process_stage_variants(
        source_image,
        build_stage_parameter_grid("clahe", clahe_clip_limits=[1.0], clahe_tile_grid_sizes=[(8, 8)]),
    )
    contact_sheet = create_contact_sheet(original_image=source_image, variants=variants)

    metadata = save_lab_outputs(
        output_dir=tmp_path,
        input_path="sample.jpg",
        original_image=source_image,
        variants=variants,
        contact_sheet=contact_sheet,
    )

    assert (tmp_path / "original.png").exists()
    assert (tmp_path / "variant_001.jpg").exists()
    assert (tmp_path / "contact_sheet.jpg").exists()
    assert (tmp_path / "results.json").exists()
    assert metadata["stage"] == "clahe"
    assert metadata["original_width"] == 90
    assert metadata["original_height"] == 60
    variant_metadata = metadata["variants"][0]  # type: ignore[index]
    assert variant_metadata["stage"] == "clahe"
    assert variant_metadata["byte_size"] > 0
    assert variant_metadata["params"]["clahe_tile_grid_size"] == [8, 8]
