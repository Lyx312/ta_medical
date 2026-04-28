import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init import initialize_database
from app.main import create_app
from eval.cli_utils import ensure_family_candidates, guess_mime_type, register_test_user
from fastapi.testclient import TestClient


def _discover_image_files(folder: Path) -> list[Path]:
    images_dir = folder / "images"
    base_dir = images_dir if images_dir.is_dir() else folder
    return [
        path
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]


def _blank_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _blank_like(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_blank_like(item) for item in value]
    return None


async def generate_template_for_folder(
    folder: Path,
    client: TestClient,
    token: str,
    patient_name_candidates: list[str],
) -> dict[str, Any]:
    image_files = _discover_image_files(folder)
    if not image_files:
        raise ValueError(f"No image files found in {folder}")

    files_for_request = []
    for path in image_files:
        with open(path, "rb") as file_handle:
            relative_name = str(path.relative_to(folder)).replace("\\", "/")
            files_for_request.append((relative_name, file_handle.read(), guess_mime_type(path)))

    files = [("files", (name, content, mime_type)) for name, content, mime_type in files_for_request]
    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Scan failed: HTTP {response.status_code}: {response.text}")

    predicted = response.json()
    images_rel = [str(path.relative_to(folder)).replace("\\", "/") for path in image_files]
    return {
        "images": images_rel,
        "patient_name_candidates": patient_name_candidates,
        "expected": _blank_like(
            {
                "summary": predicted.get("summary", {}),
                "patient_info": predicted.get("patient_info", {}),
                "medical_context": predicted.get("medical_context", {}),
                "medicines": predicted.get("medicines", []),
                "other_charges": predicted.get("other_charges", []),
                "financial_recap": predicted.get("financial_recap", {}),
            }
        ),
        "_predicted": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ground-truth template from a real scan result")
    parser.add_argument("--folder", type=str, required=True, help="Path to claim folder containing images")
    parser.add_argument(
        "--candidates",
        type=str,
        nargs="*",
        default=[],
        help="Patient name candidates (space-separated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ground_truth.json (default: <folder>/ground_truth.json)",
    )
    parser.add_argument("--db-url", type=str, default=None, help="Database URL (default: use app config)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    os.environ["APP_ENV"] = "test"
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    try:
        from app.core.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass

    initialize_database()
    app = create_app()
    with TestClient(app) as client:
        primary_name = args.candidates[0] if args.candidates else "Eval User"
        token = register_test_user(client, "eval-template@example.com", primary_name)
        ensure_family_candidates(client, token, args.candidates, primary_name)
        print(f"Scanning images in {folder}... ", end="", flush=True)
        template = asyncio.run(generate_template_for_folder(folder, client, token, args.candidates))
        print("done")

    output_path = Path(args.output) if args.output else folder / f"{folder.name}_ground_truth.json"
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(template, file_handle, indent=2, ensure_ascii=False)

    print(f"\nTemplate written to {output_path}")
    print("Fill in the 'expected' values in the ground_truth.json to create a proper test case.")
    print("The '_predicted' field contains the raw model output for reference only.")
    print("Remove the '_predicted' field before using the file for evaluation.")


if __name__ == "__main__":
    main()
