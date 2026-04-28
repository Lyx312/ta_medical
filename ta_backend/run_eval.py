import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
REQUIREMENTS_MARKER = VENV_DIR / ".requirements.sha256"
REENTRY_FLAG = "--in-project-venv"

load_dotenv(ROOT_DIR / ".env")


def requirements_hash() -> str:
    content = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(content).hexdigest()


def ensure_venv_exists() -> None:
    if VENV_PYTHON.exists():
        return
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=str(ROOT_DIR))


def ensure_running_in_project_venv() -> None:
    ensure_venv_exists()

    current_python = Path(sys.executable).resolve()
    project_python = VENV_PYTHON.resolve()
    if current_python == project_python:
        return

    args = [str(project_python), str(ROOT_DIR / "run_eval.py"), REENTRY_FLAG]
    args.extend(arg for arg in sys.argv[1:] if arg != REENTRY_FLAG)
    try:
        subprocess.check_call(args, cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        sys.exit(0)
    sys.exit(0)


def ensure_requirements_installed() -> None:
    if not REQUIREMENTS_FILE.exists():
        return

    target_hash = requirements_hash()
    current_hash = ""
    if REQUIREMENTS_MARKER.exists():
        current_hash = REQUIREMENTS_MARKER.read_text(encoding="utf-8").strip()

    if current_hash == target_hash:
        return

    subprocess.check_call(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=str(ROOT_DIR),
    )
    REQUIREMENTS_MARKER.write_text(target_hash, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run backend evaluation utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run extraction evaluation over a dataset folder.",
    )
    evaluate_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to `python -m eval.run_evaluation`.",
    )

    template_parser = subparsers.add_parser(
        "template",
        help="Generate a ground-truth template for one claim folder.",
    )
    template_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to `python -m eval.generate_template`.",
    )

    clear_parser = subparsers.add_parser(
        "clear-cache",
        help="Delete cached evaluation scan results.",
    )
    clear_parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Dataset folder used by eval.run_evaluation (contains claim folders).",
    )
    clear_parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Cache directory path (default: <folder>/results).",
    )
    clear_parser.add_argument(
        "--claim",
        type=str,
        default=None,
        help="Optional single claim id to clear only one claim cache.",
    )
    clear_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting files.",
    )
    return parser


def clear_cache(
    *,
    folder: str,
    results_dir: str | None,
    claim: str | None,
    dry_run: bool,
) -> int:
    dataset_folder = Path(folder)
    if not dataset_folder.exists():
        print(f"Error: folder not found: {dataset_folder}")
        return 1

    cache_root = Path(results_dir) if results_dir else dataset_folder / "results"
    if claim:
        target = cache_root / claim
        if not target.exists():
            print(f"No cache found for claim '{claim}' at {target}")
            return 0
        if dry_run:
            print(f"[dry-run] Would remove: {target}")
            return 0
        shutil.rmtree(target)
        print(f"Removed claim cache: {target}")
        return 0

    if not cache_root.exists():
        print(f"No cache folder found at {cache_root}")
        return 0
    if dry_run:
        print(f"[dry-run] Would remove: {cache_root}")
        return 0

    shutil.rmtree(cache_root)
    print(f"Removed cache folder: {cache_root}")
    return 0


def main() -> None:
    parser = build_parser()
    raw_args = [arg for arg in sys.argv[1:] if arg != REENTRY_FLAG]
    parsed = parser.parse_args(raw_args)
    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    if parsed.command == "clear-cache":
        raise SystemExit(
            clear_cache(
                folder=parsed.folder,
                results_dir=parsed.results_dir,
                claim=parsed.claim,
                dry_run=parsed.dry_run,
            )
        )

    ensure_running_in_project_venv()
    ensure_requirements_installed()

    module = "eval.run_evaluation" if parsed.command == "evaluate" else "eval.generate_template"
    passthrough_args = list(parsed.args)
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    command = [sys.executable, "-m", module, *passthrough_args]
    raise SystemExit(subprocess.call(command, cwd=str(ROOT_DIR)))


if __name__ == "__main__":
    main()
