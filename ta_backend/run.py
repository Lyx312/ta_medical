import hashlib
import os
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

    args = [str(project_python), str(ROOT_DIR / "run.py"), REENTRY_FLAG]
    args.extend(arg for arg in sys.argv[1:] if arg != REENTRY_FLAG)
    try:
        subprocess.check_call(args, cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        # Ctrl+C from the parent bootstrap process should exit quietly.
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


def main() -> None:
    ensure_running_in_project_venv()
    ensure_requirements_installed()

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "1") == "1"

    import uvicorn

    uvicorn.run("main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
