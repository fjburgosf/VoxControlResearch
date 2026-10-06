"""Build the Windows application folder and its zip.

    .venv\Scripts\python.exe tools\build_exe.py

Result: dist/VoxControlResearch/ (VoxControlResearch.exe, configs/, _internal/) and
entregables/VoxControlResearch_<version>_Windows_x64.zip
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voxcontrol import SOFTWARE_NAME, __version__  # noqa: E402

DIST = ROOT / "dist"
APP = DIST / SOFTWARE_NAME
MAX_PATH = 120


def main() -> int:
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST),
                    "--workpath", str(ROOT / "build"), str(ROOT / "packaging" / f"{SOFTWARE_NAME}.spec")],
                   cwd=ROOT / "packaging", check=True)
    shutil.copytree(ROOT / "configs", APP / "configs", dirs_exist_ok=True)
    shutil.copy2(ROOT / "README.md", APP / "README.md")
    longest = max((len(str(p.relative_to(DIST))) for p in APP.rglob("*")), default=0)
    if longest > MAX_PATH:
        raise SystemExit(f"an internal path has {longest} characters (limit {MAX_PATH})")
    out = ROOT / "entregables"
    out.mkdir(exist_ok=True)
    zip_base = out / f"{SOFTWARE_NAME}_{__version__}_Windows_x64"
    shutil.make_archive(str(zip_base), "zip", DIST, SOFTWARE_NAME)
    print(f"built {APP} (longest internal path {longest} characters) and {zip_base}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
