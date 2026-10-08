r"""Build the Windows application folder and its zip.

    .venv\Scripts\python.exe tools\build_exe.py

Result: dist/VoxControlResearch/ (VoxControlResearch.exe, README.md, configs/ and lib/, the folder with the
Python runtime and libraries) and entregables/VoxControlResearch_<version>_Windows_x64.zip
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


def _bundled_distributions() -> list:
    """Third-party distributions whose modules ended up in the application."""
    import importlib.metadata as md
    import re
    toc = (ROOT / "build" / SOFTWARE_NAME / "PYZ-00.toc").read_text(encoding="utf-8", errors="ignore")
    names = {m.group(1).split(".")[0] for m in re.finditer(r"\('([A-Za-z0-9_.]+)',", toc)}
    names |= {p.name.split(".")[0] for p in (APP / "lib").iterdir() if p.is_dir()}
    owners = md.packages_distributions()
    found = {d for n in names for d in owners.get(n, []) if d.lower() not in ("voxcontrol", "voxcontrolresearch")}
    return [md.distribution(d) for d in sorted(found, key=str.lower)]


def _third_party_notices() -> int:
    """Copy the notice files that each bundled third-party library ships (they must travel with it)."""
    target = APP / "lib" / "third_party_notices"
    copied = 0
    for dist in _bundled_distributions():
        notices = [f for f in dist.files or [] if any(k in f.name.upper() for k in ("LICEN", "COPYING", "NOTICE"))]
        for i, f in enumerate(notices, 1):
            src = Path(dist.locate_file(f))
            if src.is_file():          # numbered: one library can ship several files with the same name
                dst = target / dist.metadata["Name"] / f"{i:02d}_{f.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    return copied


def main() -> int:
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST),
                    "--workpath", str(ROOT / "build"), str(ROOT / "packaging" / f"{SOFTWARE_NAME}.spec")],
                   cwd=ROOT / "packaging", check=True)
    _third_party_notices()
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
