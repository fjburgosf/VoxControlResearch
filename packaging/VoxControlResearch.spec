# PyInstaller specification for VoxControlResearch (folder distribution).
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

hidden = collect_submodules("voxcontrol") + collect_submodules("sklearn.utils") + ["comtypes.client", "soundfile", "matplotlib.backends.backend_agg", "matplotlib.backends.backend_svg",
                                                                                   "matplotlib.backends.backend_pdf"]
datas = collect_data_files("voxcontrol") + collect_data_files("faster_whisper")
binaries = collect_dynamic_libs("ctranslate2")

a = Analysis(
    ["launcher.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["torch", "tensorflow", "pandas", "IPython", "pytest", "_pytest", "py", "nvidia", "notebook", "jupyter",
              "lxml", "docx"],
    noarchive=False,
)
# only what the program needs to run: no test suites, sample data sets or link libraries of the dependencies
_DROP = ("/tests/", "matplotlib/mpl-data/sample_data/", "sklearn/datasets/data/", "sklearn/datasets/descr/",
         "sklearn/datasets/images/", "tcltest")


def _keep(entry):
    dest = "/" + entry[0].replace("\\", "/")
    return not any(d in dest for d in _DROP) and not dest.endswith(".lib")


a.datas = [d for d in a.datas if _keep(d)]
a.binaries = [b for b in a.binaries if _keep(b)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VoxControlResearch", console=False,
          contents_directory="lib")
coll = COLLECT(exe, a.binaries, a.datas, name="VoxControlResearch")
