@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo No existe .venv. Siga la seccion de instalacion del README.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m voxcontrol gui
