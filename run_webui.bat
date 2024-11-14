@echo off

set GIT=
set VENV_DIR=C:\Users\Mr. RIAH\Documents\GenAI\sd_env
set WORK_DIR=C:\Users\Mr. RIAH\Documents\GenAI\_Visual\Anilluminus.AI

set PYTHON="C:\Program Files\Python310\python.exe"
:set PYTHON="%VENV_DIR%\Scripts\Python.exe"

:activate_venv
echo %VENV_DIR%\Scripts\activate
call %PYTHON% -m pip install numpy==1.23.5 gradio==4.44.1

:launch
CD /D %WORK_DIR%
call %PYTHON% -m src.webui

echo.
echo Launch unsuccessful. Exiting.
pause
