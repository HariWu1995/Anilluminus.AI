@echo off

set GIT=
set VENV_DIR=C:\Users\Mr. RIAH\Documents\GenAI\sd_env

set PYTHON="C:\Program Files\Python310\python.exe"
:set PYTHON="%VENV_DIR%\Scripts\Python.exe"

::TIMEOUT /T 1

echo %VENV_DIR%\Scripts\activate

::call %PYTHON% -m pip install -r requirements.txt
call %PYTHON% -m pip install gradio==4.25.0
call %PYTHON% -m pip install diffusers==0.24.0

call %PYTHON% app.py

echo.
echo Launch unsuccessful. Exiting.
pause
