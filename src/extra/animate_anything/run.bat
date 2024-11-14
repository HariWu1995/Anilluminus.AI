@echo off

set GIT=
set VENV_DIR=C:\Users\Mr. RIAH\Documents\GenAI\sd_env
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
::set PYTHON="C:\Program Files\Python310\python.exe"

::TIMEOUT /T 1

::call "C:\Users\Mr. RIAH\Documents\sd_env\Scripts\activate.bat"

::call %PYTHON% -m pip install -r requirements.txt
call %PYTHON% -m pip install gradio==4.25.0
call %PYTHON% app.py

echo.
echo Launch unsuccessful. Exiting.
pause
