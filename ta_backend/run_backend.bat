@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run.py
    endlocal
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python run.py
    endlocal
    exit /b %errorlevel%
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py
    endlocal
    exit /b %errorlevel%
)

echo Python interpreter was not found in PATH.
echo Install Python 3.11+ and try again.
endlocal
exit /b 1
