@echo off
rem Starts Story Builder: double-click this file. See docs/gui.md.
cd /d "%~dp0"
if not exist ".venv\Scripts\nde.exe" (
    echo Story Builder is not set up yet, because the .venv folder is missing.
    echo Run "uv sync" in this folder first.
    pause
    exit /b 1
)
rem This batch file ends at once, so Ctrl+C never asks "Terminate batch job (Y/N)?".
rem The window stays open on an error, so that the message can be read.
start "" /b cmd /c ".venv\Scripts\nde.exe gui %* || pause"
