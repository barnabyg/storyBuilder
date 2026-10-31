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
rem Story Builder gets its own window: sharing a terminal with PowerShell, its prompt
rem would take the Ctrl+C. The window stays open on an error, so that the message can be read.
rem Ctrl+C gives a negative exit code, which "if errorlevel 1" does not count as an error.
start "Story Builder" cmd /c ".venv\Scripts\nde.exe gui %* & if errorlevel 1 pause"
