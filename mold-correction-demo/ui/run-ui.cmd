@echo off
setlocal
chcp 65001 >nul
REM Keep cmd parsing minimal. The PowerShell launcher safely handles Korean
REM paths, quoted arguments, dependency checks and diagnostic logs.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-ui.ps1"
exit /b %ERRORLEVEL%
