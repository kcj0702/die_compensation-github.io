@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-ui.ps1"
set "AJIN_EXIT_CODE=%ERRORLEVEL%"
if not "%AJIN_EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Startup failed. Review the error shown above.
  pause
)
exit /b %AJIN_EXIT_CODE%
