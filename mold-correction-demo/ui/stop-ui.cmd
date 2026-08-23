@echo off
setlocal EnableDelayedExpansion

echo Stopping AJIN Die Insight...
set "AJIN_FOUND=0"

for /f "tokens=5" %%P in ('netstat -aon ^| findstr /R /C:":3000 .*LISTENING" /C:":8000 .*LISTENING"') do (
  set "AJIN_FOUND=1"
  taskkill /PID %%P /T /F >nul 2>nul
)

if "!AJIN_FOUND!"=="0" (
  echo AJIN Die Insight is not running.
  timeout /t 2 /nobreak >nul
  exit /b 0
)

timeout /t 1 /nobreak >nul
set "AJIN_REMAINING=0"
for /f "tokens=5" %%P in ('netstat -aon ^| findstr /R /C:":3000 .*LISTENING" /C:":8000 .*LISTENING"') do set "AJIN_REMAINING=1"

if "!AJIN_REMAINING!"=="1" (
  echo [ERROR] Some processes could not be stopped.
  echo Right-click this file and choose Run as administrator.
  pause
  exit /b 1
)

echo UI and vision engines stopped successfully.
timeout /t 2 /nobreak >nul
exit /b 0
