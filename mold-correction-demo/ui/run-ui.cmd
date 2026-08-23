@echo off
setlocal
pushd "%~dp0"

set "AJIN_NODE_DIR=C:\Users\KDT013\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
set "AJIN_PNPM=C:\Users\KDT013\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
set "AJIN_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
set "PATH=%AJIN_NODE_DIR%;%PATH%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "TOKENIZERS_PARALLELISM=false"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js 22 or later is required.
  echo Install Node.js and run this file again.
  pause
  exit /b 1
)

if not exist "%AJIN_PNPM%" (
  where pnpm >nul 2>nul
  if not errorlevel 1 set "AJIN_PNPM=pnpm"
)

if not exist "%AJIN_PNPM%" if not "%AJIN_PNPM%"=="pnpm" (
  echo [ERROR] pnpm was not found.
  echo Run corepack enable in CMD and try again.
  pause
  exit /b 1
)

if not exist "node_modules\.bin\vinext.cmd" (
  echo Installing UI packages...
  call "%AJIN_PNPM%" install
  if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
  )
)

if not exist "%AJIN_PYTHON%" (
  echo [ERROR] Python engine environment was not found.
  echo Expected: %AJIN_PYTHON%
  pause
  exit /b 1
)

powershell -NoProfile -Command "$client = New-Object Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8000); exit 0 } catch { exit 1 } finally { $client.Dispose() }"
if not errorlevel 1 (
  echo Local vision engines are already running.
) else (
  echo Starting local vision engines...
  start "AJIN Vision Engines" /B "%AJIN_PYTHON%" "%~dp0backend\server.py"
  powershell -NoProfile -Command "$deadline = (Get-Date).AddSeconds(20); do { $client = New-Object Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8000); $client.Dispose(); exit 0 } catch { $client.Dispose(); Start-Sleep -Milliseconds 500 } } while ((Get-Date) -lt $deadline); exit 1"
  if errorlevel 1 (
    echo [ERROR] Local vision engines failed to start.
    pause
    exit /b 1
  )
)

powershell -NoProfile -Command "$client = New-Object Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 3000); exit 0 } catch { exit 1 } finally { $client.Dispose() }"
if not errorlevel 1 (
  echo.
  echo AJIN Die Insight is already running.
  echo Open http://127.0.0.1:3000 in your browser.
  popd
  exit /b 0
)

echo.
echo Starting AJIN Die Insight...
echo Open http://127.0.0.1:3000 in your browser.
echo Press Ctrl+C to stop the server.
echo.

call "node_modules\.bin\vinext.cmd" dev
set "AJIN_EXIT=%ERRORLEVEL%"
popd
exit /b %AJIN_EXIT%
