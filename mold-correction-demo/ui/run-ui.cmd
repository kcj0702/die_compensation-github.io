@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
pushd "%~dp0"

REM ── Node.js 찾기 ──────────────────────────────────────────────
REM 원래는 특정 PC 경로가 박혀 있었다. PATH 를 먼저 보고,
REM 없으면 표준 설치 위치를 뒤진다.
set "AJIN_NODE="
where node >nul 2>nul && set "AJIN_NODE=node"
if not defined AJIN_NODE (
  if exist "%ProgramFiles%\nodejs\node.exe" (
    set "PATH=%ProgramFiles%\nodejs;%PATH%"
    set "AJIN_NODE=node"
  )
)
if not defined AJIN_NODE (
  if exist "%ProgramFiles(x86)%\nodejs\node.exe" (
    set "PATH=%ProgramFiles(x86)%\nodejs;%PATH%"
    set "AJIN_NODE=node"
  )
)
if not defined AJIN_NODE (
  echo [ERROR] Node.js 를 찾지 못했습니다. Node.js 22 이상을 설치하세요.
  echo         https://nodejs.org
  pause
  exit /b 1
)

REM ── pnpm 찾기 ─────────────────────────────────────────────────
set "AJIN_PNPM="
if exist "%APPDATA%\npm\pnpm.cmd" (
  set "PATH=%APPDATA%\npm;%PATH%"
  set "AJIN_PNPM=%APPDATA%\npm\pnpm.cmd"
)
if not defined AJIN_PNPM (
  where pnpm >nul 2>nul && set "AJIN_PNPM=pnpm"
)
if not defined AJIN_PNPM (
  echo pnpm 을 찾지 못해 설치합니다...
  echo   방금 Node.js/pnpm 을 설치했다면 탐색기가 새 PATH 를 아직
  echo   모르는 상태일 수 있습니다. 로그아웃 후 재로그인하거나 PC 를
  echo   재시작한 뒤 다시 실행해 보세요.
  call npm install -g pnpm
  if errorlevel 1 (
    echo [ERROR] pnpm 설치에 실패했습니다.
    pause
    exit /b 1
  )
  set "PATH=%APPDATA%\npm;%PATH%"
  set "AJIN_PNPM=%APPDATA%\npm\pnpm.cmd"
)

REM ── Python 찾기 ───────────────────────────────────────────────
REM Node 와 같은 방식이다. 특정 경로를 박아 두지 않고 아래 순서로 찾는다.
REM   첫째 AJIN_PYTHON 직접 지정, 둘째 저장소 안의 .venv, 셋째 PATH 의 python.
REM 마지막 경우는 엔진 의존성을 실제로 import 해 보고 통과한 것만 쓴다.
REM conda 처럼 이미 의존성이 갖춰진 환경이 있으면 venv 를 새로 만들 필요가 없다.
if defined AJIN_PYTHON (
  if not exist "!AJIN_PYTHON!" (
    echo [ERROR] AJIN_PYTHON 이 가리키는 파일이 없습니다: !AJIN_PYTHON!
    pause
    exit /b 1
  )
) else (
  set "AJIN_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
  if not exist "!AJIN_PYTHON!" set "AJIN_PYTHON=%~dp0..\.venv\Scripts\python.exe"
  if not exist "!AJIN_PYTHON!" (
    set "AJIN_PYTHON="
    for /f "delims=" %%P in ('where python 2^>nul') do (
      if not defined AJIN_PYTHON (
        "%%P" -c "import cv2, numpy, starlette, uvicorn" >nul 2>nul && set "AJIN_PYTHON=%%P"
      )
    )
  )
)

if not defined AJIN_PYTHON (
  echo [ERROR] 엔진을 실행할 Python 을 찾지 못했습니다.
  echo         .venv 도 없고, PATH 의 python 에도 cv2/starlette/uvicorn 이 없습니다.
  echo.
  echo   이미 의존성이 깔린 Python 이 있으면 그 경로를 지정하세요:
  echo     set AJIN_PYTHON=C:\path\to\python.exe
  echo.
  echo   또는 저장소 루트에서 가상환경을 만드세요:
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  echo     .venv\Scripts\python.exe -m pip install -r mold-correction-demo\deviation_extraction\requirements.txt
  echo     .venv\Scripts\python.exe -m pip install -r mold-correction-demo\ui\backend\requirements.txt
  pause
  exit /b 1
)
echo 엔진 Python: !AJIN_PYTHON!

REM Qwen 판독기는 transformers 4.49 이상을 요구한다. 낮으면 서버는 뜨고 정렬과
REM 라벨 검출도 되지만 편차값을 못 읽어 결과 포인트가 0개로 나온다. 화면에는
REM 원인이 잘 드러나지 않으므로 여기서 미리 알린다.
"!AJIN_PYTHON!" -c "import sys,transformers; v=transformers.__version__.split('.'); sys.exit(0 if [int(v[0]),int(v[1])] >= [4,49] else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo [주의] 이 Python 의 transformers 가 4.49 미만이거나 설치되어 있지 않습니다.
  echo        정렬과 라벨 검출은 동작하지만 Qwen 편차값 판독이 실패해
  echo        검출 포인트가 0개로 표시됩니다.
  echo        의존성이 갖춰진 다른 환경이 있으면 이렇게 지정하세요:
  echo          set AJIN_PYTHON=C:\path\to\python.exe
  echo.
)

REM 보안 방침상 외부 모델 다운로드를 막는다 (8/12 멘토링 확인 사항)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "TOKENIZERS_PARALLELISM=false"

REM ── UI 패키지 설치 ────────────────────────────────────────────
if not exist "node_modules\.bin\vinext.cmd" (
  echo UI 패키지를 설치합니다...
  call "%AJIN_PNPM%" install
  if errorlevel 1 (
    echo [ERROR] 패키지 설치에 실패했습니다.
    pause
    exit /b 1
  )
)

REM ── 엔진 서버 (8000) ──────────────────────────────────────────
powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try{$c.Connect('127.0.0.1',8000); exit 0}catch{exit 1}finally{$c.Dispose()}"
if not errorlevel 1 (
  echo 엔진 서버가 이미 실행 중입니다.
) else (
  echo 엔진 서버를 시작합니다...
  start "AJIN Vision Engines" /B "%AJIN_PYTHON%" "%~dp0backend\server.py"
  powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(30); do{$c=New-Object Net.Sockets.TcpClient; try{$c.Connect('127.0.0.1',8000); $c.Dispose(); exit 0}catch{$c.Dispose(); Start-Sleep -Milliseconds 500}}while((Get-Date) -lt $d); exit 1"
  if errorlevel 1 (
    echo [ERROR] 엔진 서버가 시작되지 않았습니다.
    echo         아래를 직접 실행해 오류를 확인하세요:
    echo         "%AJIN_PYTHON%" "%~dp0backend\server.py"
    pause
    exit /b 1
  )
)

REM ── 화면 (3000) ───────────────────────────────────────────────
powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try{$c.Connect('127.0.0.1',3000); exit 0}catch{exit 1}finally{$c.Dispose()}"
if not errorlevel 1 (
  echo.
  echo AJIN Die Insight 가 이미 실행 중입니다.
  echo 브라우저에서 http://127.0.0.1:3000 을 여세요.
  popd
  exit /b 0
)

echo.
echo AJIN Die Insight 를 시작합니다...
echo 브라우저에서 http://127.0.0.1:3000 을 여세요.
echo 종료하려면 Ctrl+C 를 누르세요.
echo.

call "node_modules\.bin\vinext.cmd" dev
set "AJIN_EXIT=%ERRORLEVEL%"
popd
exit /b %AJIN_EXIT%
