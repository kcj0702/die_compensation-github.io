@echo off
REM 품번 파일 정리 데모용 MariaDB를 Docker로 띄운다.
REM UI의 "MariaDB 상태 -> 데모 설정" 버튼이 채우는 접속 정보와 반드시 일치해야 한다:
REM   mysql://file_demo:file_demo_password@127.0.0.1:3307/file_organizer

where docker >nul 2>nul
if errorlevel 1 (
    echo [오류] Docker Desktop이 설치되어 있지 않거나 PATH에 없습니다.
    pause
    exit /b 1
)

cd /d "%~dp0"
docker compose up -d
if errorlevel 1 (
    echo [오류] MariaDB 컨테이너를 시작하지 못했습니다. Docker Desktop이 실행 중인지 확인하세요.
    pause
    exit /b 1
)

echo.
echo MariaDB 데모 컨테이너가 127.0.0.1:3307 에서 실행 중입니다.
echo AJIN Die Insight 화면의 "품번 파일 정리" -> MariaDB 상태 -> 데모 설정 -> 연결 테스트/저장을 눌러 연결하세요.
pause
