$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$uiRoot = $PSScriptRoot
$repoRoot = Split-Path (Split-Path $uiRoot -Parent) -Parent
$runtimeDir = Join-Path $uiRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

function Fail([string]$Message, [string]$Hint = '') {
  Write-Host "`n[ERROR] $Message" -ForegroundColor Red
  if ($Hint) { Write-Host "       $Hint" -ForegroundColor Yellow }
  exit 1
}

function Test-TcpPort([int]$Port) {
  $client = New-Object Net.Sockets.TcpClient
  try { $client.Connect('127.0.0.1', $Port); return $true }
  catch { return $false }
  finally { $client.Dispose() }
}

function Wait-HttpHealth([int]$Seconds) {
  $until = (Get-Date).AddSeconds($Seconds)
  do {
    try {
      $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
      if ($response.ok) { return $true }
    } catch { }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $until)
  return $false
}

function Find-Node {
  if ($env:AJIN_NODE -and (Test-Path -LiteralPath $env:AJIN_NODE -PathType Leaf)) {
    return (Resolve-Path -LiteralPath $env:AJIN_NODE).Path
  }
  $command = Get-Command node.exe -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }

  $candidates = [System.Collections.Generic.List[string]]@(
    (Join-Path $env:ProgramFiles 'nodejs\node.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'nodejs\node.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\nodejs\node.exe'),
    (Join-Path $env:LOCALAPPDATA 'Volta\bin\node.exe'),
    (Join-Path $env:USERPROFILE 'scoop\apps\nodejs\current\node.exe'),
    (Join-Path $env:USERPROFILE 'scoop\apps\nodejs-lts\current\node.exe')
  )
  $nvmRoot = if ($env:NVM_HOME) { $env:NVM_HOME } else { Join-Path $env:APPDATA 'nvm' }
  if (Test-Path -LiteralPath $nvmRoot -PathType Container) {
    Get-ChildItem -LiteralPath $nvmRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending |
      ForEach-Object { $candidates.Add((Join-Path $_.FullName 'node.exe')) }
  }
  $codexRuntimeRoot = Join-Path $env:LOCALAPPDATA 'OpenAI\Codex\runtimes\cua_node'
  if (Test-Path -LiteralPath $codexRuntimeRoot -PathType Container) {
    Get-ChildItem -LiteralPath $codexRuntimeRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      ForEach-Object { $candidates.Add((Join-Path $_.FullName 'bin\node.exe')) }
  }
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
      try {
        $null = & $candidate --version 2>$null
        if ($LASTEXITCODE -eq 0) { return $candidate }
      } catch { }
    }
  }
  return $null
}

function Invoke-Pnpm([string[]]$Arguments) {
  if ($script:PnpmPath) { & $script:PnpmPath @Arguments; return $LASTEXITCODE }
  & $script:NodePath $script:CorepackPath pnpm @Arguments
  return $LASTEXITCODE
}

Push-Location $uiRoot
try {
  # Do not silently download a model in a different deployment environment.
  $env:HF_HUB_OFFLINE = '1'
  $env:TRANSFORMERS_OFFLINE = '1'
  $env:TOKENIZERS_PARALLELISM = 'false'

  $script:NodePath = Find-Node
  if (-not $NodePath) { Fail 'Node.js를 찾지 못했습니다.' 'Node.js 22 LTS를 설치한 뒤 다시 실행하세요: https://nodejs.org' }
  $nodeVersion = (& $NodePath --version).Trim()
  $nodeMajor = [int](($nodeVersion -replace '^v', '').Split('.')[0])
  if ($nodeMajor -lt 22) { Fail "Node.js $nodeVersion 는 지원하지 않습니다." 'Node.js 22 이상이 필요합니다.' }

  $pnpmCommand = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
  $script:PnpmPath = if ($pnpmCommand) { $pnpmCommand.Source } else { $null }
  $script:CorepackPath = Join-Path (Split-Path $NodePath -Parent) 'corepack.cmd'

  $pythonCandidates = @()
  if ($env:AJIN_PYTHON) { $pythonCandidates += $env:AJIN_PYTHON }
  $pythonCandidates += (Join-Path $repoRoot '.venv\Scripts\python.exe')
  $pythonCandidates += (Join-Path (Split-Path $uiRoot -Parent) '.venv\Scripts\python.exe')
  $pythonCandidates += (Get-Command python.exe -ErrorAction SilentlyContinue | ForEach-Object Source)
  $pythonPath = $pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
  if (-not $pythonPath) { Fail '엔진 Python을 찾지 못했습니다.' "저장소 루트에 .venv를 만들거나 AJIN_PYTHON을 지정하세요. 예상 경로: $repoRoot\.venv\Scripts\python.exe" }

  Write-Host "Node: $nodeVersion"
  Write-Host "Python: $pythonPath"
  # 서버 시작에 반드시 필요한 모듈만 검사한다. OCP와
  # fast_simplification은 STEP/CAD 기능을 사용할 때만 지연 로드된다.
  & $pythonPath -c "import cv2,numpy,uvicorn,starlette,trimesh,openpyxl,scipy"
  if ($LASTEXITCODE -ne 0) {
    Fail 'Python 엔진 의존성이 누락되었습니다.' "& '$pythonPath' -m pip install -r '$uiRoot\backend\requirements.txt'"
  }

  # node_modules exists even after a partial install; verify every imported
  # runtime package instead of checking only vinext.
  $vinextCli = Join-Path $uiRoot 'node_modules\vinext\dist\cli.js'
  $threePackage = Join-Path $uiRoot 'node_modules\three\package.json'
  if (-not (Test-Path -LiteralPath $vinextCli -PathType Leaf) -or
      -not (Test-Path -LiteralPath $threePackage -PathType Leaf)) {
    if (-not $PnpmPath -and -not (Test-Path -LiteralPath $CorepackPath)) {
      Fail 'UI 의존성이 누락되었고 pnpm 또는 Corepack도 찾지 못했습니다.' 'Node.js 22 LTS를 설치한 뒤 pnpm install을 실행하세요.'
    }
    Write-Host 'UI dependencies are missing. Installing from pnpm-lock.yaml...'
    if ((Invoke-Pnpm @('install', '--frozen-lockfile')) -ne 0) {
      Fail 'UI package installation failed.' 'pnpm-lock.yaml과 package.json이 같은 커밋인지 확인하세요. 잠금파일을 임의로 지우지 마세요.'
    }
  }

  if (-not (Test-TcpPort 8000)) {
    $backendOut = Join-Path $runtimeDir 'backend.out.log'
    $backendErr = Join-Path $runtimeDir 'backend.err.log'
    Remove-Item -LiteralPath $backendOut, $backendErr -Force -ErrorAction SilentlyContinue
    Write-Host 'Starting engine server (http://127.0.0.1:8000)...'
    Start-Process -FilePath $pythonPath -ArgumentList @((Join-Path $uiRoot 'backend\server.py')) -WorkingDirectory $uiRoot -WindowStyle Hidden -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr | Out-Null
    if (-not (Wait-HttpHealth 35)) {
      $tail = if (Test-Path -LiteralPath $backendErr) { (Get-Content -LiteralPath $backendErr -Tail 20) -join "`n" } else { '(backend.err.log 없음)' }
      Fail '엔진 서버가 35초 안에 시작되지 않았습니다.' "로그: $backendErr`n$tail"
    }
  } elseif (-not (Wait-HttpHealth 3)) {
    Fail '8000 포트를 다른 프로세스가 사용 중이지만 AJIN 엔진 서버가 아닙니다.' '해당 프로세스를 종료한 뒤 다시 실행하세요.'
  }

  if (Test-TcpPort 3000) {
    Write-Host 'AJIN Die Insight is already running: http://127.0.0.1:3000'
    exit 0
  }

  Write-Host "`nAJIN Die Insight: http://127.0.0.1:3000"
  Write-Host 'Press Ctrl+C to stop the UI server.'
  if (Test-Path -LiteralPath $vinextCli -PathType Leaf) {
    & $NodePath $vinextCli dev
    exit $LASTEXITCODE
  }
  exit (Invoke-Pnpm @('dev'))
}
catch { Fail $_.Exception.Message }
finally { Pop-Location }

