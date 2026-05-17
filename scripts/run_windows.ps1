param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvRoot = Join-Path $ProjectRoot ".alife_env"
$VenvDir = Join-Path $EnvRoot "windows_venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$HashFile = Join-Path $EnvRoot "windows_requirements.sha256"
$MinVersion = [version]"3.11.0"

function Write-Section([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-CompatiblePython {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.13") },
        @{ Exe = "py"; Args = @("-3.12") },
        @{ Exe = "py"; Args = @("-3.11") },
        @{ Exe = "py"; Args = @("-3") },
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() }
    )

    $probe = 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|{sys.executable}")'

    foreach ($candidate in $candidates) {
        try {
            $out = & $candidate.Exe @($candidate.Args + @("-c", $probe)) 2>$null
            if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) {
                continue
            }

            $parts = "$out".Trim().Split("|")
            if ($parts.Count -lt 2) {
                continue
            }

            $version = [version]$parts[0]
            if ($version -ge $MinVersion) {
                return [pscustomobject]@{
                    Exe = $candidate.Exe
                    Args = $candidate.Args
                    Version = $version
                    Path = $parts[1]
                }
            }
        } catch {
            continue
        }
    }

    return $null
}

function Invoke-BasePython($PythonInfo, [string[]]$Args) {
    & $PythonInfo.Exe @($PythonInfo.Args + $Args)
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $EnvRoot | Out-Null

Write-Section "Checking Python"
$PythonInfo = Find-CompatiblePython
if ($null -eq $PythonInfo) {
    Write-Host "Python 3.11+ was not found." -ForegroundColor Red
    Write-Host "Install Python 3.11 or newer, then double-click PLAY_WINDOWS.bat again."
    Write-Host "Recommended: https://www.python.org/downloads/windows/"
    Write-Host "During install, enable: Add python.exe to PATH."
    exit 1
}
Write-Host "Using Python $($PythonInfo.Version) at $($PythonInfo.Path)"

$created = $false
if (-not (Test-Path $VenvPython)) {
    Write-Section "Creating local virtual environment"
    Invoke-BasePython $PythonInfo @("-m", "venv", $VenvDir)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Could not create the virtual environment." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $created = $true
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment Python was not created correctly: $VenvPython" -ForegroundColor Red
    exit 1
}

$currentHash = ""
if (Test-Path $Requirements) {
    $currentHash = (Get-FileHash -Algorithm SHA256 $Requirements).Hash
}
$previousHash = ""
if (Test-Path $HashFile) {
    $previousHash = (Get-Content $HashFile -Raw).Trim()
}

$needsInstall = $created -or ($currentHash -ne $previousHash)

if ($needsInstall) {
    Write-Section "Installing/updating dependencies"
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip upgrade failed; trying ensurepip." -ForegroundColor Yellow
        & $VenvPython -m ensurepip --upgrade
        & $VenvPython -m pip install --upgrade pip
    }

    if (Test-Path $Requirements) {
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Dependency installation failed." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        Set-Content -Path $HashFile -Value $currentHash -NoNewline
    } else {
        Write-Host "requirements.txt not found; skipping dependency install." -ForegroundColor Yellow
    }
} else {
    Write-Section "Dependencies already up to date"
}

Write-Section "Starting Artificial Life Sandbox"
& $VenvPython (Join-Path $ProjectRoot "run.py") @RunArgs
exit $LASTEXITCODE
