$ErrorActionPreference = "Stop"

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
    try {
        & $python --version | Out-Null
    } catch {
        $python = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe"
    }
}

& $python scripts/smoke_app.py
if ($LASTEXITCODE -ne 0) {
    throw "App smoke test failed"
}
