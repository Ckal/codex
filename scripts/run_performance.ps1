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

$pytestCache = "$env:TEMP\openbmb-workbench-pytest-cache"

& $python -m pytest tests/performance -m performance -o "cache_dir=$pytestCache"
if ($LASTEXITCODE -ne 0) {
    throw "Performance tests failed"
}
