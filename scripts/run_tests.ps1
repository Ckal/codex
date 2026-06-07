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
$coverageFile = "$env:TEMP\openbmb-workbench.coverage"

& $python -m coverage run --data-file $coverageFile -m unittest discover -s tests -p "test_*.py"
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed"
}
& $python -m coverage report --data-file $coverageFile
if ($LASTEXITCODE -ne 0) {
    throw "Coverage failed"
}
