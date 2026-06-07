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
$scripts = ".venv\Scripts"
if (-not (Test-Path $scripts)) {
    $scripts = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"
}
$ruff = "$scripts\ruff.exe"
$mypyCache = "$env:TEMP\openbmb-workbench-mypy-cache"
$pipAuditCache = "$env:TEMP\openbmb-workbench-pip-audit-cache"
$pytestCache = "$env:TEMP\openbmb-workbench-pytest-cache"
$coverageFile = "$env:TEMP\openbmb-workbench.coverage"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Command
    )

    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $($Command -join ' ')"
    }
}

Write-Host "Running tests"
Invoke-Checked @($python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")

Write-Host "Running app smoke test"
Invoke-Checked @($python, "scripts/smoke_app.py")

Write-Host "Running coverage"
Invoke-Checked @(
    $python,
    "-m",
    "coverage",
    "run",
    "--data-file",
    $coverageFile,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-p",
    "test_*.py"
)
Invoke-Checked @($python, "-m", "coverage", "report", "--data-file", $coverageFile)

Write-Host "Running performance tests"
Invoke-Checked @(
    $python,
    "-m",
    "pytest",
    "tests/performance",
    "-m",
    "performance",
    "-o",
    "cache_dir=$pytestCache"
)

Write-Host "Running ruff"
Invoke-Checked @($ruff, "check", ".", "--no-cache")

Write-Host "Running mypy"
Invoke-Checked @($python, "-m", "mypy", ".", "--cache-dir", $mypyCache)

Write-Host "Running pylint"
Invoke-Checked @($python, "-m", "pylint", "--persistent=n", "app.py", "core", "datasets", "models", "ui")

Write-Host "Running bandit"
Invoke-Checked @($python, "-m", "bandit", "-r", "app.py", "core", "datasets", "models", "ui")

Write-Host "Running pip-audit"
Invoke-Checked @($python, "-m", "pip_audit", "--cache-dir", $pipAuditCache)
