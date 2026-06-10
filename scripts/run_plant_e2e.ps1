$ErrorActionPreference = "Stop"

$env:WORKBENCH_DEPLOYMENT = "space"
$env:PORT = "7861"
$env:PYTHONUTF8 = "1"
$env:PLANT_MAX_NEW_TOKENS = "320"
$env:PLANT_AUTO_THINKING = "0"

$server = Start-Process python -ArgumentList "plant_space_app.py" -PassThru -WindowStyle Hidden
try {
    Start-Sleep -Seconds 8
    $env:E2E_NO_WEBSERVER = "1"
    $env:E2E_BASE_URL = "http://127.0.0.1:7861"
    npx playwright test tests/e2e/plant_real_model.spec.ts --config playwright.plant.config.ts
    exit $LASTEXITCODE
}
finally {
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
