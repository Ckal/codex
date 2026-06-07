$ErrorActionPreference = "Stop"

$required = @(
    "AGENTS.md",
    "README.md",
    "app.py",
    "requirements.txt",
    "config/models.yaml",
    "config/training.yaml",
    "docs/README.md",
    "docs/TASKS.md",
    "docs/IMPLEMENTATION_STATUS.md",
    "docs/ACCEPTANCE_CRITERIA.md",
    "docs/ROADMAP.md",
    "docs/USAGE.md",
    "docs/ARCHITECTURE.md",
    "docs/EXTENDING.md",
    "core/events.py",
    "core/registry.py",
    "models/model_catalog.py",
    "models/response_parsing.py",
    "models/base.py",
    "models/llama_cpp_python_service.py",
    "models/llama_cpp_service.py",
    "models/ollama_service.py",
    "models/placeholder_service.py",
    "models/service_factory.py",
    "datasets/field_notes.py",
    "datasets/loader.py",
    "scripts/run_smoke.ps1",
    "scripts/smoke_app.py",
    "ui/chat_tab.py",
    "ui/vision_tab.py",
    "ui/dataset_tab.py",
    "ui/train_tab.py",
    "ui/export_tab.py",
    "ui/notes_tab.py",
    "ui/traces_tab.py",
    "ui/agent_tab.py",
    "ui/status_tab.py"
)

$missing = @()
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing required files:"
    foreach ($path in $missing) {
        Write-Host " - $path"
    }
    exit 1
}

Write-Host "Structure check passed. $($required.Count) required files found."
