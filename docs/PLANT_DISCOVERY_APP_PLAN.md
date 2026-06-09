# Plant Discovery Reference App Plan

This checklist turns the rough `plant/` sample into the first concrete app built around the
template. It is intentionally detailed so future apps can copy the pattern.

## Product Story

Plant Discovery helps a gardener, teacher, or field-note collector:

1. Upload one or more plant images.
2. Get a structured plant identification.
3. Correct mistakes.
4. Browse a small field guide.
5. Export corrections as local training data.
6. Later fine-tune or evaluate a small vision model.

This is stronger than a generic workbench for hackathon judging because it has a real user outcome.

## Current Implementation Status

- [x] Add `plant/` package.
- [x] Add standalone `plant/app.py`.
- [x] Add clean `plant/models.yaml`.
- [x] Add deterministic no-model demo service.
- [x] Add optional MiniCPM-V service adapter.
- [x] Add explicit `demo`, `openbmb`, and `finetuned` model modes.
- [x] Make OpenBMB MiniCPM-V the default real model mode.
- [x] Add fine-tuned adapter loading path through PEFT when configured.
- [x] Avoid hard `torch`/`transformers` imports at module import time.
- [x] Add plant schema/parser through `PlantID`.
- [x] Add local species index builder.
- [x] Add local folder loader.
- [x] Add field-note export to plant training JSONL.
- [x] Add focused Gradio UI with Identify, Field Guide, Corrections, and Stats.
- [x] Replace direct training execution with non-executing training plan.
- [x] Add optional pure tool functions and lazy MCP server builder.
- [x] Add non-executing plant training planner.
- [x] Add Plant model/training how-to.
- [x] Add plant unit tests.
- [x] Add no-model app build verification.

## Remaining Work

### P0 - Make The Demo Judgeable

- [x] Run `python -m plant.app --no-model --port 7861`.
- [ ] Install Node.js/npm.
- [ ] Generate Playwright screenshots for the plant app.
- [ ] Add Plant Discovery screenshots to README.
- [ ] Add a 60-90 second demo script focused on plant correction.
- [ ] Decide whether the hackathon Space should launch `app.py` or `plant/app.py`.
- [ ] If using `plant/app.py` for Space, add a Space-specific entrypoint or README note.

### P1 - Real Model Path

- [ ] Install optional plant dependencies from `plant/requirements.txt`.
- [ ] Verify `PlantVisionService.dependency_report()` is fully available.
- [ ] Run one MiniCPM-V identification on a local public-safe plant image.
- [ ] Capture model ID, hardware, latency, and output JSON.
- [ ] Add a real-backend integration test profile that is skipped unless dependencies are present.
- [ ] Document that model weights are not downloaded on startup.

### P2 - Data And Correction Loop

- [ ] Add a tiny public-safe sample plant image folder under ignored local data.
- [ ] Add a sample `plant/data/plantnet_labels.json` cache or document how to create it.
- [ ] Add path allowlist before public Space deployment.
- [ ] Add JSONL schema documentation for exported corrections.
- [ ] Add validation for minimum correction count before recommending training.

### P3 - Template Extraction

- [ ] Identify reusable parts that should move from `plant/` to generic modules.
- [ ] Keep domain-specific prompt/schema/UI inside `plant/`.
- [ ] Consider `domains/<name>/` convention if more than one reference app is added.
- [ ] Add a cookiecutter-style checklist or scaffold script only after the second app proves the pattern.

### P4 - Security And Public Mode

- [ ] Add public/local mode config.
- [ ] In public mode, disable arbitrary local file paths.
- [ ] In public mode, disable arbitrary backend URLs.
- [ ] Add max upload size and allowed image extension checks.
- [ ] Add tests for path traversal and malformed image/data inputs.
- [ ] Add a plant-specific disclaimer: identification can be wrong; do not use for medical, edible,
  or toxicity-critical decisions without expert verification.

### P5 - Training And Evaluation

- [ ] Collect at least 30 corrected examples before any LoRA experiment.
- [ ] Split exported corrections into train/eval.
- [ ] Add exact species-name evaluation.
- [ ] Add before/after model comparison.
- [ ] Run SWIFT or LLaMA-Factory locally only after dependencies and hardware are approved.
- [ ] Publish adapter only if it improves the evaluation set.
- [ ] Update `plant/models.yaml` with the real adapter repo.
- [ ] Verify `python -m plant.app --model-mode finetuned --port 7861`.

## Recommended Next Command Sequence

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_plant_reference_app.py -q
.venv\Scripts\ruff.exe check plant tests/unit/test_plant_reference_app.py --no-cache
.venv\Scripts\python.exe -m mypy plant tests/unit/test_plant_reference_app.py --cache-dir "$env:TEMP\openbmb-workbench-mypy-cache"
.venv\Scripts\python.exe -m plant.app --no-model --port 7861
.venv\Scripts\python.exe scripts\plan_plant_training.py --corrected-examples 30
```

## Template Lessons From Plant Discovery

- Domain app first screen should be the product, not the infrastructure.
- Demo/no-model mode is essential for tests and screenshots.
- Heavy model dependencies should be optional.
- Correction loops are more valuable than promising immediate fine-tuning.
- Tool/MCP functions should be pure and importable without a running server.
- Training should be a plan until data, hardware, and dependencies are verified.
