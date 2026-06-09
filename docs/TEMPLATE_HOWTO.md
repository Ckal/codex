# Template How-To: Build A New Domain App

This repository is a local-first Gradio AI app template. The base workbench provides shared
patterns for model configuration, field notes, tracking, export planning, tests, docs, and
deployment. A domain app is a focused product built around those patterns.

Use `plant/` as the first reference domain app.

## Core Principle

Do not start by training a model. Start by shipping a useful zero-shot or demo-mode workflow:

```text
domain idea
  -> user story
  -> schema
  -> model choice
  -> focused UI
  -> correction loop
  -> export data
  -> optional fine-tune
  -> deploy and document
```

Training is a later optimization after you have corrected examples and a reason to tune.

## Recommended Branch Flow

1. Keep `main` as the reusable template.
2. Create a branch for each app:

   ```powershell
   git checkout -b plant-discovery-app
   ```

3. Build the app under a domain folder such as `plant/`, `invoice/`, `recipe/`, or `field_notes/`.
4. Keep domain-specific heavy requirements in `<domain>/requirements.txt`.
5. Merge reusable improvements back into `main` only after they are generic.

## Domain App File Contract

Each generated app should have these files:

```text
<domain>/
  __init__.py
  app.py              # standalone Gradio entrypoint
  models.yaml         # domain config, model IDs, data sources, training defaults
  <domain>_service.py # optional real model adapter plus demo/no-model fallback
  <domain>_loader.py  # data loading, schema normalization, export rows
  <domain>_tab.py     # focused Gradio UI
  <domain>_tools.py   # optional MCP/local tools with no hard optional imports
  requirements.txt    # optional heavy dependencies for this app only
```

Add tests under:

```text
tests/unit/test_<domain>_reference_app.py
```

Add docs under:

```text
docs/<DOMAIN>_APP_PLAN.md
```

## Step-By-Step Build Process

### 1. Define The Product

- [ ] Pick one user.
- [ ] Pick one job they need done.
- [ ] Write one sentence: "This app helps X do Y without Z."
- [ ] Choose one golden path that works in under two minutes.
- [ ] Decide whether the app is a standalone product or a tab inside the workbench.
- [ ] Decide whether it must run on a public Hugging Face Space.

Example:

> Plant Discovery helps gardeners identify a plant from a photo, correct mistakes, and export
> local training examples without sending private field notes to a cloud API.

### 2. Define The Domain Schema

- [ ] Create a dataclass for the structured output.
- [ ] Include confidence and model metadata.
- [ ] Include a `to_dict()` method for Gradio JSON.
- [ ] Add a robust parser for model responses.
- [ ] Add tests for valid JSON, fenced JSON, trailing commas, and unparseable text.

Plant example: `PlantID` in `plant/plant_service.py`.

### 3. Pick The Model

- [ ] Pick a small model at or below 32B parameters.
- [ ] Document the exact model ID.
- [ ] Add model metadata to `<domain>/models.yaml`.
- [ ] Avoid loading weights on startup.
- [ ] Add a deterministic demo/no-model service for screenshots and tests.
- [ ] Add an unavailable-path response when optional packages are missing.
- [ ] Add explicit runtime modes such as `demo`, `base-model`, and `finetuned`.
- [ ] Do not claim a fine-tuned model until a real adapter/checkpoint is configured and verified.

For vision apps, start with a VLM such as MiniCPM-V. For text apps, start with a small instruct
model through LM Studio, Ollama, llama.cpp, or Transformers.

### 4. Build The Focused UI

- [ ] Make the first screen the golden path, not a generic dashboard.
- [ ] Add only the controls needed for the user story.
- [ ] Keep advanced setup behind a secondary tab or accordion.
- [ ] Add visible status messages.
- [ ] Add structured JSON output for debugging and reproducibility.
- [ ] Add correction capture if model output can be wrong.
- [ ] Add screenshots through Playwright after the UI is stable.

### 5. Add The Correction Loop

- [ ] Save user corrections locally.
- [ ] Reuse `datasets.field_notes.FieldNoteStore` where possible.
- [ ] Mark training-ready rows explicitly.
- [ ] Export JSONL without starting training.
- [ ] Add tests for save, filter, and export.

### 6. Add Data Loaders

- [ ] Support a small local demo dataset.
- [ ] Support domain data from local folders or CSV/JSONL.
- [ ] Keep Hugging Face dataset loading optional and explicit.
- [ ] Do not download large datasets on startup.
- [ ] Normalize every source into one training row schema.
- [ ] Add loader tests with temporary local files.

### 7. Add Optional Tools

- [ ] Keep MCP/tool imports optional.
- [ ] Tool functions should work locally without starting a server.
- [ ] Add `build_mcp_server()` only if `mcp` is installed.
- [ ] Avoid direct shell execution from tools.
- [ ] Return command plans rather than running commands.
- [ ] Add tests for pure tool functions.

### 8. Add Training Plans

- [ ] Start with a non-executing training plan.
- [ ] Include required dependencies, hardware notes, and command preview.
- [ ] Require enough corrected examples before recommending training.
- [ ] Keep real training as a separate local command or approved action.
- [ ] Add evaluation before/after tuning.
- [ ] Add a small script that prints the training plan as JSON.

### 9. Add Security Guardrails

- [ ] Escape model text rendered as HTML.
- [ ] Restrict file paths in public Space mode.
- [ ] Disable arbitrary backend URL checks in public Space mode.
- [ ] Do not execute subprocesses from Gradio callbacks.
- [ ] Keep tokens, private data, model weights, and exports out of git.
- [ ] Add tests for path traversal and malformed inputs when public deployment is planned.

### 10. Verify The App

Minimum local verification:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_<domain>_reference_app.py -q
.venv\Scripts\ruff.exe check <domain> tests/unit/test_<domain>_reference_app.py --no-cache
.venv\Scripts\python.exe -m mypy <domain> tests/unit/test_<domain>_reference_app.py --cache-dir "$env:TEMP\openbmb-workbench-mypy-cache"
.venv\Scripts\python.exe -c "from <domain>.app import build_app; app=build_app(no_model=True); print(type(app).__name__)"
```

Before claiming it works:

- [ ] Run the standalone app.
- [ ] Generate screenshots.
- [ ] Add screenshot links to README/docs.
- [ ] Run full quality checks.
- [ ] Commit and push.

## When To Integrate Into The Main Workbench

Keep the domain app standalone if:

- it has its own brand/story,
- it needs a focused judging experience,
- it has domain-specific dependencies,
- it should become a Hugging Face Space.

Add it to the main workbench only if:

- it is a generic reusable tab,
- it does not add heavy dependencies,
- it strengthens the template for all future apps.

For the hackathon, standalone `plant/` is the better route because judges need one clear product.

## What "Done" Means For A Domain App

- [ ] Standalone no-model app builds.
- [ ] Optional real model adapter is documented and lazy-loaded.
- [ ] Golden path has tests.
- [ ] Corrections export to training data.
- [ ] Training is planned, not accidentally executed.
- [ ] Screenshots are generated.
- [ ] README explains setup, model choice, demo flow, and limitations.
- [ ] Space deployment is verified or blocker is documented.
