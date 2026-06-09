# AGENTS.md

## Project Goal
Build a Gradio-based OpenBMB Local AI Workbench for the Build Small Hackathon.
The app should run locally first, deploy as a Hugging Face Space, and document the
small-model choices clearly.

## Project Memory
- Start with `docs/README.md`.
- Use `docs/TASKS.md` as the task checklist.
- Use `docs/IMPLEMENTATION_STATUS.md` to decide whether something is really done.
- Use `docs/PRD_IMPLEMENTATION_MATRIX.md` before claiming PRD or extension PRD completion.
- Use `docs/ACCEPTANCE_CRITERIA.md` before marking work complete.
- Use `docs/ROADMAP.md` to prioritize hackathon work over stretch work.
- Use `docs/ROADMAP_V2_CRITICAL_IMPROVEMENT_PLAN.md` for hard judge-oriented prioritization.
- Use `docs/TEMPLATE_HOWTO.md` before creating a new domain app from the template.
- Use `docs/PLANT_DISCOVERY_APP_PLAN.md` before changing the Plant Discovery reference app.
- Use `docs/ARCHITECTURE.md` before changing app structure.
- Use `docs/EXTENDING.md` before adding models, tabs, services, or training.
- Main PRD: `HF_PRD_v1.md`.
- Extension PRD: `HF_PRD_ext.md`.

## Hackathon Constraints
- Use Gradio as the app surface.
- Keep all model choices at or below 32B parameters.
- Prefer local/offline inference paths when practical.
- Avoid cloud APIs unless the user explicitly approves them.
- Make the README useful for judges: goal, setup, model choice, local run, Space deployment, demo flow.

## Current MVP Scope
- Start with a working Gradio app and deterministic placeholder service.
- Add real inference incrementally through Ollama, llama.cpp, Transformers, or SGLang.
- Keep training, GGUF export, Trackio, MCP, and agent functionality as documented extension points until implemented.
- Never claim the full PRD is done while placeholders or unchecked PRD tasks remain.

## Commands
- Create venv: `python -m venv .venv`
- Activate venv: `.venv\Scripts\Activate.ps1`
- Install: `python -m pip install -r requirements.txt`
- Install dev tools: `python -m pip install -r requirements-dev.txt`
- Run locally: `python app.py`
- Run tests: `.\scripts\run_tests.ps1`
- Run performance tests: `.\scripts\run_performance.ps1`
- Run quality checks: `.\scripts\run_quality.ps1`

## Code Style
- Prefer small modules with clear responsibilities.
- Keep model IDs in `config/models.yaml`; avoid hardcoding them in UI logic.
- Keep user-facing text concise and demo-oriented.
- Do not download model weights automatically on app startup.

## Verification
- Before claiming the app runs, verify `python app.py` starts locally.
- If Python or dependencies are missing, say exactly what is missing.
- Keep generated caches, model weights, exports, and virtualenvs out of git.
- After implementing a task, update `docs/TASKS.md` and `docs/IMPLEMENTATION_STATUS.md`.
- Every feature needs tests. Add or update unit tests for logic and user-story tests for workflows.
- Whenever a bug, failed check, or regression is found, create or update a test that would catch it before fixing the code.
- Do not mark a feature done until its tests pass or the test blocker is documented.
- Avoid blanket coverage escapes. Do not add `pragma: no cover` without a specific documented reason.
- For PRD work, update `docs/PRD_IMPLEMENTATION_MATRIX.md` along with task/status docs.
