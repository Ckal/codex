# Implementation Status

This file answers: "How do we know what is done?"

An item is done only when:

1. The code or document exists.
2. It is linked from the relevant docs.
3. It has been manually reviewed.
4. If it is executable, it has been run locally or the blocker is documented.

## Current Status

| Area | Status | Evidence |
| --- | --- | --- |
| PRDs | Done | `HF_PRD_v1.md`, `HF_PRD_ext.md` exist at repo root |
| Codex guidance | Done | `AGENTS.md` exists |
| Docs folder | Done | `docs/` exists with task, usage, architecture, extension docs |
| README | Done | `README.md` exists |
| Hackathon submission package | Drafted | `docs/HACKATHON_SUBMISSION.md` contains track, story, user, benefit, model compliance, demo flow, video script, social post draft, and submission checklist |
| Critical improvement roadmap | Done | `docs/ROADMAP_V2_CRITICAL_IMPROVEMENT_PLAN.md` gives a hard judge-oriented rating, architecture/security critique, and second roadmap |
| Template how-to | Done | `docs/TEMPLATE_HOWTO.md` documents how to build new domain apps from the template |
| Plant Discovery plan | Done | `docs/PLANT_DISCOVERY_APP_PLAN.md` tracks the first reference app and remaining work |
| Plant model/training how-to | Done | `docs/PLANT_MODEL_AND_TRAINING_HOWTO.md` documents demo, OpenBMB zero-shot, fine-tuned adapter mode, correction export, training plan, and adapter configuration |
| Gradio app shell | Implemented and launch-verified | `app.py` builds and foreground launch holds the server open; not currently running |
| Model config | Implemented | `config/models.yaml` exists |
| Expanded model config | Implemented | MiniCPM text, thinking, 4.1, V 4.6, V thinking, omnimodal entries, NVIDIA Nemotron Nano 9B v2, GGUF metadata, and backend capability metadata including OpenAI-compatible text serving |
| Catalog validation | Implemented | `validate_catalog()` and Status tab warnings |
| Training config | Implemented | `config/training.yaml` exists |
| Placeholder chat | Implemented | `ui/chat_tab.py`, `models/placeholder_service.py` |
| Placeholder vision | Implemented | `ui/vision_tab.py`, `models/placeholder_service.py` |
| Service abstraction | Implemented | `models/base.py`, `models/service_factory.py` |
| Local backend config | Implemented | `models/local_backend_config.py` saves ignored local settings in `data/local_backends.yaml` |
| llama.cpp backend | Implemented and locally verified for text CLI | `models/llama_cpp_service.py`; Status tab can pick GGUF/mmproj paths and build a command using `C:\llama-b9587-bin-win-cuda-13.3-x64\llama-server.exe`; `llama-cli.exe` generated a real response from a local GGUF |
| llama-cpp-python backend | Implemented and locally verified | `models/llama_cpp_python_service.py`; `llama_cpp 0.3.8` generated a real local response from `Llama-3.2-1B-Instruct-Q4_K_M.gguf`; Workbench Playwright captures this response |
| Ollama backend | Implemented, not locally verified | `models/ollama_service.py`; Status tab lists local Ollama models and prepares explicit `ollama pull` commands; Ollama executable not found on PATH |
| OpenAI-compatible backend | Implemented and live-verified | `models/openai_compatible_service.py`; Status tab stores LM Studio/vLLM-style base URL and optional served model name, checks `/v1/models`, and posts to `/v1/chat/completions` only when selected; verified `http://192.168.188.37:1234` with `llama-3.2-1b-instruct` |
| Transformers text backend | Implemented, package installed | `models/transformers_text.py`; lazy-loads tokenizer/model only when selected; `transformers 5.10.2` is installed for MiniCPM-V support |
| MiniCPM vision backend | Implemented and Plant-verified | `models/minicpm_vision.py` and `plant/plant_service.py` use `AutoProcessor` and `AutoModelForImageTextToText` lazily; `assets/plant_sample.jpg` produced a structured OpenBMB MiniCPM-V result |
| SGLang backend | Implemented, optional local backend | `models/sglang_runner.py`; builds explicit local start commands, reports health, sends OpenAI-compatible chat requests, and provides a shutdown request; SGLang server launch remains unverified and is not a root Space dependency |
| App state | Implemented | `core/app_state.py` records local events and dispatches through `EventBus`; `core/tab_feedback.py` emits tab-level UI errors |
| Service registry | Implemented | `models/service_factory.py` registers text and vision backend factories |
| Dataset tab | Partial | Local CSV/JSONL preview, optional HF dataset preview, schema, split selector, row count, samples, stats, dataset event emission, and tab-level error status |
| Local dataset preview | Implemented | CSV, JSONL, NDJSON preview and statistics via `datasets/loader.py` |
| MCP tools | Implemented locally, Gradio MCP path selected | `mcp_tools/tools.py` provides dataset stats, HF dataset preview/search-style helper, safe calculator, and model inference tool functions; `mcp_tools/bridge.py` documents `/gradio_api/mcp/sse` and verifies local invocation |
| VINDEX boundary | Implemented locally, execution disabled | `mcp_tools/vindex_tool.py` validates the eight PRD methods, builds non-executing call plans, caps risky edit parameters, and reports local package/server availability |
| Training tab | Partial | `ui/train_tab.py` builds a LoRA dry-run plan, checkpoint output path, hardware notes, local deterministic evaluation, and optional loss-based perplexity summary |
| Training planner | Implemented, non-executing | `training/planner.py` parses LoRA/training config, validates dry runs, and never starts training |
| LoRA trainer planner | Implemented locally, execution disabled | `training/lora_trainer.py` reports PEFT/TRL/Transformers/Torch availability, builds a non-executing LoRA request, and documents SWIFT/LLaMA-Factory vision fine-tuning path |
| Evaluation | Implemented, local-only | `training/evaluation.py` provides prompt cases, exact-match scoring, optional loss-based perplexity, qualitative table, base-vs-tuned comparison, and JSONL logging |
| Export tab | Partial | `ui/export_tab.py` builds GGUF download/conversion/quantization plans, lists exported files, and exposes existing exported files through a download output |
| Export planner | Implemented, non-executing | `training/export.py` detects llama.cpp tools, builds explicit commands, and does not run downloads/conversions |
| Reward evaluation | Implemented locally | `training/reward_eval.py` provides deterministic reward scoring, best-of-N selection, DPO pair generation, and LoRA-vs-base reward reports |
| Synthetic data generation | Implemented locally | `datasets/synthetic.py` provides deterministic generation, validation, quality filtering, augmentation, and JSONL export |
| Field notes | Partial | `ui/notes_tab.py` saves CSV, supports media paths/training flag, imports uncertain OCR predictions, emits field-note events, and exports JSONL/local HF Dataset files |
| Field note module | Implemented | CSV save, SQLite store, corrected/tag/training filters, JSONL export, local HF Dataset export via `datasets/field_notes.py` |
| OCR correction loop | Implemented locally | `datasets/ocr.py` loads local CSV/JSONL OCR predictions, filters by confidence threshold, imports uncertain rows to Field Notes, and exports corrected OCR JSONL |
| Tracking | Implemented with local fallback | `tracking/trackio_client.py` loads Trackio config, writes local JSONL traces, and calls Trackio when installed/enabled |
| Traces tab | Partial | `ui/traces_tab.py` previews app events, reads local trace rows, shows tracking status, and exports traces |
| Agent mode | Implemented locally, non-autonomous | `agent/runner.py` provides system prompt, deterministic research-plan-implement-verify trace, paper-to-code trace mode, safety gates, tool registry integration, JSONL trace save/export, and local HF Dataset-style export |
| Agent tab | Implemented locally | `ui/agent_tab.py` drafts task traces, paper-to-code traces, and exports trace files/datasets |
| Status tab | Implemented | `ui/status_tab.py` lists model config, backend status, local llama.cpp setup, LM Studio/OpenAI-compatible setup, SGLang setup, and Ollama list/pull planning |
| Tab-level error messages | Implemented | Chat, Vision, and Dataset tabs show status/error messages and emit `ui_error` trace events |
| Loading/progress states | Implemented | `ui/progress.py` applies full Gradio progress indicators to tab actions |
| Compact responsive layout | Implemented | `APP_CSS` constrains app width, keeps tabs scrollable, sizes touch targets, and adds mobile padding/type rules |
| Structure verification | Done | `scripts/verify_structure.ps1` passed |
| Unit tests | Passing | 187 unit/user-story tests pass |
| User-story tests | Passing | Included in the 187-test suite |
| Coverage | Passing | 68% line/branch coverage at current configured threshold |
| Performance tests | Passing | 2 lightweight performance tests pass |
| Playwright E2E | Passing with real response screenshots | Workbench Playwright captures a real local GGUF `llama-cpp-python` chat response; Plant Playwright with `RUN_REAL_MODEL_E2E=1` captures a real OpenBMB MiniCPM-V image result from `assets/plant_sample.jpg` |
| CI pipeline | Added, not run remotely | `.github/workflows/ci.yml` |
| Quality tooling | Passing | Tests, coverage, performance, ruff, mypy, pylint, bandit, and pip-audit pass through `scripts/run_quality.ps1` |
| Secrets and model-weight git policy | Implemented | `.gitignore` excludes env files, keys, caches, generated data/exports, and common model weight formats; policy has a unit test |
| Real model inference | Partial but materially verified | Verified paths: LM Studio/OpenAI-compatible text, llama.cpp CLI text, llama-cpp-python GGUF text, and OpenBMB MiniCPM-V Plant image inference. Remaining unverified paths: Ollama generation, SGLang server generation, llama.cpp MiniCPM-V mmproj vision, full Transformers text generation |
| Hugging Face Space deploy | Pushed, startup/build pending | Workbench pushed to `build-small-hackathon/workbench` at `6aafdc2083a9b82e9dca2cca5b87c3a1be05121b`; Plant pushed to `build-small-hackathon/plant_identification_tool` at `50897b3167a844b6a66ca0552b73a1791cdff926`; both include Python 3.10 compatibility fixes for HF Spaces |
| HF Space deployment helper | Implemented locally | `deployment/hf_space.py` and `scripts/plan_hf_space.py` validate required files, README Space metadata, Workbench/Plant remote status, and manual `hf` deployment commands |
| vLLM serving tab | Implemented locally, not locally verified | `models/vllm_runner.py` and `ui/vllm_tab.py` build explicit vLLM commands, check health, parse metrics, log benchmark metrics through local tracking, and use OpenAI-compatible chat when a server is running |
| Plant Discovery reference app | Implemented locally, no-model verified; Space wrapper added | `plant/` is a standalone template-built app with demo/no-model service for local tests, default OpenBMB MiniCPM-V mode, optional fine-tuned adapter mode, local species index, correction export, non-executing training plan, optional MCP tools, unit tests, HTTP smoke verification on port 7861, and `plant_space_app.py` for real-model Space launch |
| GitHub push | Done | GitHub remote `https://github.com/Ckal/codex.git`; commits pushed to `origin/main` |

## Known Blockers

- `python` is available on PATH as Python 3.13 in the current shell. The documented `.venv` was not
  visible during the latest verification run, so tests ran against the global Python environment
  after reinstalling/updating `requirements.txt`.
- App launch was verified by Playwright through `python app.py`; no long-running server is currently active.
- llama.cpp tools are installed at `C:\llama-b9587-bin-win-cuda-13.3-x64`; they are not on PATH,
  so the app stores/uses the explicit `llama-server.exe` path.
- Export planning works, but actual conversion/quantization remains blocked until a specific export
  run is requested and verified.
- A GGUF path is configured locally for `Llama-3.2-1B-Instruct-Q4_K_M.gguf`; direct
  `llama-cli.exe` and `llama-cpp-python` generation both produced real text.
- Ollama is installed on PATH (`ollama version is 0.24.0`). `ollama pull openbmb/minicpm-v4.6`
  succeeded and `ollama list` shows `openbmb/minicpm-v4.6:latest`; a tiny `ollama run` prompt
  currently fails with a 500 model-load error, so real Ollama generation remains unverified.
- LM Studio/OpenAI-compatible setup is implemented and verified for `http://192.168.188.37:1234`
  with served model override `llama-3.2-1b-instruct`. This local override is stored in ignored
  `data/local_backends.yaml`.
- `transformers 5.10.2`, `torch`, and MiniCPM-V dependencies are installed. This conflicts with
  the currently installed `sentence-transformers 3.4.1` requirement of `transformers<5.0.0`, so
  future quality runs should watch for that dependency edge.
- MiniCPM-V Plant inference is verified with `assets/plant_sample.jpg`; the app also supports
  bounded E2E inference through `PLANT_MAX_NEW_TOKENS=320` and `PLANT_AUTO_THINKING=0`.
- SGLang command planning, health, stop, and chat client code is implemented. SGLang remains an
  optional local backend, not a root Space dependency, because the local Python 3.13 package index
  cannot install the newer non-vulnerable SGLang server stack.
- vLLM command planning, health, metrics, benchmark logging, and chat client code is implemented,
  but the `vllm` package/server is not installed or running in this workspace.
- LoRA training request planning is implemented, but real execution remains blocked until PEFT/TRL,
  Transformers, Torch, and final hardware are approved and installed.
- Trackio is optional; local JSONL tracing works without the `trackio` package, but remote Trackio/HF
  sync still needs package availability and credentials.
- LoRA dry-run planning works locally, but real training remains blocked until a final backend,
  PEFT/TRL or SWIFT/LLaMA-Factory path, and hardware are chosen.
- Hugging Face dataset preview is optional and requires the external `datasets` package; the app
  reports a clear status when it is not installed.
- Hugging Face Space deployment is pushed with Python 3.10 compatibility fixes for `StrEnum` and
  UTC timestamps. Final run verification and smoke workflows remain open until rebuilt Spaces
  report `RUNNING`.
- Node.js and npm are installed (`node v23.11.0`, `npm 10.9.2`). `npm install`,
  `npm run e2e:install`, and `npm run e2e` pass.
- Plant Discovery OpenBMB MiniCPM-V inference is implemented as the default real mode, and
  `plant_space_app.py` launches that mode for Space deployment. `RUN_REAL_MODEL_E2E=1`
  Playwright passed and captured a real response screenshot.
- Plant Discovery fine-tuned adapter mode is implemented, but no trained plant adapter exists in
  this workspace yet.
- Plant Discovery public Space mode still needs path/url hardening and screenshots.
- Full PRD implementation is not complete. There are still unchecked tasks in `docs/TASKS.md`.
- Current unchecked task count needs recounting after the latest Workbench/Space changes. Several PRD/ext PRD items still
  need real local setup, credentials, hardware, product decisions, or hackathon submission artifacts.

## Latest Local Verification

- `.\scripts\run_tests.ps1` passed: 200 tests and 70% coverage.
- `.\scripts\run_quality.ps1` passed: tests, app smoke, 70% coverage, performance, Ruff, mypy,
  Pylint, Bandit, and project-scoped pip-audit.
- Plant Discovery no-model HTTP smoke passed on `http://127.0.0.1:7861`; the process was stopped
  after verification.
- `python scripts\plan_plant_training.py --corrected-examples 30` prints a non-executing SWIFT /
  LLaMA-Factory adapter training plan; current environment is missing torch, transformers, PEFT,
  TRL, and SWIFT.
- `pytest tests/unit/test_local_backend_config.py tests/unit/test_llama_cpp_service.py tests/unit/test_model_catalog.py tests/unit/test_service_factory.py -q` passed: 30 tests.
- `npm run e2e:workbench` passed and captured a visible local GGUF response through `llama-cpp-python`.
- `RUN_REAL_MODEL_E2E=1 PLANT_MAX_NEW_TOKENS=320 PLANT_AUTO_THINKING=0 npx playwright test tests/e2e/plant_real_model.spec.ts --config playwright.plant.config.ts --reporter=list` passed and captured a real MiniCPM-V response.
- `hf upload build-small-hackathon/workbench C:\tmp\workbench_space_payload . --repo-type space` pushed commit `6aafdc2083a9b82e9dca2cca5b87c3a1be05121b`.
- `hf upload build-small-hackathon/plant_identification_tool C:\tmp\plant_space_payload . --repo-type space` pushed commit `50897b3167a844b6a66ca0552b73a1791cdff926`.
- `hf spaces variables add` set `WORKBENCH_DEPLOYMENT=space` for Workbench and Plant; Plant also has `PLANT_MAX_NEW_TOKENS=320` and `PLANT_AUTO_THINKING=0`.
- `hf spaces info` showed Workbench in `APP_STARTING` and Plant in `BUILDING` on `zero-a10g`; final build/run verification remains pending.
- `pytest tests/unit tests/user_stories -q` passed earlier in this workstream: 197 tests.
- `node --version` returned `v23.11.0`; `npm --version` returned `10.9.2`.
- `ollama --version` returned `ollama version is 0.24.0`; `ollama list` returned `gemma4:latest`.
- `ollama pull openbmb/minicpm5-1b` failed because the registry manifest does not exist.
- `ollama pull openbmb/minicpm-v4.6` succeeded; `ollama run openbmb/minicpm-v4.6` failed with
  a local 500 model-load error.
- `hf --version` returned `1.17.0`; `huggingface-cli --version` reports the legacy command is
  deprecated and recommends `hf`.
- `hf auth whoami` failed with "Invalid user token"; Space push/build verification is blocked
  until `hf auth login --force` is run with a fresh token.
- `llama-server --version` failed because `llama-server` is not on PATH.
- `git remote -v` shows `space-workbench` and `space-plant` remotes configured.
- Direct `ruff check .` passed; cache-write warnings were caused by OneDrive permissions.
- Direct `mypy . --no-incremental` passed when `MYPY_CACHE_DIR` was moved to `%TEMP%`.
- LM Studio `/v1/models` at `http://192.168.188.37:1234` returned
  `text-embedding-nomic-embed-text-v1.5`, `qwen2.5-coder-3b-instruct`, and
  `llama-3.2-1b-instruct`.
- LM Studio `/v1/chat/completions` returned a text response from `llama-3.2-1b-instruct`.
- Focused OCR callback and pipeline tests pass: 8 tests.
- Focused VINDEX/MCP tests pass: 13 tests.

## Verification Commands

Run these after installing Python:

```powershell
.\scripts\verify_structure.ps1
.venv\Scripts\python.exe --version
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
.\scripts\run_tests.ps1
.\scripts\run_quality.ps1
python app.py
```

When the app starts, update this file and `docs/TASKS.md`.
