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
| Gradio app shell | Implemented and launch-verified | `app.py` builds and foreground launch holds the server open; not currently running |
| Model config | Implemented | `config/models.yaml` exists |
| Expanded model config | Implemented | MiniCPM text, thinking, 4.1, V 4.6, V thinking, omnimodal entries, GGUF metadata, and backend capability metadata including OpenAI-compatible text serving |
| Catalog validation | Implemented | `validate_catalog()` and Status tab warnings |
| Training config | Implemented | `config/training.yaml` exists |
| Placeholder chat | Implemented | `ui/chat_tab.py`, `models/placeholder_service.py` |
| Placeholder vision | Implemented | `ui/vision_tab.py`, `models/placeholder_service.py` |
| Service abstraction | Implemented | `models/base.py`, `models/service_factory.py` |
| Local backend config | Implemented | `models/local_backend_config.py` saves ignored local settings in `data/local_backends.yaml` |
| llama.cpp backend | Implemented, not locally verified | `models/llama_cpp_service.py`; Status tab can pick GGUF/mmproj paths and build a `llama-server` command; llama.cpp tools not found on PATH |
| llama-cpp-python backend | Implemented as fallback, not locally verified | `models/llama_cpp_python_service.py`; uses the configured local GGUF path when package is installed; `llama_cpp` package not installed |
| Ollama backend | Implemented, not locally verified | `models/ollama_service.py`; Status tab lists local Ollama models and prepares explicit `ollama pull` commands; Ollama executable not found on PATH |
| OpenAI-compatible backend | Implemented and live-verified | `models/openai_compatible_service.py`; Status tab stores LM Studio/vLLM-style base URL and optional served model name, checks `/v1/models`, and posts to `/v1/chat/completions` only when selected; verified `http://192.168.188.37:1234` with `llama-3.2-1b-instruct` |
| Transformers text backend | Implemented, not locally verified | `models/transformers_text.py`; lazy-loads tokenizer/model only when selected; `transformers` package/model weights are not installed or downloaded automatically |
| App state | Implemented | `core/app_state.py` records local events and dispatches through `EventBus`; `core/tab_feedback.py` emits tab-level UI errors |
| Service registry | Implemented | `models/service_factory.py` registers text and vision backend factories |
| Dataset tab | Partial | Local CSV/JSONL preview, optional HF dataset preview, schema, split selector, row count, samples, stats, dataset event emission, and tab-level error status |
| Local dataset preview | Implemented | CSV, JSONL, NDJSON preview and statistics via `datasets/loader.py` |
| MCP tools | Implemented locally, Gradio MCP path selected | `mcp_tools/tools.py` provides dataset stats, HF dataset preview/search-style helper, safe calculator, and model inference tool functions; `mcp_tools/bridge.py` documents `/gradio_api/mcp/sse` and verifies local invocation |
| VINDEX boundary | Implemented locally, execution disabled | `mcp_tools/vindex_tool.py` validates the eight PRD methods, builds non-executing call plans, caps risky edit parameters, and reports local package/server availability |
| Training tab | Partial | `ui/train_tab.py` builds a LoRA dry-run plan, checkpoint output path, hardware notes, local deterministic evaluation, and optional loss-based perplexity summary |
| Training planner | Implemented, non-executing | `training/planner.py` parses LoRA/training config, validates dry runs, and never starts training |
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
| Status tab | Implemented | `ui/status_tab.py` lists model config, backend status, local llama.cpp setup, LM Studio/OpenAI-compatible setup, and Ollama list/pull planning |
| Tab-level error messages | Implemented | Chat, Vision, and Dataset tabs show status/error messages and emit `ui_error` trace events |
| Loading/progress states | Implemented | `ui/progress.py` applies full Gradio progress indicators to tab actions |
| Compact responsive layout | Implemented | `APP_CSS` constrains app width, keeps tabs scrollable, sizes touch targets, and adds mobile padding/type rules |
| Structure verification | Done | `scripts/verify_structure.ps1` passed |
| Unit tests | Passing | 145 unit/user-story tests pass |
| User-story tests | Passing | Included in the 145-test suite |
| Coverage | Passing | 67% line/branch coverage at current configured threshold |
| Performance tests | Passing | 2 lightweight performance tests pass |
| CI pipeline | Added, not run remotely | `.github/workflows/ci.yml` |
| Quality tooling | Passing | Tests, coverage, performance, ruff, mypy, pylint, bandit, and pip-audit pass through `scripts/run_quality.ps1` |
| Secrets and model-weight git policy | Implemented | `.gitignore` excludes env files, keys, caches, generated data/exports, and common model weight formats; policy has a unit test |
| Real model inference | Partial | OpenAI-compatible/LM Studio text generation is live-verified through `llama-3.2-1b-instruct`; llama.cpp, llama-cpp-python, Ollama, and Transformers text services exist but remain unverified locally |
| Hugging Face Space deploy | Not started | Needs HF login/repo |
| HF Space deployment helper | Implemented locally | `deployment/hf_space.py` and `scripts/plan_hf_space.py` validate required files, README Space metadata, remote status, and manual deployment commands |
| GitHub push | Done | GitHub remote `https://github.com/Ckal/codex.git`; commits pushed to `origin/main` |

## Known Blockers

- `python` and `py` are not available on PATH in the current shell.
- Direct WindowsApps Python works: `%LOCALAPPDATA%\Microsoft\WindowsApps\python3.11.exe`.
- Project venv exists at `.venv`.
- App smoke launch was rerun after the latest code changes, but no long-running server is currently active.
- llama.cpp tools are not on PATH.
- Export planning works without llama.cpp tools, but actual conversion/quantization remains blocked
  until llama.cpp tooling is installed.
- A GGUF path can now be configured from the Status tab, but no GGUF model file has been selected
  and verified in this workspace.
- `llama_cpp` Python package is not installed in `.venv`.
- Ollama is not on PATH.
- Ollama setup/list/pull command planning is implemented in the Status tab, but real list/pull and
  generation still require installing and starting Ollama locally.
- LM Studio/OpenAI-compatible setup is implemented and verified for `http://192.168.188.37:1234`
  with served model override `llama-3.2-1b-instruct`. This local override is stored in ignored
  `data/local_backends.yaml`.
- `transformers`/`torch` are not installed for real local Transformers inference; selecting that
  backend reports a clear unavailable status until the packages and model weights are available.
- Trackio is optional; local JSONL tracing works without the `trackio` package, but remote Trackio/HF
  sync still needs package availability and credentials.
- LoRA dry-run planning works locally, but real training remains blocked until a final backend,
  PEFT/TRL or SWIFT/LLaMA-Factory path, and hardware are chosen.
- Hugging Face dataset preview is optional and requires the external `datasets` package; the app
  reports a clear status when it is not installed.
- Full PRD implementation is not complete. There are still unchecked tasks in `docs/TASKS.md`.
- Current unchecked task count is 56 because many PRD/ext PRD items still need real local setup,
  credentials, hardware, product decisions, or hackathon submission artifacts.

## Latest Local Verification

- `powershell -ExecutionPolicy Bypass -File scripts/run_quality.ps1` passed all gates: tests,
  smoke, coverage, performance, ruff, mypy, pylint, bandit, and pip-audit.
- `powershell -ExecutionPolicy Bypass -File scripts/run_tests.ps1` passed: 136 tests, 66% coverage
  before the callback refactor; final all-in-one quality passed with 145 tests and 67% coverage.
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
