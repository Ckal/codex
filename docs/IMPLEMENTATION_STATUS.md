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
| Gradio app shell | Implemented and launch-verified | `app.py` builds and foreground launch holds the server open; not currently running |
| Model config | Implemented | `config/models.yaml` exists |
| Expanded model config | Implemented | MiniCPM text, thinking, 4.1, V 4.6, V thinking, and omnimodal entries |
| Catalog validation | Implemented | `validate_catalog()` and Status tab warnings |
| Training config | Implemented | `config/training.yaml` exists |
| Placeholder chat | Implemented | `ui/chat_tab.py`, `models/placeholder_service.py` |
| Placeholder vision | Implemented | `ui/vision_tab.py`, `models/placeholder_service.py` |
| Service abstraction | Implemented | `models/base.py`, `models/service_factory.py` |
| Local backend config | Implemented | `models/local_backend_config.py` saves ignored local settings in `data/local_backends.yaml` |
| llama.cpp backend | Implemented, not locally verified | `models/llama_cpp_service.py`; Status tab can pick GGUF/mmproj paths and build a `llama-server` command; llama.cpp tools not found on PATH |
| llama-cpp-python backend | Implemented as fallback, not locally verified | `models/llama_cpp_python_service.py`; uses the configured local GGUF path when package is installed; `llama_cpp` package not installed |
| Ollama backend | Implemented, not locally verified | `models/ollama_service.py`; Ollama executable not found on PATH |
| App state | Implemented | `core/app_state.py` records local events and dispatches through `EventBus` |
| Service registry | Implemented | `models/service_factory.py` registers text and vision backend factories |
| Dataset tab | Partial | Local CSV/JSONL preview, optional HF dataset preview, schema, split selector, row count, samples, stats, and dataset event emission |
| Local dataset preview | Implemented | CSV, JSONL, NDJSON preview and statistics via `datasets/loader.py` |
| MCP tools | Implemented locally, not served | `mcp_tools/tools.py` provides dataset stats, HF dataset preview/search-style helper, safe calculator, and model inference tool functions |
| Training tab | Partial | `ui/train_tab.py` plans training and runs local deterministic evaluation |
| Evaluation | Implemented, local-only | `training/evaluation.py` provides prompt cases, exact-match scoring, qualitative table, base-vs-tuned comparison, and JSONL logging |
| Export tab | Partial | `ui/export_tab.py` builds GGUF download/conversion/quantization plans and lists exported files |
| Export planner | Implemented, non-executing | `training/export.py` detects llama.cpp tools, builds explicit commands, and does not run downloads/conversions |
| Field notes | Partial | `ui/notes_tab.py` saves CSV, supports media paths/training flag, emits field-note events, and exports JSONL/local HF Dataset files |
| Field note module | Implemented | CSV save, SQLite store, corrected/tag/training filters, JSONL export, local HF Dataset export via `datasets/field_notes.py` |
| Tracking | Implemented with local fallback | `tracking/trackio_client.py` loads Trackio config, writes local JSONL traces, and calls Trackio when installed/enabled |
| Traces tab | Partial | `ui/traces_tab.py` previews app events, reads local trace rows, shows tracking status, and exports traces |
| Agent mode | Implemented locally, non-autonomous | `agent/runner.py` provides system prompt, deterministic research-plan-implement-verify trace, tool registry integration, JSONL trace save/export, and local HF Dataset-style export |
| Agent tab | Implemented locally | `ui/agent_tab.py` drafts agent traces and exports trace files/datasets |
| Status tab | Implemented | `ui/status_tab.py` lists model config, backend status, and local llama.cpp setup |
| Structure verification | Done | `scripts/verify_structure.ps1` passed |
| Unit tests | Passing | 73 unit/user-story tests pass |
| User-story tests | Passing | Included in the 73-test suite |
| Coverage | Passing | 61% line/branch coverage at current configured threshold |
| Performance tests | Passing | 2 lightweight performance tests pass |
| CI pipeline | Added, not run remotely | `.github/workflows/ci.yml` |
| Quality tooling | Passing | Tests, coverage, performance, ruff, mypy, pylint, bandit, and pip-audit pass; all-in-one script can time out while waiting on network-backed checks |
| Real model inference | Partial | llama.cpp, llama-cpp-python, and Ollama services exist, but no installed/running backend was found locally |
| Hugging Face Space deploy | Not started | Needs HF login/repo |
| GitHub push | Not started | Needs GitHub repo/remote |

## Known Blockers

- `python` and `py` are not available on PATH in the current shell.
- Direct WindowsApps Python works: `%LOCALAPPDATA%\Microsoft\WindowsApps\python3.11.exe`.
- Project venv exists at `.venv`.
- App launch is verified, but no long-running server is currently active.
- llama.cpp tools are not on PATH.
- Export planning works without llama.cpp tools, but actual conversion/quantization remains blocked
  until llama.cpp tooling is installed.
- A GGUF path can now be configured from the Status tab, but no GGUF model file has been selected
  and verified in this workspace.
- `llama_cpp` Python package is not installed in `.venv`.
- Ollama is not on PATH.
- Trackio is optional; local JSONL tracing works without the `trackio` package, but remote Trackio/HF
  sync still needs package availability and credentials.
- Hugging Face dataset preview is optional and requires the external `datasets` package; the app
  reports a clear status when it is not installed.
- Full PRD implementation is not complete. There are still unchecked tasks in `docs/TASKS.md`.
- Current unchecked task count is still high because many PRD/ext PRD items need real local setup,
  credentials, hardware, or product decisions.

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
