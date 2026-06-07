# PRD Implementation Matrix

This file maps the main PRD and extension PRD to current implementation status.

## Summary

The full PRD and extension PRD are not fully implemented yet.

Current state:

- Foundation, docs, test policy, quality gates, CI, and placeholder Gradio surfaces exist.
- Shared app state, service registries, local event logging, tab-level error events, and local
  trace preview exist.
- Local llama.cpp settings, GGUF/mmproj pickers, and command generation exist without startup downloads.
- GGUF export planning exists with tool detection and explicit non-executing command plans.
- Local JSONL tracing and optional Trackio wrapper exist.
- Dataset statistics and local MCP tool functions exist.
- Local non-autonomous agent mode exists with trace export.
- Real local model inference is partially implemented through llama.cpp, llama-cpp-python, Ollama,
  and Transformers text services. The Status tab includes llama.cpp setup plus Ollama local model
  listing and explicit pull-command planning, but none has been locally verified with a real model.
- LoRA training execution, served MCP endpoint, deployment, and most extensions are not implemented.
- Placeholder services remain intentionally visible so the app never pretends to be real inference.

## Main PRD

| PRD Area | Status | Evidence / Next Step |
| --- | --- | --- |
| Purpose and design philosophy | Documented | `README.md`, `docs/ROADMAP.md` |
| Template architecture | Partial | Config-driven model catalog exists |
| System architecture | Partial | `app.py`, `core/`, `models/`, `ui/`, `datasets/`, local app state/events |
| Model registry | Partial | `config/models.yaml`, `models/model_catalog.py`; includes GGUF and backend capability metadata |
| Five inference modes | Partial | llama.cpp, llama-cpp-python, Ollama, and Transformers text services exist; SGLang and vLLM missing |
| Trackio | Partial | Local traces, optional Trackio wrapper, and HF Space sync docs exist; credentials/package setup still missing |
| MCP layer | Partial | Local tool functions exist; served MCP endpoint still missing |
| Training pipeline | Partial | `training/` package supports dry-run planning, export planning, and local evaluation; LoRA trainer missing |
| Export and quantization | Partial | `training/export.py` and Export tab plan downloads/conversion/quantization; execution/download links missing |
| Agent mode | Partial | Local deterministic agent trace loop exists; autonomous execution and remote uploads missing |
| UI tabs | Partial | Tabs exist; Chat/Vision/Dataset/Field Notes/Status have behavior; tab actions have Gradio progress indicators; Chat/Vision/Dataset have tab-level status/error messages; compact responsive CSS exists; several tabs are still placeholders |
| Field notes | Partial | CSV save, SQLite store, corrected/tag/training filters, media paths, JSONL export, and local HF Dataset export exist; remote HF upload missing |
| Directory structure | Partial | Foundation exists; many PRD packages missing |
| Configuration schema | Partial | Model/training config plus ignored local backend config exists; validation is lightweight |
| Dependencies | Partial | Runtime/dev deps exist for scaffold; full model/training deps not added |
| Hackathon demo flow | Partial | `docs/HACKATHON_SUBMISSION.md` drafts story, user, demo flow, script, social post, and URLs; real backend and Space URL still missing |
| Corrections from PRD v1 | Documented in PRD | Not all implemented |
| Roadmap and extension points | Documented | `docs/ROADMAP.md`, `docs/TASKS.md` |

## Extension PRD

| Extension | Status | Evidence / Next Step |
| --- | --- | --- |
| vLLM serving tab | Not implemented | Needs `models/vllm_runner.py` |
| Ollama quick-start | Partial | Service, UI backend selector, local model listing, explicit pull-command planning, and setup docs exist; local Ollama install/real model verification missing |
| Reward model eval | Not implemented | Needs `training/reward_eval.py` |
| Synthetic data generation | Not implemented | Needs `datasets/synthetic.py` |
| Paper-to-code agent | Not implemented | Needs real agent loop and safety gates |
| HF Spaces deploy | Partial | README metadata, deployment helper, command plan, required-file validation, and remote/build status checks exist; HF auth/remote/push still missing |
| VINDEX integration | Not implemented | Needs integration boundary and dependency |
| OCR pipeline hook | Not implemented | Needs OCR loader and correction UI |
| MiniCPM Desk-Pet | Not implemented | Needs persona schema/export |
| MiniCPM-o audio tab | Not implemented | Needs audio tab and omnimodal backend |
| Cross-extension wiring | Not implemented | Needs implemented modules first |

## Quality Coverage

Current verified gates:

- Structure check passes.
- 96 unit/user-story tests pass.
- Coverage report passes at 65%, above the current 60% configured threshold.
- 2 lightweight performance tests pass.
- Ruff passes.
- Mypy passes.
- Pylint passes at 10/10.
- Bandit reports no issues.
- Pip-audit reports no known vulnerabilities in `.venv`.
- CI workflow exists but has not run remotely.
- App launch has been verified locally, but the server is not currently left running.

## No Pretend-Done Rule

Any row marked `Partial`, `Placeholder`, or `Not implemented` must not be described as complete.
When a row is implemented, update this file, `docs/TASKS.md`, and `docs/IMPLEMENTATION_STATUS.md`.
