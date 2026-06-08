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
- OCR correction loop exists locally for CSV/JSONL prediction imports into Field Notes.
- VINDEX integration boundary exists locally as non-executing MCP-style planning tools.
- Local non-autonomous agent mode exists with trace export.
- Real local model inference is partially implemented through llama.cpp, llama-cpp-python, Ollama,
  OpenAI-compatible/LM Studio, SGLang, Transformers text, and MiniCPM vision services. The Status
  tab includes llama.cpp setup, LM Studio/OpenAI-compatible setup, SGLang command/check/stop setup,
  and Ollama local model listing plus explicit pull-command planning. LM Studio text generation is
  live-verified with `llama-3.2-1b-instruct`; the other real backends still need local verification.
- LoRA training execution, served MCP endpoint, deployment, and most extensions are not implemented.
- Placeholder services remain intentionally visible so the app never pretends to be real inference.

## Main PRD

| PRD Area | Status | Evidence / Next Step |
| --- | --- | --- |
| Purpose and design philosophy | Documented | `README.md`, `docs/ROADMAP.md` |
| Template architecture | Partial | Config-driven model catalog exists |
| System architecture | Partial | `app.py`, `core/`, `models/`, `ui/`, `datasets/`, local app state/events |
| Model registry | Partial | `config/models.yaml`, `models/model_catalog.py`; includes GGUF and backend capability metadata |
| Five inference modes | Partial | llama.cpp, llama-cpp-python, Ollama, OpenAI-compatible/LM Studio, SGLang, vLLM, Transformers text, and MiniCPM vision services exist; local verification remains incomplete for most real backends |
| Trackio | Partial | Local traces, optional Trackio wrapper, and HF Space sync docs exist; credentials/package setup still missing |
| MCP layer | Partial | Local tool functions, Gradio-native MCP path metadata, `mcp_server=True` launch flag, and local invocation tests exist; full external client verification still missing |
| Training pipeline | Partial | `training/` package supports dry-run planning, non-executing LoRA request planning, export planning, exact-match/perplexity evaluation, and local logging; real PEFT/TRL execution missing |
| Export and quantization | Partial | `training/export.py` and Export tab plan downloads/conversion/quantization and expose existing exported files for download; execution still missing |
| Agent mode | Partial | Local deterministic task and paper-to-code trace loops exist with safety gates; autonomous execution and remote uploads missing |
| UI tabs | Partial | Tabs exist; Chat/Vision/Dataset/Field Notes/Status have behavior; Status includes SGLang setup; tab actions have Gradio progress indicators; Chat/Vision/Dataset have tab-level status/error messages; compact responsive CSS exists; several tabs are still placeholders |
| Field notes | Partial | CSV save, SQLite store, corrected/tag/training filters, media paths, OCR uncertain import, JSONL export, and local HF Dataset export exist; remote HF upload missing |
| Directory structure | Partial | Foundation exists; many PRD packages missing |
| Configuration schema | Partial | Model/training config plus ignored local backend config exists; validation is lightweight |
| Dependencies | Partial | Runtime/dev deps exist for scaffold; full model/training deps not added |
| Hackathon demo flow | Partial | `docs/HACKATHON_SUBMISSION.md` drafts story, user, demo flow, script, social post, and URLs; real backend and Space URL still missing |
| Corrections from PRD v1 | Documented in PRD | Not all implemented |
| Roadmap and extension points | Documented | `docs/ROADMAP.md`, `docs/TASKS.md` |

## Extension PRD

| Extension | Status | Evidence / Next Step |
| --- | --- | --- |
| vLLM serving tab | Implemented locally, not locally verified | `models/vllm_runner.py` and vLLM tab provide command planning, health checks, metrics parsing, benchmark logging, and OpenAI-compatible chat client; needs installed/running vLLM for real serving |
| Ollama quick-start | Partial | Service, UI backend selector, local model listing, explicit pull-command planning, and setup docs exist; local Ollama install/real model verification missing |
| Reward model eval | Implemented locally | `training/reward_eval.py` provides deterministic reward scoring, best-of-N, DPO pairs, and LoRA-vs-base reward reports |
| Synthetic data generation | Implemented locally | `datasets/synthetic.py` provides deterministic generation, validation, filtering, augmentation, and JSONL export |
| Paper-to-code agent | Implemented locally | Agent tab and `agent/runner.py` support paper input, research/plan/implementation/verify trace, and safety gates without autonomous execution |
| HF Spaces deploy | Partial | README metadata, deployment helper, command plan, required-file validation, and remote/build status checks exist; HF auth/remote/push still missing |
| VINDEX integration | Implemented locally, execution disabled | `mcp_tools/vindex_tool.py` validates VINDEX methods, builds safe call plans, reports dependency/server status, and documents that actual edits require a verified local VINDEX install |
| OCR pipeline hook | Implemented locally | `datasets/ocr.py` and Field Notes tab support local OCR prediction loading, confidence thresholds, uncertain import, human correction, and corrected JSONL export |
| MiniCPM Desk-Pet | Not implemented | Needs persona schema/export |
| MiniCPM-o audio tab | Not implemented | Needs audio tab and omnimodal backend |
| Cross-extension wiring | Partial | OCR -> Field Notes -> Training, Synthetic Gen -> Reward Eval -> DPO, Agent -> Desk-Pet Persona, and HF Spaces -> Trackio are documented; remaining wiring depends on unimplemented runtime modules |

## Quality Coverage

Current verified gates:

- Structure check passes.
- 170 unit/user-story tests pass.
- Coverage report passes at 68%, above the current 60% configured threshold.
- 2 lightweight performance tests pass.
- Ruff passes.
- Mypy passes.
- Pylint passes at 10/10.
- Bandit reports no issues.
- Pip-audit reports no known vulnerabilities in `.venv`.
- LM Studio `/v1/models` and `/v1/chat/completions` are verified locally for
  `llama-3.2-1b-instruct`.
- CI workflow exists but has not run remotely.
- App launch has been verified locally, but the server is not currently left running.

## No Pretend-Done Rule

Any row marked `Partial`, `Placeholder`, or `Not implemented` must not be described as complete.
When a row is implemented, update this file, `docs/TASKS.md`, and `docs/IMPLEMENTATION_STATUS.md`.
