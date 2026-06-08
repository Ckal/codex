# Full Task Checklist

This is the shared task list for you and Codex. It covers the hackathon MVP, the main PRD, and
the extension PRD. A task is complete only when the matching acceptance criteria are met and
`docs/IMPLEMENTATION_STATUS.md` is updated.

## Legend

- `[x]` done and documented
- `[~]` partially implemented or placeholder exists
- `[ ]` not started
- `[blocked]` blocked by missing local setup, credentials, hardware, or external decision

## Phase 0 - Project Memory And Setup

- [x] Add root `README.md`.
- [x] Add root `AGENTS.md`.
- [x] Add `.gitignore`.
- [x] Add `requirements.txt`.
- [x] Add `docs/` folder.
- [x] Add docs index.
- [x] Add full task checklist.
- [x] Add implementation status doc.
- [x] Add usage guide.
- [x] Add architecture guide.
- [x] Add extension guide.
- [x] Add acceptance criteria.
- [x] Add roadmap.
- [x] Add PRD implementation matrix.
- [x] Add test folder.
- [x] Add user-story test folder.
- [x] Add dev requirements.
- [x] Add Python quality config.
- [x] Add test runner script.
- [x] Add quality runner script.
- [x] Add CI workflow.
- [x] Add coverage gate.
- [x] Add performance test script.
- [x] Install Python 3.11+.
- [x] Verify `python --version`.
- [x] Create `.venv`.
- [x] Install dependencies.
- [ ] Run `python app.py`.
- [x] Capture screenshot or note local URL.

## Phase 1 - Hackathon Definition

- [x] Choose track: Backyard AI or Thousand Token Wood.
- [x] Write one-sentence project story.
- [x] Define target user.
- [x] Define measurable user benefit.
- [x] Decide final model family and model IDs.
- [x] Confirm every model is <= 32B parameters.
- [x] Decide local-first badge target.
- [x] Decide llama.cpp badge target.
- [x] Decide open trace badge target.
- [x] Decide field notes/report badge target.
- [x] Write final demo flow.
- [x] Write demo video script.
- [x] Write social post draft.
- [ ] Add final submission checklist with exact URLs.

## Phase 2 - MVP Gradio App

- [x] Add `app.py`.
- [x] Add Gradio `Blocks` shell.
- [x] Add model config loader.
- [x] Add model metadata display.
- [x] Add Chat tab.
- [x] Add Vision tab.
- [x] Add Dataset tab placeholder.
- [x] Add Train tab placeholder.
- [x] Add Export tab placeholder.
- [x] Add Field Notes tab.
- [x] Add placeholder text service.
- [x] Add placeholder vision service.
- [x] Add Traces tab placeholder.
- [x] Add Agent tab placeholder.
- [x] Add Status tab placeholder.
- [x] Add PowerShell structure verification script.
- [x] Run structure verification script.
- [x] Run app locally.
- [x] Fix local launch errors found so far.
- [x] Add screenshot capture path to docs or README.
- [x] Add first demo GIF/video plan.

## Phase 3 - Config-Driven Model Registry

- [x] Add `config/models.yaml`.
- [x] Add text model entry for MiniCPM5-1B.
- [x] Add vision model entry for MiniCPM-V-4.6.
- [x] Add omnimodal model entry for MiniCPM-o-4.5.
- [x] Add typed `ModelInfo`.
- [x] Add `load_model_catalog()`.
- [x] Add `model_choices()`.
- [x] Add `model_summary()`.
- [x] Add MiniCPM5-1B-Thinking config.
- [x] Add MiniCPM4.1-8B config.
- [x] Add MiniCPM-V-4.6-Thinking config.
- [x] Add GGUF metadata in config.
- [x] Add backend capability metadata.
- [x] Add lightweight catalog validation helper.
- [x] Show warnings for models over 32B parameters.

## Phase 4 - Core Architecture

- [x] Add `core/events.py`.
- [x] Add `EventType`.
- [x] Add `Event`.
- [x] Add `EventBus`.
- [x] Add `core/registry.py`.
- [x] Add generic `Registry`.
- [x] Add global app state.
- [x] Register model services in a service registry.
- [x] Emit inference events from UI.
- [x] Emit field note events.
- [x] Add lightweight logging.
- [x] Add unit tests for config and registry.

## Phase 5 - Testing And Quality

- [x] Add `tests/unit/`.
- [x] Add `tests/user_stories/`.
- [x] Add model catalog unit tests.
- [x] Add field notes unit tests.
- [x] Add new-user user-story test.
- [x] Add `requirements-dev.txt`.
- [x] Add `pyproject.toml`.
- [x] Add `scripts/run_tests.ps1`.
- [x] Add `scripts/run_quality.ps1`.
- [x] Run unit and user-story tests.
- [x] Install dev quality tools.
- [x] Run `ruff`.
- [x] Run `mypy`.
- [x] Run `pylint`.
- [x] Run `bandit`.
- [x] Run `pip-audit`.
- [x] Add rule: failing bug/check requires a new or updated test.
- [x] Add coverage report.
- [x] Add lightweight performance tests.
- [x] Add CI pipeline.
- [x] Add Playwright or equivalent browser e2e test after Gradio runs.
- [ ] Add tests for each real backend as it is implemented.
- [x] Add tests for backend service selection.
- [x] Add tests for Ollama unavailable path.
- [x] Add tests for llama.cpp unavailable path and command building.
- [x] Add tests for llama-cpp-python unavailable path.
- [x] Add tests for OpenAI-compatible/LM Studio unavailable and request paths.

## Phase 6 - Local Inference Backends

- [x] Choose first real backend.
- [x] Add backend selector in UI.
- [x] Add model status panel.
- [x] Add explicit model load button.
- [x] Ensure no model weights download on startup.

### Ollama Backend

- [ ] Confirm Ollama is installed.
- [x] Add `models/ollama_service.py`.
- [x] Add local model list.
- [x] Add pull model command with explicit user action.
- [x] Add text chat through Ollama.
- [x] Add vision chat through Ollama when supported.
- [x] Document Ollama setup.

### llama.cpp Backend

- [ ] Confirm llama.cpp tools are installed.
- [x] Add `models/llama_cpp_service.py`.
- [x] Add `models/llama_cpp_python_service.py`.
- [x] Add GGUF file picker.
- [x] Add `llama-server` launch command builder.
- [x] Add health check.
- [x] Add text generation through server.
- [x] Add vision `mmproj` support metadata.
- [x] Document llama.cpp setup.

### llama-cpp-python Backend

- [x] Add optional Python binding service.
- [x] Add backend selector support.
- [blocked] Install `llama-cpp-python` locally.
- [x] Configure local GGUF path.
- [blocked] Verify real text generation through Python binding.
- [x] Decide whether to keep Python binding as fallback or primary local path.

### Transformers Backend

- [x] Add `models/transformers_text.py`.
- [x] Add `AutoModelForCausalLM` loading for text models.
- [x] Add tokenizer loading.
- [x] Add explicit trust-remote-code handling.
- [x] Add device/dtype settings.
- [x] Add streaming generation.
- [x] Document hardware expectations.

### OpenAI-Compatible / LM Studio Backend

- [x] Add `models/openai_compatible_service.py`.
- [x] Add backend selector support.
- [x] Add local base URL and served-model-name config.
- [x] Add Status tab setup and reachability check.
- [x] Add text chat through OpenAI-compatible `/v1/chat/completions`.
- [x] Document LM Studio setup.
- [x] Verify real text generation through LM Studio.

### MiniCPM Vision Backend

- [x] Add `models/minicpm_vision.py`.
- [x] Use `AutoModelForImageTextToText`.
- [x] Use `AutoProcessor`.
- [x] Add image prompt formatting.
- [x] Add thinking-mode toggle mapping.
- [x] Add video support plan.

### SGLang Backend

- [x] Add `models/sglang_runner.py`.
- [x] Add server start/stop.
- [x] Add MiniCPM5 tool parser config.
- [x] Add health check.
- [x] Add chat endpoint client.

## Phase 7 - UI Tabs From Main PRD

- [x] Chat tab placeholder.
- [x] Vision tab placeholder.
- [x] Dataset tab placeholder.
- [x] Train tab placeholder.
- [x] Export tab placeholder.
- [x] Field Notes tab minimal save.
- [x] Add Traces tab with local event preview.
- [x] Add Agent tab placeholder.
- [x] Add model/backend status tab or panel.
- [x] Add settings panel.
- [x] Add tab-level error messages.
- [x] Add loading/progress states.
- [x] Add compact responsive layout review.

## Phase 8 - Dataset Layer

- [x] Add `datasets/` package.
- [x] Add local CSV loader.
- [x] Add local JSONL loader.
- [x] Add Hugging Face dataset loader.
- [x] Add dataset schema preview.
- [x] Add split selector.
- [x] Add row count and sample preview.
- [x] Add dataset statistics tool.
- [x] Emit `DATASET_LOADED` event.
- [x] Document dataset formats.

## Phase 9 - Field Notes And Correction Loop

- [x] Save field notes to CSV.
- [x] Move field note logic out of UI into `datasets/field_notes.py`.
- [x] Add `FieldNote` dataclass.
- [x] Add SQLite-backed store.
- [x] Add JSONL export.
- [x] Add local HF Dataset export.
- [x] Add corrected-only filter.
- [x] Add tags filter.
- [x] Add image path support.
- [x] Add video path support.
- [x] Add use-for-training flag.
- [x] Add docs for correction loop.

## Phase 10 - Training Pipeline

- [x] Add training config placeholder.
- [x] Add training UI placeholder.
- [x] Add `training/` package.
- [x] Add LoRA text trainer.
- [x] Add LoRA config parser.
- [ ] Add PEFT/TRL dependencies when ready.
- [x] Add training dry-run validation.
- [x] Add local checkpoint output.
- [x] Add Trackio integration.
- [x] Add evaluation after training.
- [x] Add LoRA vs base comparison.
- [x] Add vision fine-tuning plan using SWIFT or LLaMA-Factory.
- [x] Document training hardware requirements.

## Phase 11 - Evaluation

- [x] Add `training/evaluation.py`.
- [x] Add simple prompt test set.
- [x] Add exact-match metric.
- [x] Add qualitative eval table.
- [x] Add perplexity metric where appropriate.
- [x] Add base vs tuned comparison.
- [x] Log eval results.
- [x] Document evaluation method.

## Phase 12 - Export And Quantization

- [x] Add export UI placeholder.
- [x] Add `training/export.py`.
- [x] Add official GGUF download path.
- [x] Add local HF-to-GGUF conversion path.
- [x] Add quantization selector.
- [x] Add llama.cpp tool detection.
- [x] Add exported file listing.
- [x] Add download link in UI.
- [x] Document GGUF export.

## Phase 13 - Trackio Tracing

- [x] Add `tracking/` package.
- [x] Add Trackio config.
- [x] Add `trackio.init()`.
- [x] Add `trackio.log()`.
- [x] Add `trackio.finish()`.
- [x] Log inference events locally.
- [x] Log dataset events locally.
- [x] Log training metrics.
- [x] Add Traces tab.
- [x] Add HF Space sync docs.

## Phase 13 - MCP Layer

- [x] Decide MCP path: Gradio native, `gradio.Server` 
- [x] Add MCP tools module.
- [x] Add dataset stats tool.
- [x] Add HF search tool.
- [x] Add safe calculator tool.
- [x] Add model inference tool.
- [x] Expose tools through selected MCP path.
- [x] Document MCP endpoint.
- [x] Verify endpoint locally.

## Phase 14 - Agent Mode

- [x] Add `agent/` package.
- [x] Add agent system prompt.
- [x] Add research-plan-implement loop placeholder.
- [x] Add tool registry integration.
- [x] Add session trace logging.
- [x] Add Agent tab.
- [x] Add trace export to JSONL.
- [x] Add local HF Dataset export for traces.
- [x] Document limitations.

## Phase 15 - Hugging Face Space Deployment

- [ ] Install/verify `huggingface_hub`.
- [ ] Login with `huggingface-cli`.
- [ ] Create Space.
- [x] Add Space README metadata if needed.
- [ ] Add Space remote.
- [ ] Push to Space.
- [ ] Verify Space builds.
- [ ] Add Space URL to README.
- [x] Document hardware choice.
- [x] Document model download behavior.

## Phase 16 - GitHub

- [x] Create GitHub repo.
- [x] Add GitHub remote.
- [x] Commit initial project.
- [x] Push to GitHub.
- [x] Add GitHub URL to README.
- [ ] Add issue checklist or project board if desired.

## Phase 17 - Hackathon Submission Package

- [x] Finalize app name.
- [x] Finalize track.
- [ ] Verify Gradio app polish.
- [x] Verify model-size compliance.
- [ ] Verify Space URL.
- [x] Verify GitHub URL.
- [ ] Record demo video.
- [ ] Publish social post.
- [ ] Add field notes/report link.
- [ ] Submit before June 15, 2026.

## Extension PRD Backlog

### vLLM Serving Tab

- [x] Add vLLM runner.
- [x] Add vLLM start/stop UI.
- [x] Add OpenAI-compatible client.
- [x] Add metrics parsing.
- [x] Add Trackio benchmark logging.

### Ollama Quick-Start

- [x] Add Ollama pull/list UI.
- [x] Add Ollama chat service.
- [x] Add Ollama vision service.
- [x] Add setup docs.

### Llama.cpp Champion Path

- [x] Add llama.cpp backend selection.
- [x] Add llama.cpp service.
- [x] Add llama-cpp-python service.
- [x] Add llama.cpp status check.
- [ ] Install llama.cpp locally.
- [ ] Download/pick GGUF model.
- [ ] Verify real text generation.
- [ ] Verify MiniCPM-V mmproj flow.

### Reward Model Eval

- [x] Add reward evaluator.
- [x] Add best-of-N generation.
- [x] Add DPO pair generation.
- [x] Add LoRA vs base reward report.

### Synthetic Data Generation

- [x] Add synthetic generator.
- [x] Add JSON validation.
- [x] Add quality filters.
- [x] Add augmentation flow.
- [x] Add dataset save/export.

### Paper-To-Code Agent

- [x] Add paper input UI.
- [x] Add research phase.
- [x] Add plan phase.
- [x] Add implementation trace.
- [x] Add safety gates.

### HF Spaces Deploy Tool

- [x] Add deployment helper script.
- [x] Add Space creation docs.
- [x] Add remote validation.
- [x] Add build status checks.

### VINDEX Integration

- [x] Define integration boundary.
- [x] Add tool stub.
- [x] Add verification report.
- [x] Document dependency.

### OCR Pipeline Hook

- [x] Add OCR loader.
- [x] Add confidence threshold.
- [x] Add uncertain prediction import.
- [x] Add correction UI.
- [x] Add corrected export.

### MiniCPM Desk-Pet

- [ ] Add persona data schema.
- [ ] Add persona training plan.
- [ ] Add Desk-Pet export plan.
- [ ] Add docs.

### MiniCPM-o Audio Tab

- [ ] Add audio tab.
- [ ] Add microphone input.
- [ ] Add omnimodal service.
- [ ] Add TTS plan.
- [ ] Add streaming plan.

### Cross-Extension Wiring

- [x] Document OCR -> Field Notes -> Training.
- [x] Document Synthetic Gen -> Reward Eval -> DPO.
- [x] Document Agent -> Desk-Pet Persona.
- [x] Document HF Spaces -> Trackio.

## Ongoing Maintenance

- [x] Update docs after every implemented feature.
- [x] Keep `IMPLEMENTATION_STATUS.md` current.
- [x] Keep unchecked tasks visible.
- [x] Keep secrets and model weights out of git.
- [x] Re-run local app after code changes.
