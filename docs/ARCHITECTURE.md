# Architecture

The project is intentionally small at first. The PRD describes a large workbench; this repo starts
with the smallest version that can grow into it.

## High-Level Flow

```text
app.py
  loads config/models.yaml
  configures lightweight logging
  builds Gradio tabs
  passes model catalog to UI modules

ui/*
  defines each Gradio tab
  calls service classes
  emits local app events for inference, datasets, and field notes
  uses shared progress settings for callback loading indicators

agent/*
  holds deterministic local agent planning and trace export helpers

models/*
  holds model catalog, local backend config, and inference services

datasets/*
  stores dataset, synthetic data, and correction-loop helpers

mcp_tools/*
  holds local tool functions and Gradio-native MCP bridge metadata

config/*
  holds model and training settings

training/*
  holds non-executing training, evaluation, and export planning helpers

tracking/*
  holds local JSONL tracing and optional Trackio integration

deployment/*
  holds Hugging Face Space deployment planning and validation helpers

core/*
  shared app state, event, logging, and registry helpers
```

## Files And Classes

### `app.py`

Builds and launches the Gradio app.

- `build_app()` creates the Gradio `Blocks` app.
- Loads the model catalog from `config/models.yaml`.
- Registers the current UI tabs.
- `APP_CSS` defines compact responsive layout rules for app width, mobile padding, scrollable tabs,
  and button touch targets.

### `models/model_catalog.py`

Reads model configuration and turns it into typed Python objects.

- `ModelInfo` describes one configured model.
- `load_model_catalog(path)` reads YAML and returns all configured models.
- `model_choices(catalog, model_type)` filters models for a UI dropdown.
- `model_summary(model)` returns display metadata for the Gradio JSON panel.
- `backend_capabilities` maps each model to supported local backend capabilities.

### `models/placeholder_service.py`

Deterministic placeholder model service used before real inference is wired.

- `PlaceholderModelService.chat()` returns a deterministic text response.
- `PlaceholderModelService.vision_chat()` returns a deterministic image/prompt response.

This file should be replaced or complemented by real services such as:

- `ollama_service.py`
- `llama_cpp_service.py`
- `transformers_text.py`
- `sglang_service.py`

### `models/base.py`

Defines service contracts and backend status records.

- `BackendStatus` describes whether a backend is available.
- `TextModelService` is the text chat protocol.
- `VisionModelService` is the vision chat protocol.

### `models/ollama_service.py`

Ollama-backed local inference client.

- Checks whether `ollama` is installed and reachable.
- Sends text and vision chat requests to `http://127.0.0.1:11434/api/chat`.
- Lists locally available Ollama models through `/api/tags`.
- Builds explicit `ollama pull <model>` commands for the Status tab.
- Does not pull or download models automatically.

### `models/llama_cpp_service.py`

llama.cpp HTTP client for local GGUF inference.

- Checks whether `llama-server` is installed and reachable.
- Builds explicit `llama-server -m <model.gguf>` commands.
- Supports `--mmproj <mmproj.gguf>` command metadata for multimodal models.
- Sends text chat requests to `/v1/chat/completions`.
- Does not download GGUF files or start background servers automatically.

### `models/local_backend_config.py`

User-local backend settings stored under ignored `data/local_backends.yaml`.

- `LocalBackendConfig` stores llama.cpp server URL, GGUF path, mmproj path, context length, and GPU layers.
- `save_local_backend_config()` writes local-only settings without touching tracked model config.
- `build_llama_server_command()` returns the explicit command the user can run.
- `local_backend_summary()` reports file status and confirms no startup downloads or automatic model loads.

### `models/llama_cpp_python_service.py`

Optional direct Python binding backend for GGUF inference.

- Checks whether `llama_cpp` is importable.
- Requires an explicit local GGUF path.
- Does not download model files.
- Provides text chat through `Llama.create_chat_completion()`.
- Vision support remains routed through llama-server until mmproj/image serialization is wired.

### `models/transformers_text.py`

Optional Transformers text backend.

- Checks whether the `transformers` package is installed.
- Lazy-loads `AutoTokenizer` and `AutoModelForCausalLM` only when the backend is selected.
- Reads `trust_remote_code`, device map, dtype, max token, and temperature settings from explicit config.
- Provides a simple token-list streaming helper for future Gradio streaming wiring.
- Does not download model weights on startup.

### `models/service_factory.py`

Creates the selected backend service for the UI.

- `TEXT_SERVICE_REGISTRY` registers available text backend factories.
- `VISION_SERVICE_REGISTRY` registers available vision backend factories.
- `create_text_service()` chooses placeholder, llama.cpp, llama-cpp-python, or Ollama text service.
- `create_vision_service()` chooses placeholder, llama.cpp, llama-cpp-python, or Ollama vision service.
- `backend_statuses()` reports current backend availability.
- llama.cpp and llama-cpp-python services read the ignored local GGUF settings when selected.

### `ui/chat_tab.py`

Builds the text chat tab.

- Shows text models from the catalog.
- Displays selected model metadata.
- Calls the selected backend service.
- Emits inference request and response events.

### `ui/vision_tab.py`

Builds the vision tab.

- Shows vision models from the catalog.
- Accepts an image and prompt.
- Calls the selected backend service.
- Emits inference request and response events.

### `ui/dataset_tab.py`

Local dataset preview surface.

- Previews local CSV, JSONL, and NDJSON files.
- Previews Hugging Face datasets when the optional external `datasets` package is installed.
- Shows source, row count, columns, and sample rows.
- Calculates basic local dataset statistics.
- Emits dataset loaded events.

Future behavior:

- Serve dataset tools through the selected MCP path.

### `ui/train_tab.py`

Training planning and local evaluation surface.

- Builds a LoRA dry-run training plan without launching training.
- Shows checkpoint output path, validation status, and hardware notes.
- Runs local base-vs-tuned evaluation from newline-separated response text.
- Shows exact-match summary and a qualitative eval table.
- Logs tuned evaluation reports to `data/eval_results.jsonl`.

Future behavior:

- Start LoRA training.
- Show loss and metrics.
- Write Trackio traces.

### `ui/export_tab.py`

GGUF export planning surface.

- Selects a configured model and quantization.
- Shows official GGUF download command plans when the model has GGUF metadata.
- Shows local HF-to-GGUF conversion and llama.cpp quantization command plans.
- Lists files already present under the selected export directory.
- Exposes existing exported files through a Gradio download output.
- Does not execute downloads, conversion, or quantization.

Future behavior:

- Execute downloads and conversions after explicit user action.

### `ui/notes_tab.py`

Field notes implementation.

- Saves prompt, model response, correction, and tags to `data/field_notes.csv`.
- Captures optional image path, video path, and a use-for-training flag.
- Exports corrected notes to JSONL.
- Exports local Hugging Face Dataset-style files under `data/hf_field_notes`.
- Emits field note saved events.

Future behavior:

- Push corrected notes to a remote Hugging Face Dataset after login.
- Feed notes into fine-tuning.

### `ui/traces_tab.py`

Local trace and tracking preview.

- Shows manual trace event previews.
- Shows recent local app events.
- Shows JSONL trace rows and tracking status.
- Exports local traces to `exports/traces.jsonl`.
- Calls Trackio only when the optional package is installed and enabled.

### `ui/agent_tab.py`

Local non-autonomous agent mode.

- Drafts a research-plan-implement-verify trace.
- Saves agent traces to `data/agent_traces.jsonl`.
- Exports trace JSONL and local HF Dataset-style trace files.
- Does not execute shell commands, commit, push, deploy, download models, or call external services.

### `ui/status_tab.py`

Shows configured models and backend metadata.

- Helps verify model-size compliance and backend status.
- Provides local llama.cpp settings, GGUF/mmproj file pickers, and command generation.

### `datasets/field_notes.py`

Field note data model and CSV store.

- `FieldNote` captures prompt, response, correction, tags, and timestamp.
- `FieldNote` also captures optional image/video paths and a training inclusion flag.
- `FieldNoteStore.save()` persists notes to `data/field_notes.csv`.
- `FieldNoteStore.list_notes()` filters by correction, tag, and training inclusion.
- `FieldNoteStore.export_jsonl()` writes training-ready JSONL.
- `FieldNoteStore.export_hf_dataset()` writes local HF Dataset-style files.
- `SQLiteFieldNoteStore` stores and lists notes in SQLite for larger correction loops.

### `datasets/loader.py`

Dataset preview and statistics helpers.

- `preview_local_dataset()` previews CSV, JSONL, and NDJSON files.
- `dataset_statistics()` reports row count, column count, names, and non-empty counts.
- `preview_huggingface_dataset()` optionally uses the external Hugging Face `datasets` package.

### `datasets/synthetic.py`

Deterministic local synthetic data helpers.

- `generate_synthetic_examples()` creates local prompt/response/correction examples.
- `validate_synthetic_example()` checks schema requirements.
- `quality_filter_examples()` removes incomplete or low-value examples.
- `augment_examples()` creates deterministic variants for workflow testing.
- `export_synthetic_jsonl()` writes JSONL without external services.

### `mcp_tools/tools.py`

Local MCP-style tools.

- `dataset_stats_tool()` returns local dataset statistics.
- `hf_dataset_preview_tool()` previews Hugging Face datasets when optional dependencies exist.
- `safe_calculator_tool()` evaluates numeric arithmetic only.
- `model_inference_tool()` routes text prompts through the selected model service.
- `tool_registry()` returns the local tool map for a future MCP endpoint.

### `mcp_tools/bridge.py`

Gradio-native MCP bridge metadata and local invocation helper.

- `MCP_PATH` documents `/gradio_api/mcp/sse`.
- `mcp_manifest()` returns the selected mode, path, and tool definitions.
- `invoke_mcp_tool()` verifies local tool invocation by name.

### `agent/runner.py`

Deterministic local agent trace runner.

- `AGENT_SYSTEM_PROMPT` defines the agent behavior contract.
- `run_agent_loop()` produces research, plan, implement, and verify trace steps.
- `save_agent_trace()` appends traces to JSONL.
- `export_agent_traces()` exports trace JSONL.
- `export_agent_traces_hf_dataset()` writes local HF Dataset-style trace files.
- The runner can call safe local tools, but it is not autonomous.

### `core/file_exports.py`

Shared export helper.

- `copy_text_file_or_empty()` copies a text artifact to an export path or creates an empty one.

### `training/export.py`

Non-executing GGUF export planning.

- `detect_llama_cpp_tools()` checks `llama-server`, `llama-cli`, and `llama-quantize`.
- `build_export_plan()` creates explicit download, conversion, and quantization command plans.
- `list_exported_files()` lists generated/local export files.
- `ExportPlan.as_dict()` marks that commands are not executed and no startup downloads occur.

### `training/evaluation.py`

Local deterministic evaluation helpers.

- `default_prompt_cases()` returns a small built-in prompt test set.
- `load_prompt_cases()` loads prompt/expected pairs from JSONL.
- `evaluate_responses()` computes exact-match rows and a qualitative table.
- `perplexity_from_losses()` computes perplexity from explicit negative log likelihood values.
- `compare_base_vs_tuned()` reports exact-match delta.
- `log_eval_report()` appends JSONL evaluation results.

### `training/reward_eval.py`

Deterministic local reward-style evaluation helpers.

- `RewardEvaluator.evaluate()` scores supplied responses with transparent lexical heuristics.
- `best_of_n()` selects the highest-scoring candidate without model calls.
- `create_dpo_pairs()` creates chosen/rejected pairs for DPO-style datasets.
- `eval_lora_vs_base()` compares base and LoRA response rewards.

### `training/planner.py`

Non-executing LoRA training planner.

- `load_training_config()` reads LoRA and training settings from `config/training.yaml`.
- `build_training_plan()` creates a dry-run plan with checkpoint output path.
- `validate_training_plan()` checks dataset existence and numeric training settings.
- `training_hardware_notes()` documents practical local hardware expectations.

### `tracking/trackio_client.py`

Tracking client with JSONL fallback.

- `load_tracking_config()` reads Trackio settings from `config/training.yaml`.
- `TrackingClient.init()` starts Trackio only when enabled and installed.
- `TrackingClient.log()` always writes local JSONL and optionally forwards to Trackio.
- `TrackingClient.finish()` closes optional Trackio state.
- `export_traces()` copies local traces to `exports/traces.jsonl`.
- `read_trace_rows()` returns recent local trace rows for the UI.

### `core/events.py`

Small event bus reserved for future cross-module events.

- `EventType` names app events.
- `UI_ERROR` records visible tab-level failures.
- `Event` carries event data.
- `EventBus` registers handlers and emits events.

### `core/app_state.py`

Shared local app state.

- `AppState.emit()` records events, logs them, and dispatches them through `EventBus`.
- `AppState.emit()` also writes trace events through `TrackingClient`.
- `AppState.recent_events()` returns local trace previews for the Traces tab.
- `emit_inference_response()` records shared response metadata.

### `core/tab_feedback.py`

Formats tab status text and emits `ui_error` events for visible tab-level failures.

### `ui/progress.py`

Defines the shared Gradio progress mode used by tab button callbacks.

### `core/app_logging.py`

Lightweight logging setup.

- `configure_app_logging()` configures compact process logging once.

### `core/registry.py`

Generic registry helper.

- `Registry.register(name, item)` stores a service.
- `Registry.get(name)` retrieves a service.
- `Registry.list()` lists registered services.

## Current Design Rule

The app must not download model weights on startup. Model loading should happen only after the
user chooses a backend/model and clicks an explicit action.
