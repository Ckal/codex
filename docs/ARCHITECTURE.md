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
  holds local tool functions, VINDEX call planning, and Gradio-native MCP bridge metadata

config/*
  holds model and training settings

training/*
  holds non-executing training, LoRA request, evaluation, and export planning helpers

tracking/*
  holds local JSONL tracing and optional Trackio integration

deployment/*
  holds Hugging Face Space deployment planning and validation helpers

plant/*
  holds the first reference domain app built from the template
  can run standalone with python -m plant.app --no-model
  keeps heavy model dependencies optional

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

### `plant/app.py`

Standalone Plant Discovery reference app built from the template.

- `build_app(no_model=True)` creates a Gradio app without loading model weights.
- Loads `plant/models.yaml`.
- Builds a local species index.
- Reuses `datasets.field_notes.FieldNoteStore` for corrections.
- Uses `DemoPlantVisionService` for screenshots/tests or `PlantVisionService` for optional
  MiniCPM-V inference.

### `plant/plant_service.py`

Domain service and schema for Plant Discovery.

- `PlantID` is the structured output schema.
- `DemoPlantVisionService` provides deterministic no-model results.
- `PlantVisionService` lazy-loads optional MiniCPM-V dependencies only during identification.
- `extract_json_object()` and `parse_plant_response()` make model JSON output testable.

### `plant/plant_loader.py`

Domain data and export helpers for Plant Discovery.

- `PlantRecord` normalizes plant examples into training rows.
- `LocalFolderLoader` maps species folders to image metadata.
- `SpeciesIndexBuilder` builds a no-network species index with demo fallback.
- `FieldNotesPlantExporter` exports corrected field notes to plant training JSONL.

### `plant/plant_tab.py`

Focused Gradio UI for Plant Discovery.

- Identify tab uploads images and renders a safe escaped result card.
- Field Guide tab searches the species index.
- Corrections tab saves and exports training-ready corrections.
- Stats tab summarizes species and correction counts.
- Training is represented as a non-executing plan, not a subprocess.

### `plant/plant_tools.py`

Optional local/MCP tools for Plant Discovery.

- Pure functions can be tested without an MCP server.
- `build_mcp_server()` imports `mcp` only when explicitly requested.
- Tools expose identify, species search, correction save/export, stats, and training plan.

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
- `openai_compatible_service.py`
- `sglang_runner.py`
- `minicpm_vision.py`
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

- `LocalBackendConfig` stores llama.cpp server URL, OpenAI-compatible base URL, optional served
  model name, GGUF path, mmproj path, context length, and GPU layers.
- `save_local_backend_config()` writes local-only settings without touching tracked model config.
- `build_llama_server_command()` returns the explicit command the user can run.
- `local_backend_summary()` reports file status and confirms no startup downloads or automatic model loads.

### `models/openai_compatible_service.py`

Local OpenAI-compatible chat client for LM Studio, vLLM-style servers, or similar local endpoints.

- Checks `/v1/models` for reachability.
- Sends text chat requests to `/v1/chat/completions`.
- Supports an optional served-model-name override for tools such as LM Studio.
- Returns visible unavailable/request-failed messages instead of crashing the Gradio callback.
- Does not call cloud APIs or download model weights.

### `models/llama_cpp_python_service.py`

Optional direct Python binding backend for GGUF inference.

- Checks whether `llama_cpp` is importable.
- Requires an explicit local GGUF path.
- Does not download model files.
- Provides text chat through `Llama.create_chat_completion()`.
- Vision support remains routed through llama-server until mmproj/image serialization is wired.

### `models/minicpm_vision.py`

Optional MiniCPM vision backend.

- Checks whether the `transformers` package is available.
- Lazy-loads `AutoProcessor` and `AutoModelForImageTextToText` only when selected.
- Formats image/text messages for image-text-to-text generation.
- Maps thinking mode into the prompt template.
- Provides a video support plan for future local frame sampling.

### `models/sglang_runner.py`

SGLang local server planner and OpenAI-compatible chat client.

- Builds an explicit `python -m sglang.launch_server` command.
- Includes MiniCPM tool parser configuration.
- Checks `/health`, sends chat requests to `/v1/chat/completions`, and can request `/shutdown`.
- Does not install SGLang, start a process, download model weights, or load a model on app startup.

### `models/vllm_runner.py`

vLLM local server planner and OpenAI-compatible chat client.

- Builds explicit `vllm serve <model>` command plans.
- Checks `/health`, parses Prometheus-style `/metrics`, and sends chat requests to
  `/v1/chat/completions`.
- Logs parsed benchmark metrics through `TrackingClient`.
- Does not install vLLM, start a process, download model weights, or load a model on app startup.

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
- `create_text_service()` chooses placeholder, llama.cpp, llama-cpp-python, Ollama,
  OpenAI-compatible, SGLang, or Transformers text service.
- `create_vision_service()` chooses placeholder, llama.cpp, llama-cpp-python, Ollama, or
  Transformers MiniCPM vision service.
- `backend_statuses()` reports current backend availability.
- llama.cpp, llama-cpp-python, and OpenAI-compatible services read ignored local backend settings
  when selected.

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
- Builds a non-executing LoRA trainer request with dependency status.
- Shows SWIFT/LLaMA-Factory vision fine-tuning plan.
- Shows checkpoint output path, validation status, and hardware notes.
- Runs local base-vs-tuned evaluation from newline-separated response text.
- Shows exact-match summary and a qualitative eval table.
- Logs tuned evaluation reports to `data/eval_results.jsonl`.

Future behavior:

- Start LoRA training.
- Show loss and metrics.
- Write Trackio traces.

### `ui/vllm_tab.py`

vLLM local serving planner.

- Builds explicit `vllm serve` command plans.
- Checks local vLLM `/health`.
- Fetches and parses `/metrics`.
- Logs vLLM benchmark metrics through local JSONL/Trackio fallback tracking.
- Does not install vLLM, start a process, download models, or load weights on startup.

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
- Imports uncertain OCR predictions for human correction.
- Exports corrected OCR rows to JSONL.
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
- Provides LM Studio/OpenAI-compatible base URL, optional model-name storage, and reachability check.
- Provides SGLang command planning, health check, and shutdown request controls.

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

### `datasets/ocr.py`

Local OCR correction helpers.

- `OCRPrediction` stores source path, predicted text, confidence, and optional page.
- `load_ocr_predictions()` loads local `.csv`, `.jsonl`, and `.ndjson` prediction files.
- `uncertain_predictions()` filters rows at or below a confidence threshold or with empty text.
- `import_uncertain_predictions()` creates Field Notes correction tasks for uncertain rows.
- `export_corrected_ocr_notes()` writes corrected OCR examples to JSONL for evaluation or training.
- `ocr_import_summary()` previews uncertain rows for the Field Notes tab.

### `mcp_tools/tools.py`

Local MCP-style tools.

- `dataset_stats_tool()` returns local dataset statistics.
- `hf_dataset_preview_tool()` previews Hugging Face datasets when optional dependencies exist.
- `safe_calculator_tool()` evaluates numeric arithmetic only.
- `model_inference_tool()` routes text prompts through the selected model service.
- `tool_registry()` returns the local tool map for a future MCP endpoint.

### `mcp_tools/vindex_tool.py`

Non-executing VINDEX integration boundary.

- Defines the eight VINDEX PRD methods and their local FastAPI paths.
- `build_vindex_call_plan()` validates method names and builds endpoint/payload plans.
- Caps `star_spread.n_neighbors` at 5 and `calibrated_edit.causal_window` at 3 based on the PRD
  safety notes.
- `vindex_dependency_report()` checks whether the optional `vindex` package or local health
  endpoint is available.
- `vindex_verification_report()` combines dependency status with a safe call plan and keeps
  execution disabled until the local VINDEX install is verified.

### `mcp_tools/bridge.py`

Gradio-native MCP bridge metadata and local invocation helper.

- `MCP_PATH` documents `/gradio_api/mcp/sse`.
- `mcp_manifest()` returns the selected mode, path, and tool definitions.
- `invoke_mcp_tool()` verifies local tool invocation by name.

### `agent/runner.py`

Deterministic local agent trace runner.

- `AGENT_SYSTEM_PROMPT` defines the agent behavior contract.
- `run_agent_loop()` produces research, plan, implement, and verify trace steps.
- `run_paper_to_code_loop()` produces paper-to-code research, plan, implement, and verify trace steps.
- `default_safety_gates()` lists the non-autonomous safety requirements.
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

### `training/lora_trainer.py`

Non-executing LoRA trainer request builder.

- `lora_dependency_report()` reports PEFT, TRL, Transformers, and Torch availability.
- `build_lora_training_request()` combines the training plan with dependency status and a command
  preview.
- `vision_finetuning_plan()` documents SWIFT/LLaMA-Factory as the future MiniCPM-V fine-tuning path.
- Keeps `execute_training` false until dependencies, hardware, and dataset schema are approved.

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
