# How To Use The Project

## 1. Install Python

Install Python 3.11 or newer. On Windows, make sure "Add Python to PATH" is enabled.

Verify:

```powershell
python --version
```

Before Python is available, you can still verify the repository structure:

```powershell
.\scripts\verify_structure.ps1
```

In this workspace, the direct WindowsApps interpreter worked even though `python` was not on PATH:

```powershell
& "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe" --version
```

## 2. Create A Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If `python` is not on PATH, use the WindowsApps interpreter:

```powershell
& "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe" -m venv .venv
```

## 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

For tests and quality checks:

```powershell
python -m pip install -r requirements-dev.txt
```

## 4. Run The App

```powershell
python app.py
```

## 5. Run Tests

```powershell
.\scripts\run_tests.ps1
```

## 6. Run Quality Checks

```powershell
.\scripts\run_quality.ps1
```

Open the local Gradio URL printed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## 7. Current App Behavior

The current app uses placeholder model responses. That is intentional. It lets you verify the UI,
docs, and workflow before downloading large model files.

Available tabs. User-triggered tab actions show Gradio progress while callbacks run, and the app
uses compact responsive styling for narrow screens:

- Chat - placeholder, llama.cpp, llama-cpp-python, Ollama, OpenAI-compatible/LM Studio, or Transformers text inference with tab status/errors.
- Vision - placeholder, llama.cpp, llama-cpp-python, Ollama, or Transformers MiniCPM image + prompt inference with tab status/errors.
- Dataset - local CSV/JSONL/NDJSON preview, optional Hugging Face dataset preview, stats, and tab status/errors.
- Train - LoRA dry-run training plan plus local base-vs-tuned exact-match evaluation.
- Export - GGUF download/conversion/quantization planning, exported-file listing, and existing-file downloads.
- Field Notes - saves human corrections to CSV, imports uncertain OCR predictions, captures media paths/training flags, exports corrected JSONL, corrected OCR JSONL, and local HF Dataset files.
- Traces - local event preview, JSONL trace rows, tracking status, and trace export.
- Agent - local non-autonomous research-plan-implement-verify trace mode.
- Status - shows configured models, backend metadata, local llama.cpp setup, LM Studio/OpenAI-compatible setup, SGLang command/check/stop planning, and Ollama list/pull planning.

Ollama is optional and is not installed automatically. Install and start Ollama yourself, then
pull a compatible model explicitly before selecting the Ollama backend in the app. The Status tab
can list models from a running local Ollama server and prepare an explicit
`ollama pull <model>` command. It shows the command only; it does not run downloads for you.

llama.cpp is the preferred hackathon backend path. Install llama.cpp separately, open the Status
tab, pick or type an explicit GGUF path, optionally pick an mmproj GGUF for vision, then click
`Prepare local model config`. The app writes `data/local_backends.yaml`, shows the
`llama-server` command, and still does not download or load model weights on startup.
Start `llama-server` yourself with that command, then select the `llama.cpp` backend in the app.
`llama-cpp-python` is also available as an optional backend when the package is installed and a
local GGUF path is configured.

LM Studio or another local OpenAI-compatible server is optional and is not a cloud API path. Start
the local server yourself, open the Status tab, set the server URL such as
`http://127.0.0.1:1234`, optionally enter the exact served model name shown by LM Studio, and
click `Save OpenAI-compatible config`. `Check server` calls `/v1/models`; selecting
`openai-compatible` in the Chat tab posts to `/v1/chat/completions` only after you submit a prompt.
No model weights are downloaded or loaded on app startup.

The Transformers text backend is optional. It requires installing `transformers` and a compatible
PyTorch build for your CPU/GPU, and it may download model weights when you explicitly select it and
run a prompt. It is not installed automatically and is not used on startup.

The Transformers vision backend is optional for MiniCPM-V models. It uses `AutoProcessor` and
`AutoModelForImageTextToText` only after you select `transformers` in the Vision tab and submit an
image. The current app maps the thinking toggle into the prompt template and documents video as a
future frame-sampling path; it does not download or load vision weights on startup.

SGLang is optional. The Status tab can prepare a local `python -m sglang.launch_server` command,
check `/health`, request `/shutdown`, and the Chat tab can use the `sglang` backend against
`/v1/chat/completions` after you start the server yourself. The app does not start SGLang on
startup.

Dataset preview supports local `.csv`, `.jsonl`, and `.ndjson` files, split names for optional
Hugging Face dataset preview, and basic local statistics. Field Notes can export corrected
training rows to `data/field_notes.jsonl` and local HF Dataset-style files to
`data/hf_field_notes/`.

The OCR correction hook expects a local `.csv`, `.jsonl`, or `.ndjson` prediction file with fields
such as `source_path`, `text`, and `confidence`. The Field Notes tab can preview rows at or below a
confidence threshold, import those uncertain rows as correction tasks, and export corrected OCR rows
to `data/ocr_corrections.jsonl`. This is the local OCR -> Field Notes -> Training path: import
uncertain predictions, correct them in Field Notes, then export corrected JSONL/HF Dataset files for
future fine-tuning or evaluation.

Local MCP-style tools live in `mcp_tools/tools.py`. The selected MCP path is Gradio native MCP,
enabled by `launch(mcp_server=True)`. The documented endpoint path is `/gradio_api/mcp/sse`.
`mcp_tools/bridge.py` exposes a local manifest and invocation helper for dataset stats, optional
HF dataset preview, safe arithmetic, model inference, and non-executing VINDEX call planning. Full
external MCP client verification still depends on launching the app and connecting a client.

The VINDEX helper is intentionally planning-only. It validates the PRD method names, reports
whether the local `vindex` package or `http://127.0.0.1:8765/health` endpoint is available, and
caps risky edit parameters. It does not edit model weights.

The Agent tab drafts local research-plan-implement-verify traces and paper-to-code traces, stores
them in `data/agent_traces.jsonl`, and can export JSONL or local HF Dataset-style trace files. It
includes safety gates and does not run shell commands, commit, push, deploy, download models, or
call external services. Agent traces can later become persona/source material for the Desk-Pet
extension, but no Desk-Pet runtime is implemented yet.

The Export tab is a planning surface. It shows explicit `huggingface-cli`,
`convert_hf_to_gguf.py`, and `llama-quantize` commands for the selected model and quantization,
plus a list of files already present in the export directory. Existing files are exposed through
the download output. It does not run those commands yet.

The Train tab does not start LoRA training yet. It builds a dry-run plan from
`config/training.yaml`, validates the dataset path, shows the planned checkpoint output directory,
documents hardware expectations, and builds a non-executing LoRA trainer request with PEFT/TRL
dependency status. It can also run a local evaluation by comparing newline-separated base and tuned
responses against the built-in prompt cases. It reports exact match, can calculate perplexity from
optional negative log likelihood values, shows a qualitative table, and appends tuned results to
`data/eval_results.jsonl`.

The vLLM tab is a local serving planner. It prepares an explicit `vllm serve ...` command, checks
`/health`, fetches `/metrics`, logs parsed benchmark metrics through local tracking, and points the
chat client at `/v1/chat/completions` only after you start vLLM yourself. It does not install vLLM
or start a server on app startup.

Tracing writes local events to `data/traces.jsonl` by default. The Traces tab can show recent app
events, show JSONL trace rows, report whether optional Trackio is available, and export local
traces to `exports/traces.jsonl`. Remote Trackio/HF sync still requires credentials and setup.

## 8. Hugging Face Space Deployment

The repo includes Hugging Face Space metadata in `README.md` and a local planning helper:

```powershell
.venv\Scripts\python.exe scripts\plan_hf_space.py --user <hf-user-or-org>
```

The helper validates required files and prints the manual commands for login, Space creation,
remote setup, and push. It does not login, create a repo, push, or store tokens.

Required Space files:

- `app.py`
- `requirements.txt`
- `README.md`
- `config/models.yaml`
- `config/training.yaml`

Hardware choice for the first Space should be CPU/basic while the app remains placeholder and
planning-first. Upgrade the Space hardware only after a real backend is selected and verified.
The app does not download model weights on startup; model downloads happen only through explicit
backend actions such as `ollama pull`, `huggingface-cli download`, or selecting a real Transformers
backend and running a prompt.

Trackio/HF sync path: local traces are written to JSONL first, then optional Trackio can be enabled
in `config/training.yaml` after package availability and credentials are ready.

Synthetic data and reward evaluation helpers are local Python utilities. `datasets/synthetic.py`
can generate, validate, filter, augment, and export JSONL examples. `training/reward_eval.py` can
score supplied responses, select best-of-N candidates, create DPO chosen/rejected pairs, and compare
base responses against LoRA responses. They do not load reward models or call external services.

## 9. How To Work With Codex

Useful prompts:

```text
Read docs/TASKS.md and implement the next unchecked MVP task.
```

```text
Read docs/IMPLEMENTATION_STATUS.md and tell me what is blocked.
```

```text
Wire the next backend, but keep automatic model downloads disabled on startup.
```

```text
Update the docs after the change and mark the matching checklist item.
```

```text
If this failed, add or update a test that catches it, then fix the code.
```

## 10. What To Avoid

- Do not start with the whole PRD at once.
- Do not download huge models automatically.
- Do not mark tasks done before running or documenting the blocker.
- Do not add features without tests.
- Do not add broad coverage escapes such as pragma no cover without a documented reason.
- Do not push secrets, tokens, model caches, GGUF files, or virtual environments to git.
