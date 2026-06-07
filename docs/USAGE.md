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

Available tabs:

- Chat - placeholder, llama.cpp, llama-cpp-python, or Ollama text inference.
- Vision - placeholder, llama.cpp, llama-cpp-python, or Ollama image + prompt inference.
- Dataset - local CSV/JSONL/NDJSON preview, optional Hugging Face dataset preview, and stats.
- Train - LoRA training plan plus local base-vs-tuned exact-match evaluation.
- Export - GGUF download/conversion/quantization planning and exported-file listing.
- Field Notes - saves human corrections to CSV, captures media paths/training flags, exports corrected JSONL, and exports local HF Dataset files.
- Traces - local event preview, JSONL trace rows, tracking status, and trace export.
- Agent - local non-autonomous research-plan-implement-verify trace mode.
- Status - shows configured models, backend metadata, and local llama.cpp setup.

Ollama is optional and is not installed automatically. Install and start Ollama yourself, then
pull a compatible model explicitly before selecting the Ollama backend in the app.

llama.cpp is the preferred hackathon backend path. Install llama.cpp separately, open the Status
tab, pick or type an explicit GGUF path, optionally pick an mmproj GGUF for vision, then click
`Prepare local model config`. The app writes `data/local_backends.yaml`, shows the
`llama-server` command, and still does not download or load model weights on startup.
Start `llama-server` yourself with that command, then select the `llama.cpp` backend in the app.
`llama-cpp-python` is also available as an optional backend when the package is installed and a
local GGUF path is configured.

Dataset preview supports local `.csv`, `.jsonl`, and `.ndjson` files, split names for optional
Hugging Face dataset preview, and basic local statistics. Field Notes can export corrected
training rows to `data/field_notes.jsonl` and local HF Dataset-style files to
`data/hf_field_notes/`.

Local MCP-style tools live in `mcp_tools/tools.py`. They are Python functions for dataset stats,
optional HF dataset preview, safe arithmetic, and model inference. They are not served through an
MCP endpoint yet.

The Agent tab drafts a local research-plan-implement-verify trace, stores it in
`data/agent_traces.jsonl`, and can export JSONL or local HF Dataset-style trace files. It does not
run shell commands, commit, push, deploy, download models, or call external services.

The Export tab is a planning surface. It shows explicit `huggingface-cli`,
`convert_hf_to_gguf.py`, and `llama-quantize` commands for the selected model and quantization,
plus a list of files already present in the export directory. It does not run those commands yet.

The Train tab does not start LoRA training yet. It can run a local evaluation by comparing
newline-separated base and tuned responses against the built-in prompt cases. It reports exact
match, shows a qualitative table, and appends tuned results to `data/eval_results.jsonl`.

Tracing writes local events to `data/traces.jsonl` by default. The Traces tab can show recent app
events, show JSONL trace rows, report whether optional Trackio is available, and export local
traces to `exports/traces.jsonl`. Remote Trackio/HF sync still requires credentials and setup.

## 8. How To Work With Codex

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

## 9. What To Avoid

- Do not start with the whole PRD at once.
- Do not download huge models automatically.
- Do not mark tasks done before running or documenting the blocker.
- Do not add features without tests.
- Do not add broad coverage escapes such as pragma no cover without a documented reason.
- Do not push secrets, tokens, model caches, GGUF files, or virtual environments to git.
