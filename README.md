# OpenBMB Local AI Workbench

A Gradio workbench for the Build Small Hackathon: small local models, practical experimentation,
and a clear path from local demo to Hugging Face Space.

## What This Is

The project turns the PRD in `HF_PRD_v1.md` into a staged implementation:

1. A working Gradio app shell.
2. Config-driven OpenBMB model registry.
3. Local-first inference path, starting with deterministic placeholder mode and optional Ollama/llama.cpp later.
4. Field notes for collecting corrections.
5. Extension points for training, GGUF export, Trackio traces, MCP tools, and agent workflows.

## Hackathon Fit

- **Track:** Backyard AI or Thousand Token Wood, depending on the final user story.
- **Canvas:** Gradio app, deployable to Hugging Face Spaces.
- **Small model rule:** target models stay at or below 32B parameters.
- **Bonus quests:** local-first, field notes/report, possible llama.cpp and trace sharing.

## Quick Start

Python is not currently available on PATH in this workspace shell. Install Python first, then:

```powershell
.\scripts\verify_structure.ps1
& "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe" -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
.\scripts\run_tests.ps1
python app.py
```

If `python` is not recognized, install Python 3.11+ from python.org or the Windows Store,
then reopen the terminal so PATH is refreshed.

Open the local URL shown by Gradio, usually `http://127.0.0.1:7860`.

## Project Structure

```text
.
|-- app.py
|-- AGENTS.md
|-- README.md
|-- requirements.txt
|-- config/
|   |-- models.yaml
|   `-- training.yaml
|-- core/
|   |-- events.py
|   `-- registry.py
|-- datasets/
|   `-- field_notes.py
|-- models/
|   |-- placeholder_service.py
|   `-- model_catalog.py
|-- ui/
|   |-- chat_tab.py
|   |-- dataset_tab.py
|   |-- export_tab.py
|   |-- notes_tab.py
|   |-- traces_tab.py
|   |-- agent_tab.py
|   |-- status_tab.py
|   |-- train_tab.py
|   `-- vision_tab.py
|-- data/
|   `-- .gitkeep
|-- exports/
|   `-- .gitkeep
|-- HF_PRD_v1.md
`-- HF_PRD_ext.md
```

## Project Docs

The working docs live in [docs/README.md](docs/README.md).

- [Task checklist](docs/TASKS.md)
- [Implementation status](docs/IMPLEMENTATION_STATUS.md)
- [PRD implementation matrix](docs/PRD_IMPLEMENTATION_MATRIX.md)
- [Acceptance criteria](docs/ACCEPTANCE_CRITERIA.md)
- [Roadmap](docs/ROADMAP.md)
- [Usage guide](docs/USAGE.md)
- [Architecture guide](docs/ARCHITECTURE.md)
- [Extension guide](docs/EXTENDING.md)
- [Test and quality policy](docs/ACCEPTANCE_CRITERIA.md)

## Current Truth

The full PRD is not implemented yet. The current app is a tested, quality-gated scaffold with
visible placeholder inference plus local backend planning surfaces. GitHub push is complete at
`https://github.com/Ckal/codex`. Real verified local inference, training execution, served MCP,
Hugging Face Space deployment, and most extension PRD items are still backlog work.

## Model Plan

Initial candidates from the PRD:

| Config ID | Model | Purpose |
| --- | --- | --- |
| `minicpm5_1b` | `openbmb/MiniCPM5-1B` | text chat, LoRA, local-first baseline |
| `minicpm_v46` | `openbmb/MiniCPM-V-4.6` | image/video understanding |
| `minicpm_o45` | `openbmb/MiniCPM-o-4.5` | omnimodal stretch goal |

The app starts in placeholder mode so it does not download large model files automatically.
llama.cpp, llama-cpp-python, and Ollama can be selected as backends, but the backend tool/package
must be installed and populated with the selected model explicitly by the user.

## Deployment Target

For Hugging Face Spaces, keep these files at repo root:

- `app.py`
- `requirements.txt`
- `README.md`
- `config/`
- `core/`
- `models/`
- `ui/`

Later deployment command:

```powershell
huggingface-cli login
huggingface-cli repo create openbmb-local-ai-workbench --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/<user>/openbmb-local-ai-workbench
git push space main
```

## Next Implementation Steps

1. Decide the exact hackathon story and user.
2. Choose model path: placeholder -> Ollama -> llama.cpp GGUF -> Transformers.
3. Add the first real inference service.
4. Add field-note export to JSONL/HF Dataset.
5. Polish README with screenshots, demo video script, and submission links.
