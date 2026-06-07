# Hackathon Submission Package

This document collects the judge-facing story, demo flow, and submission assets for the Gradio
Hugging Face Build Small Hackathon.

## Track

Recommended track: Backyard AI.

Reason: the app is local-first, small-model focused, and designed to help a solo builder inspect,
test, correct, and document OpenBMB model workflows on their own machine before moving to a Space.

## Project Story

OpenBMB Local AI Workbench is a Gradio app for trying small OpenBMB models locally, capturing
human corrections as field notes, and turning those notes into training or evaluation artifacts.

## Target User

The target user is a hackathon builder or small-team AI tinkerer who wants a practical local
workflow before committing to cloud deployment, GPU rentals, or a larger training stack.

## Measurable Benefit

The app reduces setup uncertainty by keeping model choices, backend availability, field-note
exports, traces, and deployment next steps visible in one Gradio surface.

## Final Model Family

Primary family: OpenBMB MiniCPM.

Current configured models:

| Config ID | Model | Parameters | Role |
| --- | --- | ---: | --- |
| `minicpm5_1b` | `openbmb/MiniCPM5-1B` | 1B | local text baseline |
| `minicpm5_1b_thinking` | `openbmb/MiniCPM5-1B-Thinking` | 1B | reasoning/text variant |
| `minicpm41_8b` | `openbmb/MiniCPM4.1-8B` | 8B | long-context text candidate |
| `minicpm_v46` | `openbmb/MiniCPM-V-4.6` | 1.3B | vision candidate |
| `minicpm_v46_thinking` | `openbmb/MiniCPM-V-4.6-Thinking` | 1.3B | vision reasoning candidate |
| `minicpm_o45` | `openbmb/MiniCPM-o-4.5` | 8B | omnimodal stretch candidate |

All configured models are at or below the 32B hackathon limit.

## Badge Targets

- Local-first: yes, through placeholder, Ollama, llama.cpp, and llama-cpp-python paths.
- llama.cpp: target badge path; requires local llama.cpp install and GGUF model verification.
- Open trace: yes, through local JSONL tracking and trace export.
- Field notes/report: yes, through corrected field notes, JSONL export, and local HF Dataset-style export.

## Demo Flow

1. Open the Gradio app locally.
2. Show the Status tab and explain model-size compliance plus backend availability.
3. Use Chat in placeholder mode to demonstrate the workflow without downloading weights.
4. Use the Dataset tab to preview a local JSONL/CSV training candidate.
5. Save a correction in Field Notes and export corrected rows to JSONL.
6. Open Traces to show local event history and optional Trackio status.
7. Open Export to show GGUF conversion and quantization planning.
8. Explain the llama.cpp/Ollama path for replacing placeholder output with real local inference.
9. Show the GitHub repo and, when available, the Hugging Face Space URL.

## Demo Video Script

1. "This is OpenBMB Local AI Workbench, a Gradio app for small-model local experimentation."
2. "The model registry keeps every configured model below 32B parameters."
3. "The app starts safely in placeholder mode, so it never downloads model weights on startup."
4. "The Status tab shows which local backends are configured or missing."
5. "A user can try a prompt, capture a correction, and export those corrections as training data."
6. "The Traces tab records local workflow events for reproducibility."
7. "The Export tab prepares explicit GGUF conversion and quantization commands."
8. "The next deployment step is pushing this same Gradio app to a Hugging Face Space."

## Social Post Draft

Built OpenBMB Local AI Workbench for the Gradio Hugging Face Build Small Hackathon: a local-first
Gradio app for testing MiniCPM models, collecting field-note corrections, exporting training data,
planning GGUF/llama.cpp workflows, and keeping traceable evidence of small-model experiments.

GitHub: https://github.com/Ckal/codex
Space: pending

## Submission Checklist

- GitHub URL: https://github.com/Ckal/codex
- Hugging Face Space URL: pending
- Demo video URL: pending
- Social post URL: pending
- Field notes/report URL: pending
- Final track: Backyard AI
- App name: OpenBMB Local AI Workbench
- Deadline: June 15, 2026
