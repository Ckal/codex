---
title: Plant Identification Tool
colorFrom: green
colorTo: gray
sdk: gradio
app_file: ../plant_space_app.py
pinned: false
---

# Plant Identification Tool

A Gradio reference app from the OpenBMB Local AI Workbench template: small local models,
field notes for corrections, and a clear path from local demo to Hugging Face Space.

## Quick Start (Local)

Python 3.10+:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m plant.app --port 7861
```

## HF Space Deployment

This Space runs Plant Identification on OpenBMB MiniCPM-V or fine-tuned variants.
The model downloads on first inference; subsequent calls are fast.

- Identify plants from images
- Correct identifications and export as training data
- Zero-GPU compatible for Hugging Face Spaces with `@spaces.GPU` decorator
