# How To Extend The Workbench

## Add A New Model

1. Open `config/models.yaml`.
2. Add a new model entry.
3. Set `type` to `text`, `vision`, or `omnimodal`.
4. Keep `parameters_b` at or below 32 for hackathon eligibility.
5. Keep `backend: placeholder` until a real service supports it.

Example:

```yaml
models:
  my_model:
    hf_id: org/model-name
    display_name: My Model
    type: text
    parameters_b: 7
    backend: placeholder
    context_length: 32768
    local_first: true
    notes: Why this model is useful.
```

## Add A Real Backend

Create a service in `models/`, for example:

```text
models/ollama_service.py
```

The service should expose a small interface:

```python
class OllamaService:
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        ...
```

Then update the service factory or relevant UI tab to choose between placeholder and real services.

## Add A New Gradio Tab

1. Create `ui/new_tab.py`.
2. Add a `build_new_tab(...)` function.
3. Import it in `app.py`.
4. Add it inside the `gr.Tabs()` block.
5. Update `docs/ARCHITECTURE.md`.
6. Add a checklist item in `docs/TASKS.md`.
7. Update `docs/IMPLEMENTATION_STATUS.md`.

## Add Field Notes Export

Next useful field notes upgrade:

- Add a button to export `data/field_notes.csv` to JSONL.
- Add a button to upload that JSONL as a Hugging Face Dataset.
- Document the dataset schema in `README.md`.

Suggested JSONL schema:

```json
{"model_id":"minicpm5_1b","prompt":"...","response":"...","correction":"...","tags":["demo"]}
```

## Add OCR Corrections

The local OCR extension starts from prediction files rather than running an OCR engine directly.
Use `.csv`, `.jsonl`, or `.ndjson` rows with fields like:

```json
{"source_path":"receipt.png","text":"Tota1 12.30","confidence":0.54}
```

The Field Notes tab can preview uncertain rows, import them as correction tasks, and export
corrected OCR rows to `data/ocr_corrections.jsonl`. The intended wiring is:

```text
OCR predictions -> uncertain Field Notes -> corrected JSONL/HF Dataset -> training/evaluation
```

## Add VINDEX Execution

The current VINDEX integration is a safety boundary, not an edit runner. It validates the eight PRD
methods, builds non-executing local FastAPI call plans, and reports whether a local VINDEX package
or `http://127.0.0.1:8765/health` server is available.

Before allowing execution:

1. Verify the local VINDEX package or FastAPI server.
2. Re-check the PRD bug list: GPU cache cleanup, dead-code paths, star-spread over-editing, and
   causal-window limits.
3. Keep `star_spread.n_neighbors <= 5` and `calibrated_edit.causal_window <= 3` until the scaling
   formula is validated.
4. Add protected-relation tests for every edit workflow.
5. Only then add an explicit user-triggered execute button or MCP tool.

## Add Training

Training should be added only after local inference works.

Recommended order:

1. Export field notes to JSONL.
2. Load JSONL as a dataset.
3. Add PEFT/TRL LoRA for text model.
4. Add Trackio logging.
5. Add checkpoint output folder.
6. Add README instructions.

## Add Hugging Face Space Deployment

After the local app runs:

```powershell
.venv\Scripts\python.exe scripts\plan_hf_space.py --user <hf-user-or-org>
huggingface-cli login
huggingface-cli repo create openbmb-local-ai-workbench --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/<user>/openbmb-local-ai-workbench
git push space main
```

Never commit Hugging Face tokens.
