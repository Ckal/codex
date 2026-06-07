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
