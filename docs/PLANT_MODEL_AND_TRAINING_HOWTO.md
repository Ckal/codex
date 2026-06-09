# Plant Model And Training How-To

This answers three common questions:

1. Does Plant Discovery use an LLM?
2. Does it use a specially trained plant model?
3. How do we train and then use one?

## Current Model Modes

Plant Discovery has three explicit modes.

| Mode | Command | Uses an LLM/VLM? | Trained specifically for this app? |
| --- | --- | --- | --- |
| Demo | `python -m plant.app --model-mode demo --port 7861` | No | No |
| OpenBMB zero-shot | `python -m plant.app --model-mode openbmb --port 7861` | Yes, MiniCPM-V | No, uses base OpenBMB model |
| Fine-tuned adapter | `python -m plant.app --model-mode finetuned --port 7861` | Yes, MiniCPM-V plus adapter | Yes, after you train/publish an adapter |

The app defaults to `openbmb` mode. The `--no-model` flag is an alias for demo mode and exists for
tests, screenshots, and machines without GPU/model dependencies.

## What Runs Today

- Demo mode is fully verified locally.
- OpenBMB mode is implemented but requires optional dependencies and model weights.
- Fine-tuned mode is implemented as an adapter-loading path, but there is no real adapter yet.

This means the app can use a real OpenBMB VLM, but this workspace has not installed the heavy
`plant/requirements.txt` dependencies or downloaded model weights.

## Step 1 - Use OpenBMB MiniCPM-V Zero-Shot

Install optional plant dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r plant\requirements.txt
```

Run the app with the real OpenBMB model path:

```powershell
.venv\Scripts\python.exe -m plant.app --model-mode openbmb --port 7861
```

Open `http://127.0.0.1:7861`, upload a public-safe plant image, and click `Identify plant`.

Expected behavior:

- The UI shows model status with `uses_llm: true`.
- The first identify action loads `openbmb/MiniCPM-V-4.6`.
- If confidence is low and auto-thinking is enabled, the app can try
  `openbmb/MiniCPM-V-4.6-Thinking`.
- Output is parsed into the `PlantID` JSON schema.

## Step 2 - Collect Corrections

Use the app normally:

1. Upload plant image.
2. Identify.
3. Add a human correction when the model is wrong.
4. Save correction.
5. Open `Corrections`.
6. Click `Export training JSONL`.

The export path is:

```text
data/plant_training.jsonl
```

Do not train on one or two examples. The current minimum recommendation is 30 corrected examples,
and more is better.

## Step 3 - Plan Training

Generate a non-executing training plan:

```powershell
.venv\Scripts\python.exe scripts\plan_plant_training.py --corrected-examples 30
```

This prints:

- dependency availability,
- SWIFT command preview,
- LLaMA-Factory command preview,
- publish commands,
- command to run the trained adapter.

The Gradio UI also exposes the same non-executing plan from the `Corrections` tab.

## Step 4 - Train Locally

Preferred path for MiniCPM-V vision LoRA is SWIFT or another tool that supports multimodal LoRA.
The generated SWIFT command looks like this:

```powershell
swift sft --model openbmb/MiniCPM-V-4.6 --dataset data/plant_training.jsonl --lora_rank 16 --num_train_epochs 3 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --learning_rate 2.0e-4 --freeze_vit true --output_dir checkpoints/plant_lora
```

Run it only after:

- optional dependencies are installed,
- GPU memory is sufficient,
- exported JSONL has been reviewed,
- private images/notes are removed,
- you accept that training may take time and disk space.

## Step 5 - Publish Or Configure The Adapter

After training, publish the adapter or keep it local. If publishing:

```powershell
huggingface-cli repo create your-username/minicpm-v46-plant-lora --type model
huggingface-cli upload your-username/minicpm-v46-plant-lora checkpoints/plant_lora .
```

Then edit `plant/models.yaml`:

```yaml
plant_vlm_finetuned:
  base_model: openbmb/MiniCPM-V-4.6
  adapter_id: your-username/minicpm-v46-plant-lora
```

Run:

```powershell
.venv\Scripts\python.exe -m plant.app --model-mode finetuned --port 7861
```

## Important Honesty Rule

Do not claim Plant Discovery uses a specially trained model until:

- a real adapter exists,
- it is configured in `plant/models.yaml`,
- `--model-mode finetuned` has been run,
- evaluation shows it improves over the base OpenBMB model.

Until then, the honest claim is:

> Plant Discovery uses OpenBMB MiniCPM-V zero-shot, with a correction loop and documented path to
> fine-tune a plant adapter.
