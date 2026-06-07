# OpenBMB Workbench — Planned Extensions: Detailed Spec
> 2026-06-05 | Companion to PRD v2.0

---

## Table of Contents

1. [vLLM Serving Tab](#1-vllm-serving-tab)
2. [Ollama Quick-Start](#2-ollama-quick-start)
3. [Reward Model Eval](#3-reward-model-eval)
4. [Synthetic Data Gen](#4-synthetic-data-gen)
5. [Paper-to-Code Agent](#5-paper-to-code-agent)
6. [HF Spaces Deploy](#6-hf-spaces-deploy)
7. [VINDEX Integration](#7-vindex-integration)
8. [OCR Pipeline Hook](#8-ocr-pipeline-hook)
9. [MiniCPM Desk-Pet](#9-minicpm-desk-pet)
10. [MiniCPM-o Audio Tab](#10-minicpm-o-audio-tab)
11. [Cross-Extension Wiring](#11-cross-extension-wiring)

---

## 1. vLLM Serving Tab

### What it is

vLLM is a production-grade inference engine built around *PagedAttention* — a KV-cache management
algorithm that treats GPU memory like virtual memory pages. The result is dramatically higher
throughput when multiple requests run concurrently, compared to naive Transformers inference.

In the workbench context, vLLM adds a fourth inference mode alongside llama.cpp, SGLang, and
Ollama. You use it when you want OpenAI-compatible HTTP endpoints, continuous batching, or when
benchmarking production serving latency.

### Why it matters

| Scenario | Benefit |
|----------|---------|
| Benchmarking fine-tuned LoRA | Compare throughput before/after fine-tune |
| Multi-user demo | Queue and batch concurrent requests |
| Production deployment | OpenAI-compatible API, drop-in for existing tooling |
| MiniCPM4.1-8B long context | PagedAttention shines on 128K context — avoids OOM |

### Architecture

```
models/vllm_runner.py
  VLLMRunner
    .start(model_id, cfg)      → subprocess: vllm serve ...
    .stop()                    → terminate subprocess
    .chat(messages) → str      → POST /v1/chat/completions
    .batch(prompts) → list[str]→ concurrent POST via asyncio
    .stats() → dict            → GET /metrics (Prometheus)
```

```python
# models/vllm_runner.py
import subprocess, asyncio, requests
from openai import AsyncOpenAI

class VLLMRunner:
    def __init__(self, cfg: dict):
        self.model_id   = cfg["hf_id"]
        self.port       = cfg.get("port", 8000)
        self.gpu_memory = cfg.get("gpu_memory_utilization", 0.85)
        self.trust_rc   = cfg.get("trust_remote_code", False)
        self._proc      = None
        self._client    = AsyncOpenAI(
            base_url=f"http://localhost:{self.port}/v1",
            api_key="vllm-local"
        )

    def start(self):
        cmd = [
            "vllm", "serve", self.model_id,
            "--port",                    str(self.port),
            "--gpu-memory-utilization",  str(self.gpu_memory),
        ]
        if self.trust_rc:
            cmd.append("--trust-remote-code")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._wait_ready()

    def _wait_ready(self, timeout: int = 60):
        import time
        for _ in range(timeout):
            try:
                r = requests.get(f"http://localhost:{self.port}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM server did not become healthy")

    async def chat(self, messages: list[dict], **kwargs) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return resp.choices[0].message.content

    async def batch(self, prompts: list[str], system: str = "") -> list[str]:
        tasks = [
            self.chat([
                {"role": "system",  "content": system},
                {"role": "user",    "content": p},
            ])
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    def stats(self) -> dict:
        """Prometheus metrics parsed to dict. Returns throughput + latency."""
        r = requests.get(f"http://localhost:{self.port}/metrics")
        lines = r.text.splitlines()
        metrics = {}
        for line in lines:
            if line.startswith("vllm:") and not line.startswith("#"):
                key, val = line.rsplit(" ", 1)
                metrics[key] = float(val)
        return metrics

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None
```

### UI tab (models_tab.py → extend)

Add a "vLLM" section to the models tab:

```python
with gr.Tab("⚡ vLLM"):
    model_dd      = gr.Dropdown(label="Model", choices=list_vllm_compatible())
    gpu_mem_sl    = gr.Slider(0.5, 0.95, value=0.85, label="GPU memory utilization")
    start_btn     = gr.Button("Start server")
    stop_btn      = gr.Button("Stop server")
    status_lbl    = gr.Label(label="Status")
    metrics_json  = gr.JSON(label="Live metrics")

    def start_vllm(model_id, gpu_mem):
        cfg = {**model_registry.get(model_id).cfg,
               "gpu_memory_utilization": gpu_mem}
        runner = VLLMRunner(cfg)
        runner.start()
        model_registry.register(f"{model_id}_vllm", runner)
        return "Running", runner.stats()

    start_btn.click(start_vllm, [model_dd, gpu_mem_sl], [status_lbl, metrics_json])
```

### Trackio events fired

```python
trackio.init(project="workbench", run_name="vllm_benchmark")
trackio.log({"throughput_tok_per_s": ..., "p50_latency_ms": ..., "gpu_mem_used": ...})
trackio.finish()
```

---

## 2. Ollama Quick-Start

### What it is

Ollama is zero-configuration local model serving. One `ollama pull` downloads a quantized model
and one `ollama serve` runs it. No CUDA setup, no Python environment issues. The REST API is
OpenAI-compatible on port 11434.

MiniCPM-V-4.6 and MiniCPM5-1B are both in the Ollama registry:
```bash
ollama pull openbmb/minicpm-v4.6
ollama pull openbmb/minicpm5-1b
```

### Why it matters

Ollama is the fastest path from "nothing" to "running model" — ideal for demos, non-GPU machines
(Apple Silicon is well-optimized), and users who shouldn't need to understand quantization.

### Architecture

```
models/ollama_runner.py
  OllamaRunner
    .pull(model_id)            → subprocess: ollama pull ...
    .chat(messages) → str      → POST http://localhost:11434/api/chat
    .generate(prompt) → str    → POST http://localhost:11434/api/generate (streaming)
    .list() → list[str]        → GET /api/tags
```

```python
# models/ollama_runner.py
import requests, subprocess, json
from typing import Generator

class OllamaRunner:
    BASE = "http://localhost:11434"

    def __init__(self, model_id: str):
        # Ollama uses "openbmb/minicpm-v4.6" style IDs directly
        self.model_id = model_id

    @staticmethod
    def pull(model_id: str):
        subprocess.run(["ollama", "pull", model_id], check=True)

    @staticmethod
    def list_local() -> list[str]:
        r = requests.get(f"{OllamaRunner.BASE}/api/tags")
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, messages: list[dict], stream: bool = False) -> str | Generator:
        payload = {"model": self.model_id, "messages": messages, "stream": stream}
        r = requests.post(f"{self.BASE}/api/chat", json=payload, stream=stream)
        if not stream:
            return r.json()["message"]["content"]
        # Generator for Gradio streaming
        def _stream():
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk["message"]["content"]
                    if chunk.get("done"):
                        break
        return _stream()

    def vision_chat(self, image_b64: str, prompt: str) -> str:
        """Multimodal chat — Ollama passes images as base64 in the message."""
        messages = [{
            "role":    "user",
            "content": prompt,
            "images":  [image_b64],
        }]
        return self.chat(messages)
```

### UI integration

The models tab gets an "Ollama" subtab with a model browser, pull button, and instant chat
that requires zero setup — the simplest entry point to the whole workbench.

```python
with gr.Tab("🦙 Ollama"):
    available = gr.Dropdown(label="Pull model",
        choices=["openbmb/minicpm-v4.6", "openbmb/minicpm5-1b"],
        allow_custom_value=True)
    pull_btn   = gr.Button("Pull")
    local_list = gr.JSON(label="Locally available")
    pull_status= gr.Textbox(label="Status")

    pull_btn.click(
        lambda m: (OllamaRunner.pull(m), OllamaRunner.list_local()),
        [available],
        [pull_status, local_list]
    )
```

---

## 3. Reward Model Eval

### What it is

A reward model is a model trained to score (prompt, response) pairs — answering "how good is this
output?" It's the missing piece between fine-tuning and verified alignment improvement. Without it
you can train a LoRA and only know quantitatively that loss went down, not whether outputs actually
got better by human-relevant criteria.

### Why it matters

- Validates that LoRA fine-tuning improved quality (not just minimized loss)
- Enables best-of-N sampling: generate N responses, keep highest-scored
- Enables DPO data creation: generate response pairs, reward model labels preferences
- Closes the RLHF loop within the workbench itself

### Reward model options

| Model | Size | Focus |
|-------|------|-------|
| `OpenAssistant/reward-model-deberta-v3-large-v2` | 450M | General helpfulness |
| `Salesforce/SFR-Reward-FsfairX-LLaMA3-RM-v0.1` | 8B | Instruction following |
| MiniCPM5-1B itself (self-eval) | 1B | Domain-specific, via prompt |

For the workbench, using MiniCPM5-1B as a judge (LLM-as-judge pattern) is the lowest-friction
option since the model is already loaded.

### Architecture

```
training/reward_eval.py
  RewardEvaluator
    .score(prompt, response) → float
    .best_of_n(prompt, n, generator) → str
    .create_dpo_pairs(dataset, generator, n=4) → Dataset
    .eval_lora_vs_base(base_svc, lora_svc, eval_ds) → dict
```

```python
# training/reward_eval.py
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class RewardEvaluator:
    def __init__(self, reward_model_id: str = "OpenAssistant/reward-model-deberta-v3-large-v2"):
        self.pipe = pipeline(
            "text-classification",
            model=reward_model_id,
            device=0 if torch.cuda.is_available() else -1,
        )

    def score(self, prompt: str, response: str) -> float:
        """Returns a scalar reward score (higher = better)."""
        text = f"Human: {prompt}\n\nAssistant: {response}"
        result = self.pipe(text, truncation=True, max_length=512)
        return result[0]["score"]

    def best_of_n(
        self,
        prompt: str,
        n: int,
        generator_fn,    # callable: prompt -> str
    ) -> tuple[str, float]:
        """Generate N responses, return the one with highest reward."""
        candidates = [(generator_fn(prompt), ) for _ in range(n)]
        scored = [(r[0], self.score(prompt, r[0])) for r in candidates]
        return max(scored, key=lambda x: x[1])

    def create_dpo_pairs(
        self,
        prompts: list[str],
        generator_fn,
        n: int = 4,
    ):
        """
        For each prompt: generate n responses, rank by reward,
        return (prompt, chosen, rejected) triplets for DPO training.
        """
        from datasets import Dataset
        rows = []
        for prompt in prompts:
            responses = [(generator_fn(prompt), ) for _ in range(n)]
            scored = sorted(
                [(r[0], self.score(prompt, r[0])) for r in responses],
                key=lambda x: x[1], reverse=True
            )
            if len(scored) >= 2:
                rows.append({
                    "prompt":   prompt,
                    "chosen":   scored[0][0],
                    "rejected": scored[-1][0],
                    "reward_gap": scored[0][1] - scored[-1][1],
                })
        return Dataset.from_list(rows)

    def eval_lora_vs_base(
        self,
        base_svc,
        lora_svc,
        eval_prompts: list[str],
    ) -> dict:
        """
        Compare base vs LoRA checkpoint by average reward score.
        Returns win rate and per-prompt scores.
        """
        base_scores = [self.score(p, base_svc.generate(p)) for p in eval_prompts]
        lora_scores = [self.score(p, lora_svc.generate(p)) for p in eval_prompts]
        wins = sum(l > b for l, b in zip(lora_scores, base_scores))
        return {
            "base_mean":   sum(base_scores) / len(base_scores),
            "lora_mean":   sum(lora_scores) / len(lora_scores),
            "lora_win_rate": wins / len(eval_prompts),
            "per_prompt":  list(zip(eval_prompts, base_scores, lora_scores)),
        }
```

### Trackio logging

```python
results = evaluator.eval_lora_vs_base(base_svc, lora_svc, eval_prompts)
trackio.init(project="workbench", run_name="reward_eval")
trackio.log({
    "base_reward_mean":     results["base_mean"],
    "lora_reward_mean":     results["lora_mean"],
    "lora_win_rate":        results["lora_win_rate"],
})
trackio.finish()
```

---

## 4. Synthetic Data Gen

### What it is

The ml-intern finding: *when real data is insufficient, have an LLM generate training data*.
This module does exactly that — it uses a capable model (MiniCPM4.1-8B or a cloud model via
HF Router) to generate diverse, high-quality (prompt, response) pairs on demand.

### Why it matters

Real-world fine-tuning is often blocked not by compute but by data. You have 50 good examples
but need 5000. Synthetic gen + quality filtering bridges that gap, especially for specialized
domains (plant species, historical OCR corrections, industrial inspection defect labels).

### Architecture

```
datasets/synthetic.py
  SyntheticGenerator
    .generate(topic, n, schema) → Dataset
    .augment(existing_ds, n) → Dataset
    .filter_quality(ds, min_score) → Dataset
    .generate_dpo_pairs(topic, n) → Dataset
```

```python
# datasets/synthetic.py
import json
from datasets import Dataset

GENERATION_PROMPT = """You are a training data generator. Generate {n} diverse, high-quality 
training examples for the topic: {topic}.

Output ONLY a valid JSON array. Each item must have these fields: {schema}
No explanation, no markdown, no preamble. Raw JSON array only."""

class SyntheticGenerator:
    def __init__(self, generator_svc):
        """generator_svc: any loaded ModelService with a .generate(prompt) method."""
        self.gen = generator_svc

    def generate(
        self,
        topic: str,
        n: int = 100,
        schema: dict | None = None,
    ) -> Dataset:
        """
        Generate n training examples on a topic.
        schema: dict of field_name → description, e.g.
          {"instruction": "task to perform", "response": "ideal answer"}
        """
        schema = schema or {"instruction": "user task", "response": "ideal answer"}
        schema_str = ", ".join(f'"{k}": "{v}"' for k, v in schema.items())

        # Generate in batches of 20 to stay within context
        rows = []
        for batch_start in range(0, n, 20):
            batch_n = min(20, n - batch_start)
            prompt = GENERATION_PROMPT.format(
                n=batch_n, topic=topic, schema="{" + schema_str + "}"
            )
            raw = self.gen.generate(prompt)
            try:
                # Strip any accidental markdown fences
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```")
                batch = json.loads(clean)
                rows.extend(batch)
            except json.JSONDecodeError:
                # Skip malformed batches; log the failure
                continue

        return Dataset.from_list(rows)

    def augment(self, existing_ds: Dataset, n: int) -> Dataset:
        """
        Use existing examples as few-shot demonstrations to generate n more.
        Samples up to 5 examples from existing_ds as context.
        """
        import random
        samples = existing_ds.shuffle().select(range(min(5, len(existing_ds))))
        examples_str = json.dumps(samples.to_list(), indent=2)

        prompt = f"""Here are {len(samples)} example training items:
{examples_str}

Generate {n} MORE diverse examples in the exact same JSON format.
Output only the JSON array, no explanation."""

        raw = self.gen.generate(prompt)
        try:
            clean = raw.strip().lstrip("```json").rstrip("```")
            new_rows = json.loads(clean)
        except json.JSONDecodeError:
            return existing_ds
        return Dataset.from_list(existing_ds.to_list() + new_rows)

    def filter_quality(
        self,
        ds: Dataset,
        reward_evaluator=None,
        min_score: float = 0.6,
    ) -> Dataset:
        """
        Filter with reward model if available, else heuristic filters.
        Heuristics: min length, no repeated n-grams, valid JSON fields.
        """
        if reward_evaluator:
            def _score(row):
                return reward_evaluator.score(
                    row.get("instruction", ""),
                    row.get("response", "")
                ) >= min_score
            return ds.filter(_score)
        else:
            # Basic heuristics
            def _heuristic(row):
                resp = row.get("response", "")
                return (
                    len(resp) >= 20 and       # not too short
                    len(resp) <= 4096 and     # not too long
                    resp.count(resp[:20]) < 3  # not repetitive
                )
            return ds.filter(_heuristic)

    def generate_for_domain(
        self,
        domain: str,
        output_path: str,
        n: int = 500,
    ):
        """
        Convenience method: generate, augment, filter, save to disk.
        Use for plant ID: domain="Plant species identification from photo descriptions"
        """
        ds = self.generate(topic=domain, n=n // 2)
        ds = self.augment(ds, n=n // 2)
        ds = self.filter_quality(ds)
        ds.save_to_disk(output_path)
        return ds
```

### Domain-specific example: plant ID

```python
gen = SyntheticGenerator(model_registry.get("minicpm41_8b"))
plant_ds = gen.generate_for_domain(
    domain="Identifying plant species from visual descriptions. "
           "Include common name, latin name, family, key visual features, and care tips.",
    output_path="data/synthetic_plants",
    n=2000,
)
# → 2000 synthetic (description → species JSON) training pairs
```

---

## 5. Paper-to-Code Agent

### What it is

An autonomous agent that takes an arXiv paper URL or title, reads the methodology section,
and implements the described technique within the workbench codebase. Directly inspired by
the ml-intern architecture (Research → Plan → Implement → Trace).

### Why it matters

The gap between reading a paper and running an experiment is usually days of engineering. This
agent compresses that to minutes for techniques that fit the workbench's model family.
Practical use cases: implement a new PEFT variant, add a new evaluation metric, adapt a new
data augmentation from a recent VLM paper.

### Architecture

```
agent/paper_agent.py
  PaperAgent
    .run(paper_ref) → AgentResult
      → Phase 1: Research (fetch + parse paper)
      → Phase 2: Plan (identify workbench integration points)
      → Phase 3: Implement (generate + write code)
      → Phase 4: Test (run + log to Trackio)
      → Phase 5: Trace (upload session to HF Dataset)
```

```python
# agent/paper_agent.py
import re
from dataclasses import dataclass, field
from huggingface_hub import HfApi
from smolagents import CodeAgent, HfApiModel
import trackio

@dataclass
class AgentResult:
    paper_title: str = ""
    summary:     str = ""
    files_modified: list[str] = field(default_factory=list)
    test_results:   dict      = field(default_factory=dict)
    trace_url:      str       = ""

class PaperAgent:
    SYSTEM_PROMPT = """You are an ML engineer working inside the OpenBMB Workbench codebase.
Given a research paper, your job is to:
1. Understand the core algorithm or technique.
2. Identify which module in the workbench it extends (training/, models/, datasets/, tools/).
3. Implement it as a new class or function, following the existing patterns.
4. Write a simple test that runs within the workbench and logs results to Trackio.

The workbench uses: transformers, peft, trl, trackio, mcp, gradio.
All new code must: fire events via the EventBus, log to Trackio, register in the Registry."""

    def __init__(self, orchestrator_model: str = "openbmb/MiniCPM4.1-8B"):
        self.model   = HfApiModel(orchestrator_model)
        self.api     = HfApi()
        self._log    = []

    def run(self, paper_ref: str) -> AgentResult:
        """
        paper_ref: arXiv URL like "https://arxiv.org/abs/2106.09685"
                   or paper title like "LoRA: Low-Rank Adaptation of Large Language Models"
        """
        result = AgentResult()

        # Phase 1: Research
        paper_text = self._fetch_paper(paper_ref)
        result.paper_title = self._extract_title(paper_text)

        # Phase 2: Plan
        plan = self._plan(paper_text)

        # Phase 3: Implement
        code_files = self._implement(plan, paper_text)
        result.files_modified = list(code_files.keys())
        for path, code in code_files.items():
            self._write_file(path, code)

        # Phase 4: Test
        trackio.init(project="workbench", run_name=f"paper_agent_{result.paper_title[:30]}")
        test_result = self._test(code_files)
        result.test_results = test_result
        trackio.log({"test_passed": test_result.get("passed", False), **test_result})
        trackio.finish()

        # Phase 5: Trace (ml-intern pattern)
        result.trace_url = self._upload_trace(result)

        return result

    def _fetch_paper(self, paper_ref: str) -> str:
        """Fetch paper text via HF Papers API or arXiv."""
        import requests
        if "arxiv.org" in paper_ref:
            arxiv_id = paper_ref.split("/abs/")[-1]
            r = requests.get(f"https://export.arxiv.org/abs/{arxiv_id}")
            return r.text
        # Fall back to HF Papers search
        from huggingface_hub import list_papers
        results = list(list_papers(query=paper_ref, limit=1))
        return str(results[0]) if results else ""

    def _plan(self, paper_text: str) -> str:
        """Ask the LLM to analyze the paper and produce an integration plan."""
        agent = CodeAgent(tools=[], model=self.model, max_steps=5)
        return agent.run(
            f"Read this paper excerpt and produce a 5-step integration plan "
            f"for the OpenBMB Workbench:\n\n{paper_text[:8000]}"
        )

    def _implement(self, plan: str, paper_text: str) -> dict[str, str]:
        """Generate code files from the plan."""
        agent = CodeAgent(tools=[], model=self.model, max_steps=15)
        code = agent.run(
            f"Implementation plan:\n{plan}\n\n"
            f"Paper details:\n{paper_text[:4000]}\n\n"
            f"Generate the Python file(s). Return a JSON dict: "
            f"{{\"path/to/file.py\": \"file_content\", ...}}"
        )
        import json
        try:
            return json.loads(code)
        except Exception:
            return {}

    def _write_file(self, path: str, content: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _test(self, code_files: dict) -> dict:
        """Run a quick import + instantiation test on generated files."""
        results = {}
        for path in code_files:
            try:
                module_name = path.replace("/", ".").replace(".py", "")
                import importlib.util, sys
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                results[path] = "imported_ok"
            except Exception as e:
                results[path] = f"error: {e}"
        results["passed"] = all("ok" in v for v in results.values())
        return results

    def _upload_trace(self, result: AgentResult) -> str:
        """Upload session trace to private HF Dataset (ml-intern pattern)."""
        import json, datetime
        session = {
            "timestamp":      datetime.datetime.utcnow().isoformat(),
            "paper":          result.paper_title,
            "files_modified": result.files_modified,
            "test_results":   result.test_results,
        }
        user = self.api.whoami()["name"]
        dataset_id = f"{user}/workbench-paper-sessions"
        # Upload as JSONL
        # ... (HF Dataset API)
        return f"https://huggingface.co/datasets/{dataset_id}"
```

---

## 6. HF Spaces Deploy

### What it is

One-click packaging and deployment of the current workbench state as a Hugging Face Space.
The Space runs the same Gradio app on HF infrastructure, shareable via URL, with ZeroGPU
support for serverless GPU access.

### Why it matters

Sharing a workbench state with a collaborator currently requires: git push, environment setup,
model download, config sync. With one-click deploy: copy URL → colleague sees the live app.
For hackathons this is especially valuable — deploy a domain-specific variant as a demo Space
in under 2 minutes.

### Architecture

```
deploy/spaces.py
  SpacesDeployer
    .prepare_repo() → creates/updates HF Space repo
    .upload_code()  → pushes app code (not model weights)
    .set_hardware(tier) → sets GPU tier in README
    .configure_secrets(env_vars) → sets HF Space secrets
    .deploy() → trigger Space rebuild
    .get_url() → returns live Space URL
```

```python
# deploy/spaces.py
import os, shutil, tempfile
from pathlib import Path
from huggingface_hub import HfApi, SpaceHardware

class SpacesDeployer:
    HARDWARE_MAP = {
        "cpu":    SpaceHardware.CPU_BASIC,
        "t4":     SpaceHardware.T4_SMALL,
        "t4_lg":  SpaceHardware.T4_MEDIUM,
        "a10":    SpaceHardware.A10G_SMALL,
        "a100":   SpaceHardware.A100_LARGE,
        "zero":   SpaceHardware.CPU_BASIC,   # ZeroGPU: uses CPU_BASIC + @spaces.GPU
    }
    EXCLUDE = {".git", "__pycache__", "exports", "data", "checkpoints",
               "*.gguf", "*.bin", "*.safetensors", ".env"}

    def __init__(self, space_id: str, hardware: str = "zero"):
        self.api      = HfApi()
        self.space_id = space_id   # "username/my-workbench"
        self.hardware = hardware

    def deploy(
        self,
        src_dir: str = ".",
        env_vars: dict | None = None,
    ) -> str:
        """Full deploy pipeline. Returns live Space URL."""
        self._create_or_update_repo()
        self._upload_code(src_dir)
        self._configure_secrets(env_vars or {})
        self._patch_app_for_zerogpu()
        return f"https://huggingface.co/spaces/{self.space_id}"

    def _create_or_update_repo(self):
        try:
            self.api.create_repo(
                repo_id=self.space_id,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Repo create/update: {e}")

    def _upload_code(self, src_dir: str):
        with tempfile.TemporaryDirectory() as tmp:
            src  = Path(src_dir)
            dest = Path(tmp)
            # Copy only non-excluded files
            for item in src.rglob("*"):
                if any(item.match(pat) for pat in self.EXCLUDE):
                    continue
                rel = item.relative_to(src)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.is_file():
                    shutil.copy2(item, target)
            self.api.upload_folder(
                folder_path=str(dest),
                repo_id=self.space_id,
                repo_type="space",
                commit_message="workbench deploy",
            )

    def _configure_secrets(self, env_vars: dict):
        for key, val in env_vars.items():
            self.api.add_space_secret(
                repo_id=self.space_id,
                key=key,
                value=val,
            )
        # Always set hardware
        self.api.request_space_hardware(
            repo_id=self.space_id,
            hardware=self.HARDWARE_MAP[self.hardware],
        )

    def _patch_app_for_zerogpu(self):
        """
        If hardware=zero, wrap inference functions with @spaces.GPU decorator.
        Uploads a patched app.py.
        """
        if self.hardware != "zero":
            return
        # Read existing app.py from Space, inject @spaces.GPU, re-upload
        app_content = self.api.hf_hub_download(
            repo_id=self.space_id, filename="app.py", repo_type="space"
        )
        with open(app_content) as f:
            code = f.read()
        # Simple injection: add import and decorator before inference functions
        patched = "import spaces\n" + code.replace(
            "def run_inference(",
            "@spaces.GPU\ndef run_inference("
        )
        self.api.upload_file(
            path_or_fileobj=patched.encode(),
            path_in_repo="app.py",
            repo_id=self.space_id,
            repo_type="space",
        )
```

### UI: deploy button in any tab header

```python
with gr.Row():
    space_id_box = gr.Textbox(placeholder="username/my-workbench", label="Space ID")
    hardware_dd  = gr.Dropdown(["cpu","t4","a10","zero"], value="zero", label="Hardware")
    deploy_btn   = gr.Button("🚀 Deploy to HF Spaces")
    deploy_url   = gr.Textbox(label="Live URL", interactive=False)

    def do_deploy(space_id, hw):
        d = SpacesDeployer(space_id, hw)
        url = d.deploy(env_vars={"HF_TOKEN": os.environ.get("HF_TOKEN","")})
        return url

    deploy_btn.click(do_deploy, [space_id_box, hardware_dd], deploy_url)
```

---

## 7. VINDEX Integration

### What it is

VINDEX is your own knowledge-editing engine (ki-fusion-labs.de). It exposes eight methods for
mechanistic interpretability and targeted weight editing on transformer models. Integrating it
into the workbench closes the loop between training a LoRA and verifying (or surgically fixing)
what the model actually knows.

VINDEX endpoints (from your PRD v1):
- `logit_lens` — per-layer prediction visualization
- `slot_neighbors` — embedding space neighbors
- `layer_contribution` — per-layer influence on final prediction
- `transition_spectrum` — attention head transition matrix
- `calibrated_edit` — targeted fact edit in weight space
- `derive_scale` — scaling factor derivation for safe edits
- `star_spread` — spread edit across semantically related slots
- `protect_relations` — guard against side effects

### Why it matters for the workbench

After LoRA fine-tuning a model on plant data:
1. `logit_lens` — does the model correctly predict "Rosa" at the right layer for a rose image?
2. `calibrated_edit` — if it consistently misidentifies Acer as Quercus, edit that slot
3. `star_spread` — propagate the Acer correction to closely related maple species
4. `protect_relations` — verify the edit didn't break "plant → living thing → organism"

This is not possible with LoRA alone — LoRA changes weight statistics globally. VINDEX does
surgical point edits, making it a *complement* to LoRA, not a replacement.

### Architecture

```
tools/vindex_tool.py
  VINDEXClient
    .logit_lens(model, tokenizer, text) → dict[layer, prediction]
    .calibrated_edit(model, subject, relation, old_obj, new_obj) → model
    .star_spread(model, anchor_subject, n=5) → list[affected_slots]
    .protect_relations(model, protected_triplets) → model
    .layer_contribution(model, text) → dict[layer, score]
    .slot_neighbors(model, token_id, n=10) → list[str]
```

```python
# tools/vindex_tool.py
"""
VINDEX integration.
Assumes VINDEX FastAPI server is running locally on port 8765,
OR VINDEX modules are importable from your local install.
"""
import requests
from mcp.server.fastmcp import FastMCP

VINDEX_BASE = "http://localhost:8765"   # your local VINDEX FastAPI

mcp = FastMCP("VINDEXTools")

@mcp.tool()
async def logit_lens(
    model_id:  str,
    text:      str,
    layer_range: tuple[int, int] = (0, -1),
) -> dict:
    """
    Run logit lens on a loaded model for the given text.
    Returns per-layer top-5 token predictions and probabilities.
    Useful for finding the 'phase layer' where the model commits to an answer.
    """
    r = requests.post(f"{VINDEX_BASE}/logit_lens", json={
        "model_id":    model_id,
        "text":        text,
        "layer_range": list(layer_range),
    })
    return r.json()

@mcp.tool()
async def calibrated_edit(
    model_id:  str,
    subject:   str,
    relation:  str,
    old_obj:   str,
    new_obj:   str,
    causal_window: int = 3,   # ±3 layers around logit lens phase layer
) -> dict:
    """
    Perform a targeted knowledge edit: change the model's belief about
    (subject, relation) from old_obj to new_obj.
    causal_window: restrict causal search to ±N layers around phase layer.
    Returns edit_success, layers_modified, side_effect_score.
    """
    r = requests.post(f"{VINDEX_BASE}/calibrated_edit", json={
        "model_id":      model_id,
        "subject":       subject,
        "relation":      relation,
        "old_obj":       old_obj,
        "new_obj":       new_obj,
        "causal_window": causal_window,
    })
    return r.json()

@mcp.tool()
async def star_spread(
    model_id:       str,
    anchor_subject: str,
    n_neighbors:    int = 5,
) -> dict:
    """
    Find semantically related slots and spread a recent edit across them.
    Example: after editing "Acer palmatum → maple", also update
    "Acer japonicum", "Acer shirasawanum" etc.
    Returns list of affected subjects and their edit scores.
    """
    r = requests.post(f"{VINDEX_BASE}/star_spread", json={
        "model_id":       model_id,
        "anchor_subject": anchor_subject,
        "n_neighbors":    n_neighbors,
    })
    return r.json()

@mcp.tool()
async def protect_relations(
    model_id:          str,
    protected_triplets: list[dict],   # [{"s": ..., "r": ..., "o": ...}]
) -> dict:
    """
    After a knowledge edit, verify that listed subject-relation-object triplets
    remain intact. Returns a pass/fail table and a side_effect_score.
    """
    r = requests.post(f"{VINDEX_BASE}/protect_relations", json={
        "model_id":          model_id,
        "protected_triplets": protected_triplets,
    })
    return r.json()

@mcp.tool()
async def layer_contribution(
    model_id: str,
    text:     str,
) -> dict:
    """
    Per-layer contribution score to the final prediction.
    Use to find which layers drive the target behavior before editing.
    """
    r = requests.post(f"{VINDEX_BASE}/layer_contribution", json={
        "model_id": model_id,
        "text":     text,
    })
    return r.json()
```

### Known bugs to fix before integration (from VINDEX PRD v1)

1. **GPU memory leak** — after repeated edits, VRAM grows unbounded. Fix: explicitly call
   `torch.cuda.empty_cache()` after each `calibrated_edit` call and detach gradient graphs.

2. **Dead-code blocks** — several helper functions in the weight-surgery path are unreachable
   after a recent refactor. Before integrating: `grep -n "def " vindex/core.py | xargs` and
   verify each function has at least one call site.

3. **Weight imbalance** — `star_spread` can over-edit related slots if `n_neighbors > 5`.
   Hard-cap at 5 in the MCP tool until the scaling formula is validated.

4. **Forward optimization** — restrict causal search to ±3 layers around the logit lens
   phase layer (already implemented as `causal_window` param above).

### UI tab: "🧠 Knowledge Editor"

```
Inputs:
  Model selector (loaded models)
  Subject text (e.g. "Acer palmatum")
  Relation (e.g. "is a type of")
  Old object (e.g. "oak")
  New object (e.g. "maple")
  [Run Logit Lens] button → displays per-layer heatmap via gr.Plot
  [Apply Edit] button → runs calibrated_edit
  [Spread] button → runs star_spread
  [Verify] button → runs protect_relations on a default triplet set
  
Outputs:
  Per-layer prediction table
  Edit success / layers modified
  Side effect score (0 = safe, 1 = dangerous)
```

---

## 8. OCR Pipeline Hook

### What it is

Your self-improving multilingual OCR pipeline (Latin, Arabic, Cyrillic) already exists and
produces output files: image + predicted_text + confidence scores. This extension hooks those
outputs directly into the workbench Field Notes system, creating a tight correction loop:

```
OCR pipeline outputs (uncertain predictions)
        ↓
Auto-created Field Notes (image + OCR text + empty correction field)
        ↓
Human reviews in UI → fills in correction
        ↓
Accepted corrections auto-tagged "use_for_training=True"
        ↓
LoRA training run on correction pairs
        ↓
Better OCR model → fewer uncertain predictions
```

This is the active learning loop your OCR pipeline was designed for but didn't yet have
a clean UI for corrections and retraining.

### Architecture

```
datasets/ocr_loader.py
  OCRPipelineLoader
    .watch(output_dir, threshold) → poll for new low-confidence outputs
    .ingest(output_dir) → batch import all outputs
    .to_field_notes(threshold) → FieldNote[] (uncertain ones only)
    .to_training_dataset() → Dataset (corrected ones only)
```

```python
# datasets/ocr_loader.py
import json, os
from pathlib import Path
from datasets.field_notes import FieldNote, FieldNoteStore
from core.events import bus, EventType, Event

class OCRPipelineLoader:
    """
    Watches a directory written by the OCR pipeline.
    Expected format per document:
      <doc_id>.json → {"image_path": ..., "predicted_text": ...,
                        "confidence": float, "script": "latin"|"arabic"|"cyrillic"}
    """
    def __init__(
        self,
        output_dir: str,
        store: FieldNoteStore,
        confidence_threshold: float = 0.85,
    ):
        self.output_dir  = Path(output_dir)
        self.store       = store
        self.threshold   = confidence_threshold

    def ingest(self, limit: int | None = None) -> int:
        """
        Read all pipeline outputs. Create Field Notes for uncertain predictions
        (confidence < threshold). Skip already-ingested docs.
        Returns number of new Field Notes created.
        """
        count = 0
        json_files = sorted(self.output_dir.glob("*.json"))
        if limit:
            json_files = json_files[:limit]

        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue

            # Skip high-confidence outputs
            if data.get("confidence", 1.0) >= self.threshold:
                continue

            note = FieldNote(
                id=f"ocr_{jf.stem}",
                model_id="ocr_pipeline",
                modality="image",
                image_path=data["image_path"],
                prompt=(
                    f"Transcribe this {data.get('script','latin')} text accurately. "
                    f"OCR predicted: '{data['predicted_text']}'"
                ),
                response=data["predicted_text"],
                correction="",      # human fills this in
                tags=[
                    f"script:{data.get('script','unknown')}",
                    f"conf:{data.get('confidence',0.0):.2f}",
                    "source:ocr_pipeline",
                ],
            )
            self.store.save(note)
            count += 1

        # Fire event
        import asyncio
        asyncio.run(bus.emit(Event(
            type=EventType.DATASET_LOADED,
            payload={"source": "ocr_pipeline", "new_notes": count}
        )))
        return count

    def watch(self, poll_interval: int = 30):
        """
        Background thread: poll output_dir every N seconds, ingest new files.
        Use in production when OCR pipeline runs continuously.
        """
        import threading, time
        seen = set()
        def _poll():
            while True:
                for jf in self.output_dir.glob("*.json"):
                    if jf.stem not in seen:
                        seen.add(jf.stem)
                        self.ingest.__wrapped__([jf])  # single-file ingest
                time.sleep(poll_interval)
        t = threading.Thread(target=_poll, daemon=True)
        t.start()

    def to_training_dataset(self, script_filter: str | None = None):
        """
        Export corrected field notes as a training dataset.
        schema: {"image_path": ..., "instruction": ..., "response": ...}
        Ready to pass to LoRATextTrainer or a vision LoRA config.
        """
        from datasets import Dataset
        query = "SELECT data FROM notes WHERE json_extract(data,'$.correction') != ''"
        if script_filter:
            query += f" AND json_extract(data,'$.tags') LIKE '%script:{script_filter}%'"

        rows = [
            json.loads(r[0])
            for r in self.store.conn.execute(query)
        ]
        training_rows = [
            {
                "image_path":  r["image_path"],
                "instruction": r["prompt"],
                "response":    r["correction"],  # human-corrected
                "script":      next(
                    (t.split(":")[1] for t in r["tags"] if t.startswith("script:")),
                    "unknown"
                ),
            }
            for r in rows
        ]
        return Dataset.from_list(training_rows)
```

### UI: OCR correction view (Field Notes tab, new subtab)

```
[OCR Pipeline Output dir: ____]  [Confidence threshold: 0.85]  [Ingest]
  
Table of uncertain predictions:
  | Image | OCR text          | Confidence | Your correction | Save |
  | [img] | "Rechung 18. Ap"  | 0.73       | [____________]  | [✓]  |
  | [img] | "Beschlußprotoko" | 0.69       | [____________]  | [✓]  |

[Export corrections as training dataset]  [Start LoRA retrain]
```

### Connection to the active learning loop

Your OCR pipeline already has:
- `abstention logic` (the "council abstains" on uncertain predictions)
- `acceptance-gated fine-tuning`
- `RAG-based post-correction`

The workbench hook provides the missing UI layer: human-in-the-loop corrections that feed
the acceptance gate. The `FieldNoteStore.to_hf_dataset()` output plugs directly into the
pipeline's `acceptance-gated fine-tuning` step.

---

## 9. MiniCPM Desk-Pet

### What it is

OpenBMB ships `MiniCPM-Desk-Pet`, a desktop companion app powered by MiniCPM5-1B, alongside
the model release (2026-05-19). Key features:
- Runs locally on Apple Silicon, NVIDIA GPU, or CPU
- LoRA persona switching — different personalities loaded as adapters
- Integrates with coding agents (Cursor, Claude Code, Codex)
- Tiny footprint (~2GB VRAM with Q4_K_M)

The workbench extension lets you train LoRA personas directly and export them to the Desk-Pet
format.

### What "persona" means here

A LoRA persona is a small adapter (rank 8–16) trained on ~100–500 conversation examples in a
specific voice or style. Examples:
- "Botanist assistant" — answers in scientific plant terminology
- "Friendly field guide" — casual, encouraging tone for beginners
- "Historical document expert" — formal, precise, citation-aware (connects to OCR pipeline)

Training data is small enough that synthetic gen (Extension 4) can produce it in minutes.

### Architecture

```
agent/desk_pet.py
  DeskPetExporter
    .train_persona(name, style_desc, n_examples) → LoRA checkpoint
    .export_to_deskpet(checkpoint_path) → deskpet_compatible.gguf
    .list_personas() → [PersonaMeta]
    .load_persona(name) → activates adapter in current session
```

```python
# agent/desk_pet.py
from dataclasses import dataclass
from pathlib import Path
import json, shutil

@dataclass
class PersonaMeta:
    name:        str
    description: str
    checkpoint:  str
    gguf_path:   str | None = None
    n_examples:  int = 0

class DeskPetExporter:
    PERSONA_DIR = Path("data/personas")

    def __init__(self, base_model_id: str = "openbmb/MiniCPM5-1B"):
        self.base_model_id = base_model_id
        self.PERSONA_DIR.mkdir(parents=True, exist_ok=True)

    def train_persona(
        self,
        name:        str,
        style_desc:  str,
        n_examples:  int = 200,
        lora_rank:   int = 8,      # small rank: personas need only ~50-100 examples
    ) -> str:
        """
        1. Use SyntheticGenerator to create conversation examples in the persona style.
        2. Fine-tune MiniCPM5-1B LoRA.
        3. Save checkpoint.
        Returns checkpoint path.
        """
        from datasets.synthetic import SyntheticGenerator
        from training.lora import LoRATextTrainer
        from models.minicpm_text import MiniCPMTextService
        import torch

        # Generate persona training data
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base_tok = AutoTokenizer.from_pretrained(self.base_model_id)

        gen_svc = MiniCPMTextService.__new__(MiniCPMTextService)
        gen_svc.model, gen_svc.tokenizer = base_model, base_tok

        synth = SyntheticGenerator(gen_svc)
        ds = synth.generate(
            topic=f"Conversation examples in the style of: {style_desc}. "
                  f"Each example: a user message and a response in that persona's voice.",
            n=n_examples,
            schema={"instruction": "user message", "response": "persona reply"},
        )

        # LoRA fine-tune
        trainer = LoRATextTrainer(cfg={
            "lora_rank":      lora_rank,
            "lora_alpha":     lora_rank * 2,
            "epochs":         2,
            "batch_size":     8,
            "grad_accum":     2,
        })
        output_dir = str(self.PERSONA_DIR / name / "checkpoint")
        trainer.train(base_model, base_tok, ds, run_name=f"persona_{name}")

        # Save metadata
        meta = PersonaMeta(
            name=name, description=style_desc,
            checkpoint=output_dir, n_examples=n_examples
        )
        (self.PERSONA_DIR / name / "meta.json").write_text(
            json.dumps(meta.__dict__, indent=2)
        )
        return output_dir

    def export_to_deskpet(self, persona_name: str) -> str:
        """
        Merge LoRA into base weights, then export as GGUF for Desk-Pet.
        Returns path to merged GGUF.
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from training.export import GGUFExporter

        meta_path = self.PERSONA_DIR / persona_name / "meta.json"
        meta      = PersonaMeta(**json.loads(meta_path.read_text()))

        # Merge LoRA into base
        base  = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, torch_dtype=torch.bfloat16
        )
        peft_model = PeftModel.from_pretrained(base, meta.checkpoint)
        merged = peft_model.merge_and_unload()

        merged_path = str(self.PERSONA_DIR / persona_name / "merged")
        merged.save_pretrained(merged_path)
        AutoTokenizer.from_pretrained(self.base_model_id).save_pretrained(merged_path)

        # Export GGUF
        exporter  = GGUFExporter()
        gguf_path = exporter.export(
            model_path=merged_path,
            output_dir=str(self.PERSONA_DIR / persona_name / "gguf"),
            quant="Q4_K_M",
            model_type="text",
        )[1]   # [0] = F16, [1] = quantized

        # Update metadata
        meta.gguf_path = gguf_path
        meta_path.write_text(json.dumps(meta.__dict__, indent=2))
        return gguf_path
```

### Usage flow

```
1. UI: "New persona" → enter name + style description
2. Synthetic gen: 200 examples of that voice → fine-tune LoRA (rank 8, ~10 min on RTX)
3. Export → Q4_K_M GGUF
4. Copy to Desk-Pet personas/ dir
5. Desk-Pet: switch persona → instant personality change
```

---

## 10. MiniCPM-o Audio Tab

### What it is

MiniCPM-o-4.5 (released 2026-05-17) is a true omnimodal model — it sees, listens, and speaks
simultaneously in real-time. It supports proactive interactions (proactive reminding) and
real-time conversation with both visual and audio input.

This extension adds a new Gradio tab with a microphone + camera (or image) interface, streaming
audio output, and real-time MiniCPM-o inference.

### Architecture

```
ui/audio_tab.py
  OmnimodalTab
    .build() → gr.Column with audio+image inputs and streaming output
    
models/minicpm_omni.py
  MiniCPMOmniService
    .stream_chat(audio_bytes, image=None, text=None) → Generator[str]
    .speak(text) → bytes  (TTS for audio output)
```

```python
# models/minicpm_omni.py
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

class MiniCPMOmniService:
    """
    MiniCPM-o-4.5: omnimodal service.
    Handles text + image + audio simultaneously.
    """
    MODEL_ID = "openbmb/MiniCPM-o-4.5"

    def __init__(self, cfg: dict):
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.thinking = cfg.get("thinking_mode", False)

    def chat(
        self,
        text:        str | None = None,
        image=None,             # PIL Image
        audio:       np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> str:
        """
        Full omnimodal chat: pass any combination of text, image, audio.
        MiniCPM-o-4.5 handles them natively.
        """
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if audio is not None:
            content.append({
                "type":        "audio",
                "audio":       audio,
                "sample_rate": sample_rate,
            })
        if text:
            content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        inputs   = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": self.thinking},
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
```

```python
# ui/audio_tab.py
import gradio as gr
import numpy as np

def build_audio_tab(model_registry):
    with gr.Column():
        gr.Markdown("### 🎙️ Omnimodal — MiniCPM-o-4.5")
        gr.Markdown("Speak, show an image, or type — all at once.")

        with gr.Row():
            audio_in  = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Microphone input",
                streaming=True,
            )
            image_in  = gr.Image(
                type="pil",
                label="Optional image",
            )

        text_in   = gr.Textbox(label="Optional text", placeholder="Or type here...")
        submit    = gr.Button("Send")
        output    = gr.Textbox(label="Response", lines=8)
        audio_out = gr.Audio(label="Spoken response (TTS)", autoplay=True)

        def respond(audio_data, image, text):
            svc = model_registry.get("minicpm_o45")
            audio_arr = None
            if audio_data is not None:
                sr, arr = audio_data
                audio_arr = arr.astype(np.float32) / 32768.0
            response = svc.chat(text=text, image=image, audio=audio_arr)
            # Optional: TTS for spoken output
            # spoken = tts(response)
            return response, None   # None = no audio out yet

        submit.click(respond, [audio_in, image_in, text_in], [output, audio_out])
```

### Real-time streaming version (advanced)

For true real-time conversation (proactive reminding, interrupt detection):

```python
# streaming audio inference using gradio's streaming audio + SSE
demo = gr.Interface(
    fn=stream_omni_response,
    inputs=[
        gr.Audio(streaming=True, sources=["microphone"]),
        gr.Image(type="pil"),
    ],
    outputs=gr.Textbox(),
    live=True,
)
```

### TTS for audio output

MiniCPM-o-4.5 has its own audio generation capability — check the model card for the
`generate_audio` method. Fallback: use `kokoro-82M` (Apache 2.0, 82M, runs locally) for TTS.

```python
# Kokoro TTS fallback (lightweight, local)
from kokoro import KPipeline
tts_pipe = KPipeline(lang_code="en-us")
audio, sr = tts_pipe(response_text, voice="af_heart")
```

---

## 11. Cross-Extension Wiring

Most extensions are independent, but several combinations unlock powerful compound workflows:

### OCR → VINDEX

Low-confidence OCR outputs → Field Notes → human corrections → LoRA retrain. But additionally:
if the OCR model consistently misreads a specific character class, use VINDEX's `logit_lens` to
identify which layer is responsible, then `calibrated_edit` to target that slot directly — a
faster fix than a full retraining cycle.

### Synthetic Gen → Reward Model → DPO

```
SyntheticGenerator.generate(topic, n=1000)
    → RewardEvaluator.create_dpo_pairs(prompts, generator, n=4)
    → DPO training via TRL DPOTrainer
    → Trackio logs win rate
    → VINDEX verify alignment not broken
```

### Paper Agent → Desk-Pet Persona

```
PaperAgent reads: "Persona-based dialogue systems for domain experts"
    → Implements: persona training data format
    → DeskPetExporter.train_persona("expert_botanist", "...")
    → Export GGUF → load in Desk-Pet
```

### HF Spaces + vLLM + Trackio

```
SpacesDeployer.deploy(hardware="a10")   # production GPU
    → app.py runs VLLMRunner on A10G
    → All requests logged via Trackio with space_id=deployed_space
    → Dashboard visible to collaborators at trackio Space URL
```

### Full active-learning loop (all extensions combined)

```
MiniCPM-o Audio Tab: user speaks + shows image of plant
        ↓
OmnimodalService: identify species (low confidence)
        ↓
OCRPipelineLoader: auto-create Field Note (uncertain prediction)
        ↓
Human: corrects species name in UI
        ↓
SyntheticGenerator: augment with 50 similar examples
        ↓
RewardEvaluator: filter synthetic examples
        ↓
LoRATextTrainer + TRL + Trackio: fine-tune
        ↓
VINDEX: verify the target species slot was corrected
        ↓
DeskPetExporter: export updated persona
        ↓
SpacesDeployer: push updated app to HF Spaces
```

---

*Extensions spec v1.0 — Christof Kaller / ki-fusion-labs.de — 2026-06-05*