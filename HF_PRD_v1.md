# OpenBMB Local AI Workbench — Technical PRD v2.0
> Revised: 2026-06-05 | Corrections from online research embedded inline

---

## Table of Contents

1. [Purpose & Design Philosophy](#1-purpose--design-philosophy)
2. [Design Principles (Template Architecture)](#2-design-principles-template-architecture)
3. [System Architecture](#3-system-architecture)
4. [Model Registry](#4-model-registry)
5. [Inference Stack — Five Modes](#5-inference-stack--five-modes)
6. [Tracking: Trackio (Corrected API)](#6-tracking-trackio-corrected-api)
7. [MCP Layer (Three Integration Paths)](#7-mcp-layer-three-integration-paths)
8. [Training Pipeline](#8-training-pipeline)
9. [Export & Quantization](#9-export--quantization)
10. [Agent Mode (ml-intern Inspired)](#10-agent-mode-ml-intern-inspired)
11. [UI — Tab Specification](#11-ui--tab-specification)
12. [Field Notes & Correction Loop](#12-field-notes--correction-loop)
13. [Directory Structure](#13-directory-structure)
14. [Configuration Schema](#14-configuration-schema)
15. [Dependencies](#15-dependencies)
16. [Hackathon Demo Flow](#16-hackathon-demo-flow)
17. [Corrections from PRD v1](#17-corrections-from-prd-v1)
18. [Roadmap & Extension Points](#18-roadmap--extension-points)

---

## 1. Purpose & Design Philosophy

**One-line:** A modular AI experimentation platform for the OpenBMB model family that covers
dataset ingestion → LoRA training → evaluation → GGUF export → llama.cpp/SGLang deployment →
multimodal inference → MCP tool exposure → trace sharing — in a single Gradio app.

**Broader framing:** The platform is deliberately designed as a *template*. The `config/models.yaml`
and `config/training.yaml` files drive almost all model-specific behaviour. Replacing the OpenBMB
section with any other model family (Qwen, Phi, Gemma …) should require only config changes and
a matching service class, never core rewrites.

**Why OpenBMB?**
- MiniCPM5-1B (2026-05-19): SOTA 1B on-device, 128K context, LlamaForCausalLM — no custom kernels
- MiniCPM4.1-8B (2025-09): sparse InfLLM-v2 attention, ~7× long-context speedup vs Qwen3-8B
- MiniCPM-V-4.6 (2026-05-11): 1.3B VLM, SigLIP2-400M + Qwen3.5-0.8B, 262K ctx, mixed 4×/16× token compression
- MiniCPM-V-4.6-Thinking: same architecture + long-CoT for multimodal reasoning
- MiniCPM-o 4.5 (2026-05-17): omnimodal (vision + audio), real-time conversation
- Apache 2.0 across the board; vLLM / SGLang / llama.cpp / Ollama native support

---

## 2. Design Principles (Template Architecture)

| Principle | Implementation |
|-----------|----------------|
| Config-driven | `models.yaml` + `training.yaml` define all models; no model names hardcoded in logic |
| Event-driven | `core/events.py` — typed event bus; all cross-module comms via events |
| Registry pattern | `core/registry.py` — all services register by name; swap without restart |
| Separation of concerns | `models/` = loading; `training/` = train/eval; `ui/` = Gradio; `mcp/` = protocol |
| Plugin tools | Any Python function decorated `@mcp.tool()` is auto-exposed; no wiring needed |
| Async-first | FastAPI backend, Gradio queuing, non-blocking inference |
| Observability | Every significant action fires a Trackio log + event |
| Template portability | Replace `config/models.yaml` → different model family; keep all else |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OpenBMB Local AI Workbench                           │
├───────────────────────────┬─────────────────────────────────────────────────┤
│        Gradio UI Layer    │          Backend Service Layer                  │
│  ┌─────────────────────┐  │  ┌─────────────────────────────────────────┐   │
│  │  gr.Blocks / Tabs   │  │  │  gradio.Server (FastAPI-based)          │   │
│  │  chat_tab.py        │  │  │  @app.api()   — Gradio-queued endpoints │   │
│  │  vision_tab.py      │  │  │  @app.mcp.tool() — MCP registration     │   │
│  │  train_tab.py       │  │  │  @app.get("/") — static/custom frontend │   │
│  │  dataset_tab.py     │  │  └─────────────────────────────────────────┘   │
│  │  export_tab.py      │  │                                                 │
│  │  traces_tab.py      │  │  ┌─────────────────────────────────────────┐   │
│  │  agent_tab.py       │  │  │  FastMCP server (standalone / bridged)  │   │
│  └─────────────────────┘  │  │  mcp = FastMCP("OpenBMBWorkbench")      │   │
│                           │  │  Registered tools (see §7)              │   │
│  SSE MCP endpoint:        │  └─────────────────────────────────────────┘   │
│  /gradio_api/mcp/sse      │                                                 │
└───────────────────────────┴─────────────────────────────────────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐        ┌────────────────────────┐
│   Model Services    │        │   Tracking & Storage   │
│  minicpm_text.py    │        │  trackio (local/Space) │
│  minicpm_vision.py  │        │  field_notes/ (SQLite) │
│  llama_cpp_runner.py│        │  exports/ (GGUF files) │
│  sglang_runner.py   │        │  data/ (HF datasets)   │
│  ollama_runner.py   │        └────────────────────────┘
└─────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│                   Inference Backends                      │
│  Transformers  │  llama.cpp  │  SGLang  │  vLLM  │ Ollama│
└───────────────────────────────────────────────────────────┘
```

### 3.1 Event Bus (core/events.py)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import asyncio

class EventType(str, Enum):
    DATASET_LOADED      = "dataset_loaded"
    TRAINING_STARTED    = "training_started"
    TRAINING_STEP       = "training_step"
    TRAINING_FINISHED   = "training_finished"
    EVAL_FINISHED       = "eval_finished"
    INFERENCE_REQUEST   = "inference_request"
    INFERENCE_RESPONSE  = "inference_response"
    TOOL_CALL           = "tool_call"
    MODEL_LOADED        = "model_loaded"
    MODEL_SWITCHED      = "model_switched"
    EXPORT_STARTED      = "export_started"
    EXPORT_FINISHED     = "export_finished"
    FIELD_NOTE_SAVED    = "field_note_saved"
    AGENT_STEP          = "agent_step"

@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

class EventBus:
    _handlers: dict[EventType, list[Callable]] = {}

    def on(self, event_type: EventType):
        """Decorator to register an event handler."""
        def decorator(fn: Callable):
            self._handlers.setdefault(event_type, []).append(fn)
            return fn
        return decorator

    async def emit(self, event: Event):
        for handler in self._handlers.get(event.type, []):
            await handler(event)

bus = EventBus()
```

### 3.2 Service Registry (core/registry.py)

```python
from typing import TypeVar, Generic, Type

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self):
        self._services: dict[str, T] = {}

    def register(self, name: str, service: T):
        self._services[name] = service

    def get(self, name: str) -> T:
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered. Available: {list(self._services)}")
        return self._services[name]

    def list(self) -> list[str]:
        return list(self._services.keys())

# Singleton registries
model_registry   = Registry()   # model services
dataset_registry = Registry()   # dataset loaders
tool_registry    = Registry()   # MCP tools
```

---

## 4. Model Registry

### 4.1 Text Models

| Config ID | HF ID | Architecture | Context | Notes |
|-----------|-------|-------------|---------|-------|
| `minicpm5_1b` | `openbmb/MiniCPM5-1B` | LlamaForCausalLM | 128K | Standard; no custom kernels; XML tool calls; SGLang `minicpm5` parser |
| `minicpm5_1b_thinking` | `openbmb/MiniCPM5-1B-Thinking` | LlamaForCausalLM | 128K | CoT mode; `chat_template_kwargs={"enable_thinking": True}` |
| `minicpm41_8b` | `openbmb/MiniCPM4.1-8B` | Sparse InfLLM-v2 | 128K (YaRN) | `--trust-remote-code` required; 7× faster long-ctx vs Qwen3-8B |

### 4.2 Vision Models

| Config ID | HF ID | Backbone | Context | Notes |
|-----------|-------|---------|---------|-------|
| `minicpm_v46` | `openbmb/MiniCPM-V-4.6` | SigLIP2-400M + Qwen3.5-0.8B | 262K | `transformers>=5.7.0` + `torchcodec`; mixed 4×/16× compression |
| `minicpm_v46_thinking` | `openbmb/MiniCPM-V-4.6-Thinking` | same + CoT head | 262K | `enable_thinking=True` in chat_template_kwargs |
| `minicpm_o45` | `openbmb/MiniCPM-o-4.5` | Omnimodal | - | Vision + Audio; real-time conversation; see §4.4 |

### 4.3 Model Architecture Notes (critical for correct code)

**MiniCPM5-1B** — standard Llama, loads with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", trust_remote_code=True)
```

**MiniCPM-V-4.6** — vision-language, loads with:
```python
# ⚠️ CORRECTION from v1: NOT AutoModelForCausalLM — use AutoModelForImageTextToText
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-4.6", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "openbmb/MiniCPM-V-4.6",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Chat format (vision):
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},   # PIL Image
            {"type": "text",  "text": prompt}
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    chat_template_kwargs={"enable_thinking": False}   # True for Thinking variant
).to(model.device)
```

**MiniCPM4.1-8B** — requires `trust_remote_code=True` for sparse attention kernel.

### 4.4 models.yaml Schema

```yaml
# config/models.yaml
# ─── Template: add any HF model here ──────────────────────────────────────
models:
  minicpm5_1b:
    hf_id:          openbmb/MiniCPM5-1B
    type:           text                        # text | vision | omnimodal
    architecture:   llama                       # llama | minicpm41 | minicpm_v
    context_length: 131072
    dtype:          bfloat16
    trust_remote_code: false                    # LlamaForCausalLM — no fork
    thinking_mode:  false                       # set true for Thinking checkpoints
    tool_call_parser: minicpm5                  # for SGLang serving
    gguf:
      available: true
      repo: openbmb/MiniCPM5-1B-GGUF
      default_quant: Q4_K_M

  minicpm41_8b:
    hf_id:          openbmb/MiniCPM4.1-8B
    type:           text
    architecture:   minicpm41                   # sparse InfLLM-v2
    context_length: 131072
    dtype:          bfloat16
    trust_remote_code: true
    thinking_mode:  false
    gguf:
      available: true
      repo: openbmb/MiniCPM4.1-8B-GGUF
      default_quant: Q4_K_M

  minicpm_v46:
    hf_id:          openbmb/MiniCPM-V-4.6
    type:           vision
    architecture:   minicpm_v
    context_length: 262144
    dtype:          bfloat16
    trust_remote_code: true
    thinking_mode:  false
    vision_encoder: siglip2_400m
    token_compression: "4x_16x"               # mixed compression
    gguf:
      available: true
      repo: openbmb/MiniCPM-V-4.6-gguf        # official repo — no surgery script needed
      main_file:  MiniCPM-V-4.6-Q4_K_M.gguf
      mmproj:     mmproj-MiniCPM-V-4.6-F16.gguf
      default_quant: Q4_K_M

  minicpm_v46_thinking:
    hf_id:          openbmb/MiniCPM-V-4.6-Thinking
    type:           vision
    architecture:   minicpm_v
    context_length: 262144
    dtype:          bfloat16
    trust_remote_code: true
    thinking_mode:  true
    gguf:
      available: true
      repo: openbmb/MiniCPM-V-4.6-Thinking-gguf
      default_quant: Q4_K_M

  minicpm_o45:
    hf_id:          openbmb/MiniCPM-o-4.5
    type:           omnimodal
    architecture:   minicpm_o
    context_length: 131072
    dtype:          bfloat16
    trust_remote_code: true
    thinking_mode:  false
    note: "Vision + Audio; real-time conversation mode"
```

---

## 5. Inference Stack — Five Modes

### Mode A: Transformers (train + eval + fine-tuning)

Used for: LoRA training, evaluation, PEFT
Load text: `AutoModelForCausalLM`
Load vision: `AutoModelForImageTextToText`  ← corrected from v1

```python
# models/minicpm_text.py
class MiniCPMTextService:
    def __init__(self, model_id: str, cfg: dict):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=cfg.get("trust_remote_code", False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

```python
# models/minicpm_vision.py  ← new dedicated service
class MiniCPMVisionService:
    def __init__(self, model_id: str, cfg: dict):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=cfg.get("trust_remote_code", True)
        )
        self.thinking = cfg.get("thinking_mode", False)

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": self.thinking}
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response
```

### Mode B: llama.cpp (local deployment, CPU/GPU offload)

> ⚠️ CORRECTION from v1: For MiniCPM-V 4.6, official GGUFs are already published.
> Download from `openbmb/MiniCPM-V-4.6-gguf` — no need to run `convert_hf_to_gguf.py` yourself.
> The `minicpmv-surgery.py` script was for the older MiniCPM-o-2.6; **do not use it for V4.6**.

**Text models** → `llama-server` (standard OpenAI-compatible endpoint)
**Vision models** → `llama-server --mmproj <mmproj.gguf>` (multimodal server)

```python
# models/llama_cpp_runner.py
import subprocess, requests, time

class LlamaCppDeployment:
    def __init__(self, cfg: dict):
        self.model_path  = cfg["model_path"]
        self.mmproj_path = cfg.get("mmproj_path")      # None for text models
        self.port        = cfg.get("port", 8080)
        self._proc       = None

    def start(self):
        cmd = [
            "llama-server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--ctx-size", "8192",
            "--n-gpu-layers", "-1",       # full GPU offload if available
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]   # enables vision
        self._proc = subprocess.Popen(cmd)
        self._wait_ready()

    def _wait_ready(self, timeout: int = 30):
        for _ in range(timeout):
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not start in time")

    def chat(self, messages: list[dict]) -> str:
        r = requests.post(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json={"messages": messages, "max_tokens": 512}
        )
        return r.json()["choices"][0]["message"]["content"]

    def stop(self):
        if self._proc:
            self._proc.terminate()
```

**GGUF Export** (only needed if building custom quants):
```bash
# For MiniCPM-V 4.6 — convert_hf_to_gguf.py (NOT legacy surgery scripts)
python convert_hf_to_gguf.py openbmb/MiniCPM-V-4.6 --outtype f16
# Output: MiniCPM-V-4.6-F16.gguf  +  mmproj-MiniCPM-V-4.6-F16.gguf

# Quantize
./llama-quantize MiniCPM-V-4.6-F16.gguf MiniCPM-V-4.6-Q4_K_M.gguf Q4_K_M
```

Supported quants: `Q4_K_M`, `Q5_K_M`, `Q8_0`, `Q2_K`, `F16`

### Mode C: SGLang (tool-use, function calling, recommended for MiniCPM5)

MiniCPM5-1B emits XML-style tool calls. SGLang has a native `minicpm5` parser:

```bash
python -m sglang.launch_server \
  --model-path openbmb/MiniCPM5-1B \
  --port 30000 \
  --tool-call-parser minicpm5     # or: --tool-call-parser auto
```

For MiniCPM-V-4.6 vision:
```bash
python -m sglang.launch_server \
  --model-path openbmb/MiniCPM-V-4.6-Thinking \
  --port 30000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

### Mode D: vLLM (batch serving, production)

```bash
vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code
vllm serve openbmb/MiniCPM-V-4.6 --trust-remote-code
```

### Mode E: Ollama (zero-config, consumer hardware)

```bash
ollama run openbmb/minicpm-v4.6    # vision + text
ollama run openbmb/minicpm5-1b     # text only
```

### Mode Selection Matrix

| Use case | Recommended mode |
|----------|-----------------|
| LoRA training | A (Transformers) |
| Evaluation | A (Transformers) |
| Local chat (fast) | B (llama.cpp) |
| Tool/function calling | C (SGLang) |
| Production batch | D (vLLM) |
| Consumer demo | E (Ollama) |
| Vision inference local | B (llama-server --mmproj) |
| Vision fine-tuning | A (SWIFT / LLaMA-Factory) |

---

## 6. Tracking: Trackio (Corrected API)

> ⚠️ CRITICAL CORRECTION from v1: `trackio.trace()` **does not exist**.
> There is no context-manager trace API in Trackio. The v1 PRD example was wrong.
> Correct API: `init()` / `log()` / `finish()` / `alert()` / `Table()` / `Markdown()`

Trackio is a wandb drop-in replacement, local-first, free, Gradio dashboard, HF Spaces sync.

```python
# tracking/trackio_service.py
import trackio
import time
from core.events import bus, EventType, Event

class TrackioService:
    def __init__(self, project: str, space_id: str | None = None):
        self.project  = project
        self.space_id = space_id
        self._active  = False
        self._register_handlers()

    def _register_handlers(self):
        @bus.on(EventType.TRAINING_STARTED)
        async def on_training_started(event: Event):
            trackio.init(
                project=self.project,
                space_id=self.space_id,
                run_name=event.payload.get("run_name", "training"),
            )
            self._active = True

        @bus.on(EventType.TRAINING_STEP)
        async def on_step(event: Event):
            if self._active:
                trackio.log(event.payload)

        @bus.on(EventType.TRAINING_FINISHED)
        async def on_done(event: Event):
            if self._active:
                trackio.log({"status": "finished", **event.payload})
                trackio.finish()
                self._active = False

        @bus.on(EventType.EVAL_FINISHED)
        async def on_eval(event: Event):
            trackio.init(
                project=self.project,
                space_id=self.space_id,
                run_name=event.payload.get("eval_run_name", "eval"),
            )
            trackio.log(event.payload)
            trackio.finish()

        @bus.on(EventType.INFERENCE_RESPONSE)
        async def on_inference(event: Event):
            # Log inference latency and token count
            trackio.init(
                project=self.project,
                space_id=self.space_id,
                run_name="inference",
            )
            trackio.log({
                "model":          event.payload.get("model"),
                "latency_ms":     event.payload.get("latency_ms"),
                "tokens_out":     event.payload.get("tokens_out"),
                "mode":           event.payload.get("mode", "text"),
            })
            trackio.finish()

        @bus.on(EventType.TOOL_CALL)
        async def on_tool(event: Event):
            trackio.alert(
                title=f"Tool call: {event.payload.get('tool_name')}",
                level=trackio.AlertLevel.INFO,
            )
```

**TRL LoRA integration** (recommended for training):
```python
from trl import SFTConfig
training_args = SFTConfig(
    output_dir="./checkpoints",
    report_to="trackio",   # ← one-liner Trackio integration via TRL
    run_name="lora_minicpm5_1b",
)
```

**SQL query via CLI** (LLM-friendly):
```bash
trackio query project --project workbench --sql "SELECT run_name, loss FROM metrics ORDER BY loss LIMIT 5"
```

---

## 7. MCP Layer (Three Integration Paths)

### Path 1: Gradio Blocks + `launch(mcp_server=True)` ← simplest

```python
# app.py (simple path)
import gradio as gr
from mcp.server.fastmcp import FastMCP

demo = gr.Blocks()
# ... tabs ...
demo.launch(
    mcp_server=True,    # exposes /gradio_api/mcp/sse
    server_port=7860,
)
```

Every `gr.Interface` function and `gr.api()` endpoint becomes an MCP tool automatically.
Docstrings become tool descriptions. Type hints become parameter schemas.

### Path 2: `gradio.Server` (custom frontend + MCP)

> New API published 2026-04-01. Extends FastAPI. Better for custom dashboards.

```python
# app.py (Server path)
from gradio import Server
from fastapi.responses import HTMLResponse

app = Server()

@app.api(name="run_inference")             # Gradio-queued + MCP-compatible
def run_inference(prompt: str, model_id: str) -> str:
    """Run text inference on a MiniCPM text model."""
    svc = model_registry.get(model_id)
    return svc.generate(prompt)

@app.mcp.tool()                            # explicit MCP tool registration
async def export_gguf(model_id: str, quant: str = "Q4_K_M") -> str:
    """Export a Transformers model to GGUF format with specified quantization."""
    return await export_service.run(model_id, quant)

@app.get("/", response_class=HTMLResponse) # custom frontend
async def homepage():
    return open("ui/index.html").read()

app.launch(show_error=True)
```

### Path 3: FastMCP standalone + Gradio client bridge

Use when you need stateful MCP tools, or when running multiple Gradio apps and want to save memory:

```python
# mcp/server.py
from mcp.server.fastmcp import FastMCP
from gradio_client import Client

mcp = FastMCP("OpenBMBWorkbench")
_clients: dict[str, Client] = {}

def _gradio_client(space_id: str) -> Client:
    if space_id not in _clients:
        _clients[space_id] = Client(space_id)
    return _clients[space_id]

@mcp.tool()
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    import math
    return str(eval(expression, {"__builtins__": {}}, vars(math)))

@mcp.tool()
async def dataset_stats(dataset_name: str, split: str = "train") -> dict:
    """Return basic statistics for a HuggingFace dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    return {
        "rows":    len(ds),
        "columns": ds.column_names,
        "features": str(ds.features),
    }

@mcp.tool()
async def search_dataset(query: str, max_results: int = 5) -> list[dict]:
    """Search HuggingFace Hub for datasets matching a query."""
    from huggingface_hub import list_datasets
    results = list(list_datasets(search=query, limit=max_results))
    return [{"id": d.id, "downloads": d.downloads, "tags": d.tags} for d in results]

@mcp.tool()
async def start_training(
    model_id: str,
    dataset_name: str,
    lora_rank: int = 16,
    epochs: int = 3,
) -> dict:
    """Start a LoRA fine-tuning run on the specified model and dataset."""
    # Dispatch to training service via event bus
    await bus.emit(Event(
        type=EventType.TRAINING_STARTED,
        payload={"model_id": model_id, "dataset": dataset_name,
                 "lora_rank": lora_rank, "epochs": epochs}
    ))
    return {"status": "started", "model_id": model_id}

@mcp.tool()
async def evaluate_model(model_id: str, dataset_name: str) -> dict:
    """Evaluate a model on a dataset and return metrics."""
    svc = model_registry.get(model_id)
    return await eval_service.run(svc, dataset_name)

@mcp.tool()
async def export_gguf(model_id: str, quant: str = "Q4_K_M") -> str:
    """Convert a fine-tuned model to GGUF format."""
    return await export_service.run(model_id, quant)

@mcp.tool()
async def vision_chat(image_path: str, prompt: str, model_id: str = "minicpm_v46") -> str:
    """Run multimodal inference on an image with a prompt."""
    svc: MiniCPMVisionService = model_registry.get(model_id)
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": prompt},
    ]}]
    return svc.chat(messages)

if __name__ == "__main__":
    mcp.run(transport="sse", port=8081)
```

MCP SSE endpoint: `http://localhost:8081/sse`
Gradio MCP SSE: `http://localhost:7860/gradio_api/mcp/sse`

---

## 8. Training Pipeline

### 8.1 LoRA — Text Models (PEFT)

```python
# training/lora.py
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

class LoRATextTrainer:
    DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self, model, tokenizer, dataset, run_name: str):
        lora_cfg = LoraConfig(
            r=self.cfg.get("lora_rank", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules", self.DEFAULT_TARGET_MODULES),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=f"./checkpoints/{run_name}",
            num_train_epochs=self.cfg.get("epochs", 3),
            per_device_train_batch_size=self.cfg.get("batch_size", 4),
            gradient_accumulation_steps=self.cfg.get("grad_accum", 4),
            learning_rate=self.cfg.get("lr", 2e-4),
            fp16=False,
            bf16=True,
            report_to="trackio",         # ← Trackio via TRL
            run_name=run_name,
            save_steps=100,
            logging_steps=10,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        return model
```

### 8.2 LoRA — Vision Models (SWIFT / LLaMA-Factory recommended)

> OpenBMB recommends SWIFT or LLaMA-Factory for MiniCPM-V fine-tuning.
> Vision tower is frozen; only language backbone layers are adapted.

```python
# training/lora_vision.py
class LoRAVisionConfig:
    """
    For MiniCPM-V 4.6 LoRA:
    - Freeze vision encoder (SigLIP2-400M) + projector
    - Apply LoRA only to Qwen3.5-0.8B language layers
    - Use SWIFT CLI or LLaMA-Factory for full pipeline
    """
    LANGUAGE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    FROZEN_PREFIXES = ["vpm", "resampler", "vision_model"]  # freeze vision

    @staticmethod
    def freeze_vision(model):
        for name, param in model.named_parameters():
            if any(name.startswith(pfx) for pfx in LoRAVisionConfig.FROZEN_PREFIXES):
                param.requires_grad = False
        return model
```

**SWIFT CLI approach** (recommended for MiniCPM-V 4.6):
```bash
swift sft \
  --model openbmb/MiniCPM-V-4.6 \
  --dataset <your-dataset> \
  --lora_rank 16 \
  --freeze_vit true \
  --output_dir ./checkpoints/minicpm_v46_lora
```

### 8.3 HF Jobs Offload (ml-intern pattern)

For long training runs, offload to HF Jobs (free `cpu-basic`, paid `gpu-a100`):

```python
# training/hf_jobs.py
from huggingface_hub import HfApi

class HFJobsTrainer:
    def submit(self, script_path: str, hardware: str = "gpu-a100") -> str:
        api = HfApi()
        job = api.create_job(
            repo_id=f"{api.whoami()['name']}/ml-workbench-jobs",
            script=script_path,
            hardware=hardware,
        )
        return job.job_id
```

### 8.4 Evaluation

```python
# training/evaluation.py
class Evaluator:
    def run(self, model, tokenizer, dataset, metrics=("perplexity", "accuracy")) -> dict:
        results = {}
        if "perplexity" in metrics:
            results["perplexity"] = self._perplexity(model, tokenizer, dataset)
        if "accuracy" in metrics:
            results["accuracy"] = self._accuracy(model, tokenizer, dataset)
        return results
```

---

## 9. Export & Quantization

```python
# training/export.py
import subprocess
from pathlib import Path

class GGUFExporter:
    SUPPORTED_QUANTS = ["F16", "Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K"]

    def export(
        self,
        model_path: str,
        output_dir: str,
        quant: str = "Q4_K_M",
        model_type: str = "text",    # "text" | "vision"
    ) -> list[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert HF → GGUF F16
        # For MiniCPM-V 4.6: convert_hf_to_gguf.py produces main gguf + mmproj
        # For text models: same script, no mmproj
        f16_path = output_dir / "model-f16.gguf"
        subprocess.run([
            "python", "convert_hf_to_gguf.py",
            model_path,
            "--outtype", "f16",
            "--outfile", str(f16_path),
        ], check=True)

        outputs = [str(f16_path)]

        # Step 2: Quantize
        if quant != "F16":
            quant_path = output_dir / f"model-{quant}.gguf"
            subprocess.run([
                "llama-quantize",
                str(f16_path),
                str(quant_path),
                quant,
            ], check=True)
            outputs.append(str(quant_path))

        return outputs
```

> Note: For MiniCPM-V 4.6, official GGUF files are already published at
> `openbmb/MiniCPM-V-4.6-gguf`. Download them directly unless you need custom quants.

---

## 10. Agent Mode (ml-intern Inspired)

The `agent/` module implements a lightweight research-plan-implement loop inspired by HF's
ml-intern (April 2026). This gives the workbench autonomous ML experimentation capabilities.

```
ml-intern architecture (reference):
  Research  → browse arXiv/HF Papers, read methodology, traverse citations
  Plan      → break task down, verify resources (models, datasets, hardware)
  Implement → execute scripts on HF Jobs / local GPU
  Trace     → auto-upload sessions as JSONL to HF Dataset
```

```python
# agent/loop.py
from smolagents import CodeAgent, HfApiModel
from huggingface_hub import HfApi

class WorkbenchAgent:
    """
    Lightweight agent loop for autonomous experimentation.
    Uses smolagents + HF ecosystem + registered MCP tools.
    """
    def __init__(self, model_id: str = "claude-sonnet-4-6"):
        self.model = HfApiModel(model_id)
        self.api   = HfApi()
        self._session_log = []

    def run(self, prompt: str, max_steps: int = 20) -> str:
        agent = CodeAgent(
            tools=self._get_tools(),
            model=self.model,
            max_steps=max_steps,
        )
        result = agent.run(prompt)
        self._save_session(prompt, result)
        return result

    def _save_session(self, prompt: str, result: str):
        import json, datetime
        session = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "prompt": prompt,
            "result": result,
            "log": self._session_log,
        }
        # Auto-upload to private HF dataset (ml-intern pattern)
        # Dataset: {username}/workbench-sessions
        pass

    def _get_tools(self):
        # Expose MCP tools as smolagents tools
        from mcp.server.fastmcp import FastMCP
        # ...
        return []
```

---

## 11. UI — Tab Specification

Using `gr.Blocks` with tabs. Optionally swap to `gradio.Server` for custom frontend.

```python
# app.py
import gradio as gr
from ui.chat_tab    import build_chat_tab
from ui.vision_tab  import build_vision_tab
from ui.train_tab   import build_train_tab
from ui.dataset_tab import build_dataset_tab
from ui.export_tab  import build_export_tab
from ui.traces_tab  import build_traces_tab
from ui.agent_tab   import build_agent_tab
from ui.notes_tab   import build_notes_tab

with gr.Blocks(title="OpenBMB Workbench", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# 🔬 OpenBMB Local AI Workbench")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):       build_chat_tab()
        with gr.Tab("👁️ Vision"):     build_vision_tab()
        with gr.Tab("📊 Dataset"):    build_dataset_tab()
        with gr.Tab("🏋️ Train"):      build_train_tab()
        with gr.Tab("📦 Export"):     build_export_tab()
        with gr.Tab("📈 Traces"):     build_traces_tab()
        with gr.Tab("🤖 Agent"):      build_agent_tab()
        with gr.Tab("📝 Field Notes"):build_notes_tab()

demo.launch(
    mcp_server=True,    # SSE at /gradio_api/mcp/sse
    server_port=7860,
    share=False,
)
```

### Tab Summary

| Tab | Purpose | Key components |
|-----|---------|----------------|
| Chat | Text inference on all text models | Model selector, streaming output, system prompt |
| Vision | Image + text inference on V-4.6/Thinking | Image upload, thinking toggle, bounding box overlay |
| Dataset | HF Hub browser + local CSV/JSON loader | Search bar, split preview, schema inspector |
| Train | LoRA training launcher | Config form, live loss chart (Trackio), checkpoint browser |
| Export | GGUF export + quantization | Model selector, quant dropdown, file download |
| Traces | Trackio run browser + comparison | Run table, metric plots, SQL query box |
| Agent | ml-intern style agent loop | Prompt, step log, paper browser, HF Jobs monitor |
| Field Notes | Correction capture → retrain trigger | Image+prompt+response+correction form, JSONL export |

### Vision Tab — key detail

```python
# ui/vision_tab.py
def build_vision_tab():
    with gr.Column():
        model_dd = gr.Dropdown(
            choices=["minicpm_v46", "minicpm_v46_thinking", "minicpm_o45"],
            value="minicpm_v46",
            label="Vision Model"
        )
        thinking_cb = gr.Checkbox(label="Enable Thinking Mode", value=False)
        image_in    = gr.Image(type="pil", label="Upload Image")
        video_in    = gr.Video(label="Upload Video (MiniCPM-V-4.6 supports video)")
        prompt_in   = gr.Textbox(label="Prompt")
        submit_btn  = gr.Button("Run")
        output_txt  = gr.Textbox(label="Response", lines=10)

        def infer(model_id, thinking, image, video, prompt):
            svc = model_registry.get(model_id)
            content = []
            if image:
                content.append({"type": "image", "image": image})
            if video:
                content.append({"type": "video", "video": video})
            content.append({"type": "text", "text": prompt})
            msgs = [{"role": "user", "content": content}]
            svc.thinking = thinking
            return svc.chat(msgs)

        submit_btn.click(infer, [model_dd, thinking_cb, image_in, video_in, prompt_in], output_txt)
```

---

## 12. Field Notes & Correction Loop

```python
# datasets/field_notes.py
import sqlite3, json
from dataclasses import dataclass, field, asdict
from datetime import datetime

@dataclass
class FieldNote:
    id:          str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_id:    str = ""
    modality:    str = "text"      # "text" | "image" | "video" | "multimodal"
    image_path:  str | None = None
    video_path:  str | None = None
    prompt:      str = ""
    response:    str = ""
    correction:  str = ""          # human-corrected output
    tags:        list = field(default_factory=list)
    used_in_train: bool = False

class FieldNoteStore:
    def __init__(self, db_path: str = "data/field_notes.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, note: FieldNote):
        self.conn.execute(
            "INSERT OR REPLACE INTO notes (id, data) VALUES (?, ?)",
            (note.id, json.dumps(asdict(note)))
        )
        self.conn.commit()

    def to_hf_dataset(self, output_path: str):
        """Export uncorrected notes as HF Dataset for retraining."""
        from datasets import Dataset
        rows = [json.loads(r[0]) for r in
                self.conn.execute("SELECT data FROM notes WHERE json_extract(data,'$.correction') != ''")]
        ds = Dataset.from_list(rows)
        ds.save_to_disk(output_path)
        return ds
```

---

## 13. Directory Structure

```
ml-workbench/
├── app.py                         # Entry point: gr.Blocks or gradio.Server
├── requirements.txt
│
├── config/
│   ├── models.yaml                # ← TEMPLATE AXIS: change this for any model family
│   └── training.yaml              # LoRA hyperparams, export settings
│
├── core/
│   ├── events.py                  # EventBus, EventType enum
│   ├── registry.py                # Registry[T] generic
│   └── state.py                   # Global AppState dataclass
│
├── models/
│   ├── base.py                    # Abstract ModelService
│   ├── hf_loader.py               # Generic HF Hub downloader
│   ├── minicpm_text.py            # MiniCPM5-1B, MiniCPM4.1-8B service
│   ├── minicpm_vision.py          # MiniCPM-V-4.6 / Thinking service
│   ├── minicpm_omni.py            # MiniCPM-o-4.5 service
│   ├── llama_cpp_runner.py        # llama-server wrapper
│   ├── sglang_runner.py           # SGLang server wrapper (tool-use)
│   └── ollama_runner.py           # Ollama client
│
├── datasets/
│   ├── loader.py                  # Abstract DatasetLoader
│   ├── hf_datasets.py             # HF Hub datasets.load_dataset()
│   ├── field_notes.py             # SQLite-backed correction store
│   └── synthetic.py               # ml-intern style synthetic data gen
│
├── training/
│   ├── lora.py                    # LoRA text trainer (PEFT + TRL)
│   ├── lora_vision.py             # LoRA vision config (SWIFT/LLaMA-Factory)
│   ├── evaluation.py              # Perplexity, accuracy, custom metrics
│   ├── export.py                  # GGUF exporter (convert_hf_to_gguf.py)
│   └── hf_jobs.py                 # HF Jobs offload (ml-intern pattern)
│
├── tools/
│   ├── calculator.py              # Safe math eval tool
│   ├── dataset_stats.py           # Dataset statistics tool
│   ├── hf_search.py               # HF Hub search tool
│   └── paper_search.py            # arXiv / HF Papers search (ml-intern)
│
├── tracking/
│   └── trackio_service.py         # Trackio integration (init/log/finish/alert)
│
├── mcp/
│   ├── server.py                  # FastMCP standalone server
│   └── tools.py                   # All @mcp.tool() decorators
│
├── agent/
│   ├── loop.py                    # Research → Plan → Implement agent
│   └── prompts/
│       └── system_prompt.yaml     # Agent system prompt (ml-intern style)
│
├── ui/
│   ├── chat_tab.py
│   ├── vision_tab.py
│   ├── train_tab.py
│   ├── dataset_tab.py
│   ├── export_tab.py
│   ├── traces_tab.py
│   ├── agent_tab.py
│   └── notes_tab.py
│
├── exports/                       # GGUF + quantized model outputs
└── data/                          # Cached datasets, field_notes.db
```

---

## 14. Configuration Schema

### training.yaml

```yaml
# config/training.yaml
lora:
  rank:          16
  alpha:         32
  dropout:       0.05
  target_modules_text:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  target_modules_vision:
    - q_proj
    - k_proj
    - v_proj
    - o_proj                       # language layers only; vision tower frozen

training:
  epochs:        3
  batch_size:    4
  grad_accum:    4
  lr:            2.0e-4
  bf16:          true
  report_to:     trackio           # or: wandb, tensorboard, none

export:
  default_quant: Q4_K_M
  supported_quants:
    - F16
    - Q4_K_M
    - Q5_K_M
    - Q8_0
    - Q2_K

trackio:
  project:   ml-workbench
  space_id:  null                  # set to "username/trackio-space" for HF sync

hf_jobs:
  enabled:   false
  hardware:  gpu-a100
```

---

## 15. Dependencies

```text
# requirements.txt
# Core
gradio[mcp]>=5.38.0
mcp[cli]>=1.0.0
fastapi
uvicorn[standard]

# Models
transformers>=5.7.0               # required for MiniCPM-V-4.6
torch>=2.1.0
torchvision
torchcodec                        # video decoding for MiniCPM-V-4.6
accelerate
bitsandbytes

# Training
peft>=0.10.0
trl>=0.9.0
datasets>=2.18.0

# Tracking
trackio>=0.1.0

# MCP + Agents
mcp>=1.0.0
smolagents>=1.0.0

# Serving backends (optional, install as needed)
# sglang                          # pip install sglang
# vllm                            # pip install vllm
# ollama                          # install separately

# Hub
huggingface_hub>=0.23.0
Pillow
requests
pyyaml
```

---

## 16. Hackathon Demo Flow

```
1. DATASET
   Load HF dataset via Dataset tab (e.g. "tatsu-lab/alpaca", 10K rows)
   → EventType.DATASET_LOADED
   → Trackio: trackio.init("demo-run") + trackio.log({"dataset": ..., "rows": ...})

2. TRAIN
   Start LoRA on MiniCPM5-1B (rank=16, 1 epoch demo)
   → EventType.TRAINING_STARTED
   → TRL SFTTrainer + report_to="trackio" → live loss in Traces tab

3. EXPORT
   Click "Export GGUF" → Q4_K_M
   → convert_hf_to_gguf.py → llama-quantize
   → EventType.EXPORT_FINISHED → file download

4. LLAMA.CPP CHAT
   Load exported GGUF in llama-server
   → Chat tab, backend: llama.cpp
   → Verify text generation quality

5. SWITCH TO VISION
   Select MiniCPM-V-4.6 in Vision tab
   Upload image → enter prompt → Run

6. THINKING MODE
   Toggle "Enable Thinking Mode" → switch to MiniCPM-V-4.6-Thinking
   → Observe explicit reasoning trace in response

7. FIELD NOTE
   Response is wrong → open Field Notes tab
   → Enter correction → Save
   → EventType.FIELD_NOTE_SAVED

8. AGENT MODE
   Agent tab: "Improve MiniCPM-V-4.6 on [domain] using field notes"
   → Research phase: browse HF Papers
   → Plan: dataset selection, LoRA config
   → Implement: trigger SWIFT fine-tune

9. SHARE TRACES
   Traces tab → set space_id → Sync to HF Space
   → Shareable dashboard URL
```

---

## 17. Corrections from PRD v1

| v1 Claim | Correct v2 |
|----------|-----------|
| `trackio.trace("vision_inference")` context manager | **Does not exist.** Use `trackio.init()` + `trackio.log()` + `trackio.finish()` |
| `AutoModelForCausalLM` for MiniCPM-V-4.6 | **Wrong.** Use `AutoModelForImageTextToText` with `transformers>=5.7.0` |
| `llama-mtmd-cli` for vision deployment | Use `llama-server --mmproj <mmproj.gguf>` for server; `llama-mtmd-cli` for CLI-only |
| Run `convert_hf_to_gguf.py` for V4.6 | Official GGUFs at `openbmb/MiniCPM-V-4.6-gguf` — download directly; run script only for custom quants |
| Surgery scripts for V4.6 GGUF | Surgery scripts are for MiniCPM-o-2.6 only; **do not use for V4.6** |
| `future MiniCPM-o` (vague) | **MiniCPM-o-4.5** released 2026-05-17; specific HF ID available |
| `app.launch(mcp_server=True)` only option | Two paths: `gr.Blocks` + `mcp_server=True`, or `gradio.Server` with `@app.mcp.tool()` |
| FastMCP as only MCP approach | Three paths: Gradio native, gradio.Server, standalone FastMCP (see §7) |
| MiniCPM4.1-8B: standard architecture | Has sparse InfLLM-v2 attention; requires `--trust-remote-code` |
| LoRA for vision: PEFT only | OpenBMB recommends **SWIFT** or **LLaMA-Factory** for MiniCPM-V fine-tuning |
| No HF Jobs mention | HF Jobs integration (ml-intern pattern) enables GPU cloud offload |
| No SGLang mention | SGLang is the **recommended** backend for MiniCPM5-1B tool calling |
| No video support | MiniCPM-V-4.6 supports **video understanding** (`torchcodec` required) |

---

## 18. Roadmap & Extension Points

### Swap the model family (template portability)

To build a "Qwen3 Workbench":
1. Edit `config/models.yaml` — add Qwen3 model IDs
2. Create `models/qwen3_service.py` extending `ModelService`
3. Register in `app.py` — done

No changes to training, export, MCP, Trackio, or UI layers.

### Planned extensions

| Feature | Module | Notes |
|---------|--------|-------|
| vLLM serving tab | `models/vllm_runner.py` | Batch inference, production |
| Ollama quick-start | `models/ollama_runner.py` | Zero-config consumer demo |
| Reward model eval | `training/reward_eval.py` | For RLHF experiments |
| Synthetic data gen | `datasets/synthetic.py` | ml-intern pattern: LLM writes training data |
| Paper-to-code agent | `agent/paper_agent.py` | Read arXiv → implement in workbench |
| HF Spaces deploy | `deploy/spaces.py` | One-click push as HF Space |
| VINDEX integration | `tools/vindex_tool.py` | Knowledge editing via your VINDEX engine |
| OCR pipeline hook | `datasets/ocr_loader.py` | Feed your OCR pipeline output as field notes |
| MiniCPM Desk-Pet | `agent/desk_pet.py` | LoRA persona switching via MiniCPM5-1B |
| MiniCPM-o audio | `ui/audio_tab.py` | Real-time omnimodal via MiniCPM-o-4.5 |

---

*PRD v2.0 — Christof Kaller / ki-fusion-labs.de — 2026-06-05*
*Research sources: openbmb/MiniCPM-V, openbmb/MiniCPM, gradio-app/trackio,*
*huggingface/ml-intern, huggingface.co/blog/introducing-gradio-server*