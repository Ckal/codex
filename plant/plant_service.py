from __future__ import annotations

import importlib.util
import json
import re
import time
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PlantID:
    common_name: str = "Unknown"
    latin_name: str = "Unknown"
    family: str = "Unknown"
    genus: str = "Unknown"
    confidence: float = 0.0
    key_features: list[str] = field(default_factory=list)
    care_tips: list[str] = field(default_factory=list)
    toxicity: dict[str, str] = field(
        default_factory=lambda: {"humans": "unknown", "pets": "unknown"}
    )
    habitat: str = ""
    bloom_season: str = ""
    similar_species: list[str] = field(default_factory=list)
    notes: str = ""
    thinking_trace: str = ""
    latency_ms: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_uncertain(self, threshold: float = 0.70) -> bool:
        return self.confidence < threshold

    @classmethod
    def from_mapping(cls, data: dict[str, Any], model_used: str = "") -> PlantID:
        latin_name = str(data.get("latin_name") or "Unknown")
        genus = str(data.get("genus") or _genus_from_latin(latin_name))
        return cls(
            common_name=str(data.get("common_name") or latin_name),
            latin_name=latin_name,
            family=str(data.get("family") or "Unknown"),
            genus=genus,
            confidence=_clamp_float(data.get("confidence"), 0.0, 1.0),
            key_features=_string_list(data.get("key_features")),
            care_tips=_string_list(data.get("care_tips")),
            toxicity=_toxicity_mapping(data.get("toxicity")),
            habitat=str(data.get("habitat") or ""),
            bloom_season=str(data.get("bloom_season") or ""),
            similar_species=_string_list(data.get("similar_species")),
            notes=str(data.get("notes") or ""),
            model_used=model_used,
        )


@dataclass(frozen=True)
class PlantVisionConfig:
    model_id: str = "openbmb/MiniCPM-V-4.6"
    thinking_model_id: str = "openbmb/MiniCPM-V-4.6-Thinking"
    adapter_id: str = ""
    model_key: str = "plant_vlm"
    confidence_threshold: float = 0.70
    max_new_tokens: int = 512
    temperature: float = 0.1
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"
    auto_thinking: bool = True


class DemoPlantVisionService:
    """Deterministic no-model service for screenshots, tests, and template demos."""

    model_id = "demo/plant-discovery"

    def service_status(self) -> dict[str, Any]:
        return {
            "mode": "demo",
            "uses_llm": False,
            "model_id": self.model_id,
            "ready": True,
            "notes": "Deterministic demo mode; no model weights are loaded.",
        }

    def identify(
        self,
        image: Any,
        extra_images: list[Any] | None = None,
        force_thinking: bool = False,
    ) -> PlantID:
        image_count = 1 + len(extra_images or [])
        confidence = 0.86 if force_thinking or image_count > 1 else 0.72
        return PlantID(
            common_name="Common Daisy",
            latin_name="Bellis perennis",
            family="Asteraceae",
            genus="Bellis",
            confidence=confidence,
            key_features=[
                "white ray florets around yellow disc",
                "low basal rosette",
                "small lawn-forming habit",
            ],
            care_tips=[
                "Prefers full sun to partial shade",
                "Tolerates compact lawn soil",
                "Water during long dry periods",
            ],
            toxicity={"humans": "non-toxic", "pets": "non-toxic"},
            habitat="Lawns, meadows, paths, and disturbed grassland",
            bloom_season="spring to autumn",
            similar_species=["Leucanthemum vulgare", "Erigeron annuus"],
            notes="Demo result for the template app. Replace with a real MiniCPM-V backend.",
            thinking_trace="Demo mode used a deterministic plant card." if force_thinking else "",
            latency_ms=25.0,
            model_used=self.model_id,
        )


class PlantVisionService:
    """Optional MiniCPM-V plant identification adapter.

    The class is importable without torch/transformers. Heavy dependencies and model weights are
    touched only when identify() is called.
    """

    def __init__(self, config: PlantVisionConfig | None = None) -> None:
        self.config = config or PlantVisionConfig()
        self.processor: Any | None = None
        self.model: Any | None = None
        self.thinking_processor: Any | None = None
        self.thinking_model: Any | None = None

    @staticmethod
    def dependency_report() -> dict[str, bool]:
        return {
            "transformers": importlib.util.find_spec("transformers") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "pillow": importlib.util.find_spec("PIL") is not None,
            "peft": importlib.util.find_spec("peft") is not None,
        }

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        model_key: str = "plant_vlm",
    ) -> PlantVisionService:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        models = cfg.get("models", {})
        inference = cfg.get("inference", {})
        primary = models.get(model_key, {}) or models.get("plant_vlm", {})
        thinking = models.get("plant_vlm_thinking", {})
        adapter_id = str(primary.get("adapter_id") or "")
        if not adapter_id and bool(primary.get("requires_training", False)):
            adapter_id = str(primary.get("hf_id") or "")
        return cls(
            PlantVisionConfig(
                model_id=str(
                    primary.get("base_model")
                    or primary.get("hf_id")
                    or PlantVisionConfig.model_id
                ),
                thinking_model_id=str(
                    thinking.get("hf_id") or PlantVisionConfig.thinking_model_id
                ),
                adapter_id=adapter_id,
                model_key=model_key,
                confidence_threshold=float(inference.get("confidence_threshold", 0.70)),
                max_new_tokens=int(inference.get("max_new_tokens", 512)),
                temperature=float(inference.get("temperature", 0.1)),
                trust_remote_code=bool(primary.get("trust_remote_code", True)),
                torch_dtype=str(primary.get("dtype") or "auto"),
                auto_thinking=bool(inference.get("auto_thinking", True)),
            )
        )

    def service_status(self) -> dict[str, Any]:
        dependencies = self.dependency_report()
        required = ["transformers", "torch", "pillow"]
        if self.config.adapter_id:
            required.append("peft")
        missing = [name for name in required if not dependencies[name]]
        return {
            "mode": "openbmb-finetuned" if self.config.adapter_id else "openbmb-zero-shot",
            "uses_llm": True,
            "model_id": self.config.model_id,
            "adapter_id": self.config.adapter_id,
            "model_key": self.config.model_key,
            "ready": not missing,
            "missing_dependencies": missing,
            "notes": (
                "Uses OpenBMB MiniCPM-V through Transformers. Weights load only during Identify."
            ),
        }

    def identify(
        self,
        image: Any,
        extra_images: list[Any] | None = None,
        force_thinking: bool = False,
    ) -> PlantID:
        status = self.service_status()
        missing = list(status["missing_dependencies"])
        if missing:
            return PlantID(
                common_name="Model unavailable",
                latin_name="MiniCPM-V dependencies missing",
                confidence=0.0,
                notes=(
                    "Install optional plant model dependencies before real inference: "
                    + ", ".join(missing)
                ),
                model_used=self.config.model_id,
            )

        start = time.perf_counter()
        try:
            result = self._identify_with_model(image, extra_images, force_thinking)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            return PlantID(
                common_name="Model unavailable",
                latin_name="MiniCPM-V could not run",
                confidence=0.0,
                notes=str(exc),
                model_used=self.config.model_id,
            )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _identify_with_model(
        self,
        image: Any,
        extra_images: list[Any] | None,
        force_thinking: bool,
    ) -> PlantID:
        images = [image, *(extra_images or [])][:4]
        if force_thinking:
            return self._infer(images, thinking=True)

        result = self._infer(images, thinking=False)
        if self.config.auto_thinking and result.is_uncertain(self.config.confidence_threshold):
            thinking_result = self._infer(images, thinking=True)
            if thinking_result.confidence > result.confidence:
                return thinking_result
        return result

    def _infer(self, images: list[Any], thinking: bool) -> PlantID:
        torch = import_module("torch")
        model, processor = self._load_components(thinking=thinking)
        messages = build_plant_messages(images, thinking=thinking)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens if not thinking else 1024,
                temperature=self.config.temperature if not thinking else 0.2,
                do_sample=True,
            )

        raw = processor.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        model_id = self.config.thinking_model_id if thinking else self.config.model_id
        parsed = parse_plant_response(raw, model_used=model_id)
        if thinking:
            parsed.thinking_trace = extract_thinking_trace(raw)
        return parsed

    def _load_components(self, thinking: bool) -> tuple[Any, Any]:
        transformers = import_module("transformers")
        model_id = self.config.thinking_model_id if thinking else self.config.model_id

        if thinking and self.thinking_model is not None and self.thinking_processor is not None:
            return self.thinking_model, self.thinking_processor
        if not thinking and self.model is not None and self.processor is not None:
            return self.model, self.processor

        processor = transformers.AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
        )
        if self.config.adapter_id and not thinking:
            peft = import_module("peft")
            model = peft.PeftModel.from_pretrained(model, self.config.adapter_id)
        model.eval()

        if thinking:
            self.thinking_model = model
            self.thinking_processor = processor
        else:
            self.model = model
            self.processor = processor
        return model, processor


def build_plant_messages(images: list[Any], thinking: bool = False) -> list[dict[str, Any]]:
    prompt = (
        "This is a difficult plant identification. Compare leaf shape, flowers, fruit, stem, "
        "and growth habit before answering."
        if thinking
        else "Identify this plant species from the image."
    )
    if len(images) > 1:
        prompt = f"Identify the plant using these {len(images)} images from different angles."

    content = [{"type": "image", "image": image} for image in images[:4]]
    content.append({"type": "text", "text": prompt + " Respond only as valid JSON."})
    return [
        {"role": "system", "content": plant_system_prompt()},
        {"role": "user", "content": content},
    ]


def plant_system_prompt() -> str:
    return (
        "You are a careful botanist. Return only JSON with keys: common_name, latin_name, "
        "family, genus, confidence, key_features, care_tips, toxicity, habitat, "
        "bloom_season, similar_species, notes."
    )


def parse_plant_response(raw: str, model_used: str = "") -> PlantID:
    data = extract_json_object(raw)
    if "error" in data:
        return PlantID(
            notes=f"Could not parse model JSON: {data.get('raw', '')}",
            model_used=model_used,
        )
    return PlantID.from_mapping(data, model_used=model_used)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    for candidate in _json_candidates(stripped):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return {"error": "unparseable", "raw": stripped[:300]}


def extract_thinking_trace(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _genus_from_latin(latin_name: str) -> str:
    return latin_name.split()[0] if " " in latin_name else latin_name


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _toxicity_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "humans": str(value.get("humans") or "unknown"),
            "pets": str(value.get("pets") or "unknown"),
        }
    return {"humans": "unknown", "pets": "unknown"}


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return minimum
    return min(max(parsed, minimum), maximum)
