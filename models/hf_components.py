from __future__ import annotations

from importlib import import_module

from models.model_catalog import ModelInfo


def load_tokenizer_and_causal_lm(
    model: ModelInfo,
    trust_remote_code: bool,
    device_map: str,
    torch_dtype: str,
):
    transformers_module = import_module("transformers")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        model.hf_id,
        trust_remote_code=trust_remote_code,
    )
    model_instance = transformers_module.AutoModelForCausalLM.from_pretrained(
        model.hf_id,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model_instance, tokenizer


def load_processor_and_image_text_model(
    model: ModelInfo,
    trust_remote_code: bool,
    device_map: str,
    torch_dtype: str,
):
    transformers_module = import_module("transformers")
    processor = transformers_module.AutoProcessor.from_pretrained(
        model.hf_id,
        trust_remote_code=trust_remote_code,
    )
    model_instance = transformers_module.AutoModelForImageTextToText.from_pretrained(
        model.hf_id,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model_instance, processor
