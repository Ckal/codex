from __future__ import annotations

import unittest
from pathlib import Path


class GitignorePolicyTest(unittest.TestCase):
    def test_gitignore_blocks_secrets_caches_and_model_weights(self) -> None:
        patterns = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())

        required = {
            ".env",
            ".env.*",
            "*.key",
            "*.pem",
            "data/*",
            "exports/*",
            "models_cache/",
            "hf_cache/",
            "wandb/",
            "*.gguf",
            "*.safetensors",
            "*.bin",
            "*.ckpt",
            "*.onnx",
            "*.tflite",
            "*.pt",
            "*.pth",
            "*.model",
        }

        self.assertTrue(required.issubset(patterns))


if __name__ == "__main__":
    unittest.main()
