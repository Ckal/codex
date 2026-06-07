from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.local_backend_config import (
    LocalBackendConfig,
    build_llama_server_command,
    load_local_backend_config,
    local_backend_summary,
    save_local_backend_config,
)


class LocalBackendConfigTest(unittest.TestCase):
    def test_loads_defaults_when_config_is_missing(self) -> None:
        config = load_local_backend_config("missing-local-backends.yaml")

        self.assertEqual(config.llama_cpp_server_url, "http://127.0.0.1:8080")
        self.assertEqual(config.openai_compatible_base_url, "http://127.0.0.1:1234")
        self.assertEqual(config.openai_compatible_model_name, "")
        self.assertEqual(config.gguf_path, "")

    def test_saves_and_loads_local_backend_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_backends.yaml"
            expected = LocalBackendConfig(
                llama_cpp_server_url="http://127.0.0.1:9090",
                openai_compatible_base_url="http://local.test:1234",
                openai_compatible_model_name="loaded-model",
                gguf_path="model.gguf",
                mmproj_path="mmproj.gguf",
                n_ctx=8192,
                n_gpu_layers=12,
            )

            saved = save_local_backend_config(expected, path)
            actual = load_local_backend_config(saved)

            self.assertEqual(actual, expected)

    def test_builds_llama_server_command_only_when_model_is_configured(self) -> None:
        self.assertEqual(build_llama_server_command(LocalBackendConfig()), [])

        command = build_llama_server_command(
            LocalBackendConfig(gguf_path="model.gguf", mmproj_path="mmproj.gguf")
        )

        self.assertEqual(
            command,
            ["llama-server", "-m", "model.gguf", "--mmproj", "mmproj.gguf"],
        )

    def test_summary_records_no_startup_download_or_auto_load(self) -> None:
        summary = local_backend_summary(LocalBackendConfig(gguf_path="model.gguf"))

        self.assertFalse(summary["startup_downloads"])
        self.assertFalse(summary["auto_model_load"])
        self.assertIn("llama-server -m model.gguf", summary["llama_server_command"])
        self.assertEqual(summary["openai_compatible_base_url"], "http://127.0.0.1:1234")


if __name__ == "__main__":
    unittest.main()
