from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

LOCAL_BACKEND_CONFIG_PATH = Path("data/local_backends.yaml")


@dataclass(frozen=True)
class LocalBackendConfig:
    """User-local backend settings stored outside git-tracked config."""

    llama_cpp_server_url: str = "http://127.0.0.1:8080"
    openai_compatible_base_url: str = "http://127.0.0.1:1234"
    openai_compatible_model_name: str = ""
    gguf_path: str = ""
    mmproj_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = 0

    def file_status_rows(self) -> list[list[str]]:
        return [
            ["GGUF model", self.gguf_path, _local_file_status(self.gguf_path)],
            ["mmproj", self.mmproj_path, _local_file_status(self.mmproj_path)],
        ]


def load_local_backend_config(
    path: str | Path = LOCAL_BACKEND_CONFIG_PATH,
) -> LocalBackendConfig:
    config_path = Path(path)
    if not config_path.exists():
        return LocalBackendConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llama_cpp = raw.get("llama_cpp", {})
    openai_compatible = raw.get("openai_compatible", {})
    return LocalBackendConfig(
        llama_cpp_server_url=str(
            llama_cpp.get("server_url", LocalBackendConfig.llama_cpp_server_url)
        ),
        openai_compatible_base_url=str(
            openai_compatible.get(
                "base_url",
                LocalBackendConfig.openai_compatible_base_url,
            )
        ),
        openai_compatible_model_name=str(openai_compatible.get("model_name", "")),
        gguf_path=str(llama_cpp.get("gguf_path", "")),
        mmproj_path=str(llama_cpp.get("mmproj_path", "")),
        n_ctx=int(llama_cpp.get("n_ctx", LocalBackendConfig.n_ctx)),
        n_gpu_layers=int(llama_cpp.get("n_gpu_layers", LocalBackendConfig.n_gpu_layers)),
    )


def save_local_backend_config(
    config: LocalBackendConfig,
    path: str | Path = LOCAL_BACKEND_CONFIG_PATH,
) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, Any]] = {
        "llama_cpp": {
            "server_url": config.llama_cpp_server_url,
            "gguf_path": config.gguf_path,
            "mmproj_path": config.mmproj_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
        },
        "openai_compatible": {
            "base_url": config.openai_compatible_base_url,
            "model_name": config.openai_compatible_model_name,
        },
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def build_llama_server_command(config: LocalBackendConfig) -> list[str]:
    if not config.gguf_path:
        return []

    command = ["llama-server", "-m", config.gguf_path]
    if config.mmproj_path:
        command.extend(["--mmproj", config.mmproj_path])
    return command


def local_backend_summary(config: LocalBackendConfig) -> dict[str, Any]:
    command = build_llama_server_command(config)
    return {
        **asdict(config),
        "llama_server_command": " ".join(command) if command else "",
        "files": config.file_status_rows(),
        "startup_downloads": False,
        "auto_model_load": False,
    }


def _local_file_status(path: str) -> str:
    if not path:
        return "not configured"
    return "found" if Path(path).exists() else "missing"
