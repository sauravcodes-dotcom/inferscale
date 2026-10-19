from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inferscale.config import load_settings
from inferscale.registry import Registry, load_registry
from tests.conftest import MODEL_VERSION, REGISTRY_PATH, SERVING_PATH


def test_loads_serving_profile() -> None:
    s = load_settings(SERVING_PATH)
    assert s.engine_url == "http://127.0.0.1:8081"
    assert s.admission.max_inflight == 2
    assert s.admission.max_queue == 8
    assert s.limits.max_tokens_cap == 1024
    assert s.drain_timeout_s == 30.0
    assert s.log_content is False


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERSCALE_ENGINE_URL", "http://10.0.0.1:9000")
    monkeypatch.setenv("INFERSCALE_ADMISSION__MAX_INFLIGHT", "5")
    monkeypatch.setenv("INFERSCALE_DRAIN_TIMEOUT_S", "3")
    s = load_settings(SERVING_PATH)
    assert s.engine_url == "http://10.0.0.1:9000"
    assert s.admission.max_inflight == 5
    assert s.admission.max_queue == 8  # untouched YAML value
    assert s.drain_timeout_s == 3.0


def test_config_file_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("engine_url: http://example:1\n")
    monkeypatch.setenv("INFERSCALE_CONFIG_FILE", str(p))
    assert load_settings().engine_url == "http://example:1"


def test_missing_config_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_settings("/nonexistent.yaml")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("admission:\n  max_inflght: 3\n")
    with pytest.raises(ValidationError):
        load_settings(p)


def test_registry_loads_active_model() -> None:
    reg = load_registry(REGISTRY_PATH)
    assert reg.active == MODEL_VERSION
    e = reg.active_entry
    assert e.hf_repo == "Qwen/Qwen3-8B-GGUF"
    assert e.file == "Qwen3-8B-Q4_K_M.gguf"
    assert e.sha256 == "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
    assert e.engine == "llama.cpp" and e.engine_version == "b11669"
    assert e.quant == "Q4_K_M" and e.thinking is False


def _registry_data() -> dict:  # type: ignore[type-arg]
    return yaml.safe_load(REGISTRY_PATH.read_text())  # type: ignore[no-any-return]


def test_registry_rejects_unknown_active() -> None:
    data = _registry_data()
    data["active"] = "nope"
    with pytest.raises(ValidationError, match="not in models"):
        Registry.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sha256", "abc"), ("hf_revision", "main"), ("engine", "vllm"), ("extra", 1)],
)
def test_registry_rejects_bad_entry(field: str, value: object) -> None:
    data = _registry_data()
    data["models"][MODEL_VERSION][field] = value
    with pytest.raises(ValidationError):
        Registry.model_validate(data)


def test_active_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = load_registry(REGISTRY_PATH, "qwen3-8b-q8-v1")
    assert reg.active == "qwen3-8b-q8-v1" and reg.active_entry.quant == "Q8_0"
    assert load_registry(REGISTRY_PATH).active == MODEL_VERSION  # file default unchanged
    with pytest.raises(ValidationError):
        load_registry(REGISTRY_PATH, "no-such-model")
    monkeypatch.setenv("INFERSCALE_ACTIVE_MODEL", "qwen3-8b-q8-v1")
    assert load_settings(SERVING_PATH).active_model == "qwen3-8b-q8-v1"
