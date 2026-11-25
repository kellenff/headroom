"""``HEADROOM_EMBEDDER_*`` as LocalBackend settings, shared by every memory process."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.memory.backends.local import LocalBackendConfig
from headroom.memory.embedder_env import (
    local_backend_embedder_kwargs,
    local_backend_embedder_kwargs_from_env,
    validate_memory_embedder_env,
)
from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    UnknownBackendError,
    register_embedding_backend,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    EMBEDDING_BACKENDS._reset_for_tests()
    yield
    EMBEDDING_BACKENDS._reset_for_tests()


FULL = {
    "HEADROOM_EMBEDDER_PROVIDER": "openai",
    "HEADROOM_EMBEDDER_MODEL": "nomic-embed-text",
    "HEADROOM_EMBEDDER_BASE_URL": "http://gpu-box:8080/v1",
    "HEADROOM_EMBEDDER_API_KEY": "sk-test-secret",
    "HEADROOM_EMBEDDER_DIMENSION": "768",
    "HEADROOM_EMBEDDER_HEADERS": "X-Team=memory",
    "HEADROOM_EMBEDDER_OPTIONS": '{"timeout": 5}',
}

EXPECTED = {
    "embedder_backend": "external",
    "embedder_backend_name": "openai",
    "embedder_options": {
        "model": "nomic-embed-text",
        "base_url": "http://gpu-box:8080/v1",
        "api_key": "sk-test-secret",
        "headers": {"X-Team": "memory"},
        "timeout": 5,
    },
    "vector_dimension": 768,
}


def test_unset_keeps_every_callers_default() -> None:
    assert local_backend_embedder_kwargs_from_env({}) == {}
    assert validate_memory_embedder_env({}) is None
    assert local_backend_embedder_kwargs(None) == {}


@pytest.mark.parametrize("runtime", ["cpu", "pytorch_mps"])
def test_a_device_runtime_alone_is_the_builtin_embedder(runtime: str) -> None:
    assert local_backend_embedder_kwargs_from_env({"HEADROOM_EMBEDDER_RUNTIME": runtime}) == {}


def test_a_provider_becomes_an_external_embedder() -> None:
    kwargs = local_backend_embedder_kwargs_from_env(FULL)
    assert kwargs == EXPECTED
    # The kwargs are exactly LocalBackendConfig's.
    config = LocalBackendConfig(db_path="memory.db", **kwargs)
    assert config.vector_dimension == 768


def test_runtime_openai_is_the_same_selection() -> None:
    env = {k: v for k, v in FULL.items() if k != "HEADROOM_EMBEDDER_PROVIDER"}
    env["HEADROOM_EMBEDDER_RUNTIME"] = "openai"
    assert local_backend_embedder_kwargs_from_env(env) == EXPECTED
    settings = validate_memory_embedder_env(env)
    assert settings is not None and settings.source == "HEADROOM_EMBEDDER_RUNTIME=openai"


def test_the_model_is_optional() -> None:
    env = {"HEADROOM_EMBEDDER_PROVIDER": "openai", "HEADROOM_EMBEDDER_DIMENSION": "1536"}
    assert local_backend_embedder_kwargs_from_env(env)["embedder_options"] == {}


@pytest.mark.parametrize(
    "selector",
    [{"HEADROOM_EMBEDDER_PROVIDER": "openai"}, {"HEADROOM_EMBEDDER_RUNTIME": "openai"}],
)
def test_memory_requires_the_dimension(selector: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_DIMENSION is required"):
        local_backend_embedder_kwargs_from_env(selector)


@pytest.mark.parametrize(
    "env",
    [
        {"HEADROOM_EMBEDDER_RUNTIME": "openai", "HEADROOM_EMBEDDER_PROVIDER": "fastembed"},
        {"HEADROOM_EMBEDDER_RUNTIME": "pytorch_mps", "HEADROOM_EMBEDDER_PROVIDER": "openai"},
    ],
)
def test_contradictions_are_refused(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_"):
        local_backend_embedder_kwargs_from_env({**env, "HEADROOM_EMBEDDER_DIMENSION": "3"})


def test_an_unknown_provider_names_the_variable_not_the_value() -> None:
    env = {"HEADROOM_EMBEDDER_PROVIDER": "no-such-thing", "HEADROOM_EMBEDDER_DIMENSION": "3"}
    with pytest.raises(UnknownBackendError, match="HEADROOM_EMBEDDER_PROVIDER") as excinfo:
        validate_memory_embedder_env(env)
    assert "no-such-thing" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("override", "var"),
    [
        ({"HEADROOM_EMBEDDER_DIMENSION": "zero"}, "HEADROOM_EMBEDDER_DIMENSION"),
        ({"HEADROOM_EMBEDDER_HEADERS": "nonsense"}, "HEADROOM_EMBEDDER_HEADERS"),
        ({"HEADROOM_EMBEDDER_OPTIONS": "[1]"}, "HEADROOM_EMBEDDER_OPTIONS"),
    ],
)
def test_malformed_values_name_the_variable_and_hide_secrets(
    override: dict[str, Any], var: str
) -> None:
    with pytest.raises(ValueError, match=var) as excinfo:
        local_backend_embedder_kwargs_from_env({**FULL, **override})
    assert "sk-test-secret" not in str(excinfo.value)


def test_validation_constructs_nothing() -> None:
    built: list[dict[str, Any]] = []
    register_embedding_backend("spy", lambda **options: built.append(options))
    env = {"HEADROOM_EMBEDDER_PROVIDER": "spy", "HEADROOM_EMBEDDER_DIMENSION": "3"}
    assert validate_memory_embedder_env(env) is not None
    assert built == []
