"""Tests for the pluggable model-backend core (``headroom.models.backends``)."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    BackendError,
    BackendLoadError,
    BackendRegistry,
    CallableEmbeddingBackend,
    EmbeddingBackend,
    EmbeddingBackendCallError,
    EmbeddingBackendResponseError,
    UnknownBackendError,
    as_embedding_backend,
    available_embedding_backends,
    create_embedding_backend,
    embed_texts,
    embedding_backend_settings_from_env,
    memory_embedder_settings_from_env,
    parse_backend_spec,
    register_embedding_backend,
    relevance_embedder_settings_from_env,
    unregister_embedding_backend,
)
from headroom.models.backends import registry as registry_module
from headroom.models.backends.openai_compat import (
    EmbeddingBackendRequestError,
    OpenAICompatibleEmbeddingBackend,
)
from headroom.offline import OfflineEgressBlocked


@pytest.fixture(autouse=True)
def _clean_embedding_registry():
    EMBEDDING_BACKENDS._reset_for_tests()
    yield
    EMBEDDING_BACKENDS._reset_for_tests()


class _FakeEntryPoint:
    def __init__(self, name: str, value: Any, *, group: str = "test.group") -> None:
        self.name = name
        self.value = f"fake.module:{name}"
        self.group = group
        self._target = value
        self.loads = 0

    def load(self) -> Any:
        self.loads += 1
        if isinstance(self._target, BaseException):
            raise self._target
        return self._target


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]) -> None:
    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)


class _Backend:
    """A minimal conforming embedding backend used across tests."""

    name = "fake"

    def __init__(self, model: str | None = None, dimension: int | None = 3, **options: Any):
        self.model_name = model or "fake-model"
        self.dimension = dimension
        self.options = options
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> Any:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


# =============================================================================
# parse_backend_spec / names
# =============================================================================


class TestParseBackendSpec:
    def test_name_only(self) -> None:
        assert parse_backend_spec("openai") == ("openai", None)

    def test_name_and_model(self) -> None:
        assert parse_backend_spec(" OpenAI : text-embedding-3-small ") == (
            "openai",
            "text-embedding-3-small",
        )

    def test_only_the_first_colon_splits(self) -> None:
        assert parse_backend_spec("ollama:nomic-embed-text:latest") == (
            "ollama",
            "nomic-embed-text:latest",
        )

    def test_empty_model_is_none(self) -> None:
        assert parse_backend_spec("fastembed:") == ("fastembed", None)

    @pytest.mark.parametrize("spec", ["", "   ", ":model", "bad name", "Ünicode"])
    def test_invalid(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_backend_spec(spec)


# =============================================================================
# BackendRegistry
# =============================================================================


class TestBackendRegistry:
    def test_register_and_create_with_options(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("mine", lambda **kw: ("built", kw))
        assert reg.create("mine", a=1) == ("built", {"a": 1})
        assert "mine" in reg.names()

    def test_duplicate_needs_replace(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("mine", lambda **kw: 1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("mine", lambda **kw: 2)
        reg.register("mine", lambda **kw: 2, replace=True)
        assert reg.create("mine") == 2

    def test_builtin_needs_replace_and_unregister_restores_it(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", lambda **kw: "builtin")
        with pytest.raises(ValueError):
            reg.register("core", lambda **kw: "override")
        reg.register("core", lambda **kw: "override", replace=True)
        assert reg.create("core") == "override"
        assert reg.unregister("core") is True
        assert reg.create("core") == "builtin"
        assert reg.unregister("core") is False

    @pytest.mark.parametrize("name", ["", "Upper", "-lead", "a" * 65, "sp ace", "x/y"])
    def test_invalid_names_rejected(self, name: str) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        with pytest.raises(ValueError, match="Invalid backend name"):
            reg.register(name, lambda **kw: 1)

    def test_bad_factory_rejected(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        with pytest.raises(TypeError):
            reg.register("x", 42)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="module:attr"):
            reg.register("x", "not a target")

    def test_lazy_string_factory_imports_on_first_use(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("ordered", "collections:OrderedDict")
        assert reg.create("ordered", a=1) == {"a": 1}

    def test_lazy_string_import_failure_is_a_load_error(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("ghost", "headroom_no_such_module:Thing")
        with pytest.raises(BackendLoadError, match="ghost"):
            reg.create("ghost")

    def test_unknown_lists_available_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("plugin", lambda **kw: 1)])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("mine", lambda **kw: 1)
        with pytest.raises(UnknownBackendError) as excinfo:
            reg.create("nope")
        assert isinstance(excinfo.value, ValueError)
        assert "mine" in str(excinfo.value) and "plugin" in str(excinfo.value)
        assert "test.group" in str(excinfo.value)

    def test_entry_point_loaded_lazily_and_only_when_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wanted = _FakeEntryPoint("wanted", lambda **kw: ("wanted", kw))
        other = _FakeEntryPoint("other", RuntimeError("must not be imported"))
        _patch_entry_points(monkeypatch, [wanted, other])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")

        assert reg.names() == ["other", "wanted"]
        assert wanted.loads == 0 and other.loads == 0

        assert reg.create("wanted", k="v") == ("wanted", {"k": "v"})
        assert reg.create("wanted") == ("wanted", {})
        assert wanted.loads == 1
        assert other.loads == 0

    def test_entry_point_cannot_shadow_a_builtin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shadow = _FakeEntryPoint("core", lambda **kw: "plugin")
        _patch_entry_points(monkeypatch, [shadow])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", lambda **kw: "builtin")
        assert reg.create("core") == "builtin"
        assert shadow.loads == 0

    def test_entry_point_load_failure_is_a_load_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("broken", ImportError("no dep"))])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        with pytest.raises(BackendLoadError, match="broken"):
            reg.create("broken")

    def test_non_callable_entry_point_is_a_load_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("value", 42)])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        with pytest.raises(BackendLoadError, match="not callable"):
            reg.create("value")

    def test_third_party_factory_failure_echoes_neither_options_nor_its_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def factory(**kw: Any) -> Any:
            raise RuntimeError("could not connect to https://h/?key=CANARY-SECRET")

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("remote", factory)
        with caplog.at_level("DEBUG"), pytest.raises(BackendLoadError) as excinfo:
            reg.create("remote", api_key="sk-very-secret")
        message = str(excinfo.value)
        assert "sk-very-secret" not in message and "CANARY-SECRET" not in message
        assert "RuntimeError" in message and "remote" in message
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert all("CANARY-SECRET" not in r.getMessage() for r in caplog.records)

    def test_builtin_factory_failure_keeps_its_message(self) -> None:
        def factory(**kw: Any) -> Any:
            raise ValueError("model is required")

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", factory)
        with pytest.raises(BackendLoadError, match="model is required"):
            reg.create("core")
        reg.register("core", factory, replace=True)
        with pytest.raises(BackendLoadError) as excinfo:
            reg.create("core")
        assert "model is required" not in str(excinfo.value)

    def test_duplicate_entry_point_names_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = _FakeEntryPoint("dup", lambda **kw: "first")
        second = _FakeEntryPoint("dup", lambda **kw: "second")
        _patch_entry_points(monkeypatch, [first, second])
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        with pytest.raises(BackendLoadError, match="2 installed packages"):
            reg.create("dup")
        assert first.loads == 0 and second.loads == 0

    def test_third_party_factory_backend_error_is_reduced_to_its_class(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class PluginError(BackendError):
            pass

        def factory(**kw: Any) -> Any:
            raise PluginError("could not reach https://u:CANARY-SECRET@host/")

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("remote", factory)
        with caplog.at_level("DEBUG"), pytest.raises(BackendLoadError) as excinfo:
            reg.create("remote")
        assert type(excinfo.value) is BackendLoadError
        assert "CANARY-SECRET" not in str(excinfo.value)
        assert "PluginError" in str(excinfo.value) and "remote" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, PluginError)
        assert all("CANARY-SECRET" not in r.getMessage() for r in caplog.records)

    def test_builtin_factory_backend_error_passes_through(self) -> None:
        hint = BackendLoadError("model unavailable; pre-seed the cache")

        def factory(**kw: Any) -> Any:
            raise hint

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", factory)
        with pytest.raises(BackendLoadError) as excinfo:
            reg.create("core")
        assert excinfo.value is hint
        # The same factory registered as a third-party override is not trusted.
        reg.register("core", factory, replace=True)
        with pytest.raises(BackendLoadError) as excinfo:
            reg.create("core")
        assert excinfo.value is not hint and "pre-seed" not in str(excinfo.value)

    def test_registration_racing_an_entry_point_load_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")

        def registered(**kw: Any) -> str:
            return "registered"

        class RacingEntryPoint(_FakeEntryPoint):
            def load(self) -> Any:
                # Another thread registers the name while the plugin imports.
                reg.register("x", registered)
                return super().load()

        _patch_entry_points(monkeypatch, [RacingEntryPoint("x", lambda **kw: "entry-point")])
        assert reg.get_factory("x") is registered
        assert reg.get_factory("x") is registered
        assert reg.create("x") == "registered"

    def test_builtin_registered_during_an_entry_point_load_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")

        class RacingEntryPoint(_FakeEntryPoint):
            def load(self) -> Any:
                reg.register_builtin("x", lambda **kw: "builtin")
                return super().load()

        _patch_entry_points(monkeypatch, [RacingEntryPoint("x", lambda **kw: "entry-point")])
        assert reg.create("x") == "builtin"
        assert reg.create("x") == "builtin"

    def test_is_builtin_instance(self) -> None:
        from collections import OrderedDict

        class Core:
            pass

        class Sub(Core):
            pass

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", Core)
        reg.register_builtin("lazy", "collections:OrderedDict")
        reg.register_builtin("ghost", "headroom_no_such_module:Thing")
        reg.register("alias", Core)

        assert reg.is_builtin_instance(Core())
        assert not reg.is_builtin_instance(Sub())
        assert not reg.is_builtin_instance(object())
        assert reg.is_builtin_instance(OrderedDict())
        # A lazy built-in whose module was never imported is not imported to check.
        assert not reg.is_builtin_instance(42)
        assert "headroom_no_such_module" not in sys.modules

    def test_offline_refusal_is_not_wrapped(self) -> None:
        def factory(**kw: Any) -> Any:
            raise OfflineEgressBlocked("test backend", "example.com")

        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register("remote", factory)
        with pytest.raises(OfflineEgressBlocked):
            reg.create("remote")

    def test_reset_keeps_builtins(self) -> None:
        reg: BackendRegistry[Any] = BackendRegistry("thing", "test.group")
        reg.register_builtin("core", lambda **kw: 1)
        reg.register("mine", lambda **kw: 2)
        reg._reset_for_tests()
        assert reg.names() == ["core"]


# =============================================================================
# Embedding registry helpers
# =============================================================================


class TestEmbeddingRegistry:
    def test_builtins_are_listed(self) -> None:
        names = available_embedding_backends()
        for builtin in ("openai", "fastembed", "sentence-transformers"):
            assert builtin in names

    def test_register_and_create_from_spec(self) -> None:
        register_embedding_backend("fake", _Backend)
        backend = create_embedding_backend("fake:my-model", dimension=3, extra=True)
        assert isinstance(backend, _Backend)
        assert backend.model_name == "my-model"
        assert backend.options == {"extra": True}
        assert unregister_embedding_backend("fake") is True

    def test_explicit_model_option_wins_over_spec(self) -> None:
        register_embedding_backend("fake", _Backend)
        backend = create_embedding_backend("fake:spec-model", model="option-model")
        assert backend.model_name == "option-model"

    def test_factory_returning_a_non_backend_is_rejected(self) -> None:
        register_embedding_backend("broken", lambda **kw: object())
        with pytest.raises(TypeError, match="does not implement EmbeddingBackend"):
            create_embedding_backend("broken")

    def test_backend_from_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _FakeEntryPoint("my-plugin", _Backend, group="headroom.embedding_backend")
        _patch_entry_points(monkeypatch, [ep])
        backend = create_embedding_backend("my-plugin:x")
        assert isinstance(backend, _Backend) and backend.model_name == "x"

    def test_as_embedding_backend(self) -> None:
        backend = _Backend()
        assert as_embedding_backend(backend) is backend
        with pytest.raises(TypeError):
            as_embedding_backend(backend, dimension=3)

        wrapped = as_embedding_backend(lambda texts: [[1.0, 2.0]] * len(texts), dimension=2)
        assert isinstance(wrapped, CallableEmbeddingBackend)
        assert isinstance(wrapped, EmbeddingBackend)
        assert wrapped.dimension == 2

        register_embedding_backend("fake", _Backend)
        assert isinstance(as_embedding_backend("fake"), _Backend)

        with pytest.raises(TypeError):
            as_embedding_backend(42)


# =============================================================================
# embed_texts
# =============================================================================


class TestEmbedTexts:
    def test_empty_texts_are_not_sent_and_come_back_as_zero_rows(self) -> None:
        backend = _Backend()
        out = embed_texts(backend, ["ab", "", "   ", "abcd"])
        assert backend.calls == [["ab", "abcd"]]
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [[2, 1, 0], [0, 0, 0], [0, 0, 0], [4, 1, 0]])

    def test_all_empty_uses_the_known_width(self) -> None:
        backend = _Backend(dimension=3)
        out = embed_texts(backend, ["", " "])
        assert backend.calls == []
        assert out.shape == (2, 3)
        assert embed_texts(CallableEmbeddingBackend(lambda t: []), [""]).shape == (1, 0)
        assert embed_texts(
            CallableEmbeddingBackend(lambda t: []), [""], expected_dimension=5
        ).shape == (
            1,
            5,
        )

    def test_chunks_by_max_batch_size(self) -> None:
        backend = CallableEmbeddingBackend(
            lambda texts: [[1.0, 0.0] for _ in texts], max_batch_size=2
        )
        calls: list[int] = []
        original = backend.embed

        def counting(texts: list[str]) -> Any:
            calls.append(len(texts))
            return original(texts)

        backend.embed = counting  # type: ignore[method-assign]
        out = embed_texts(backend, ["a", "b", "c", "d", "e"])
        assert calls == [2, 2, 1]
        assert out.shape == (5, 2)

    def test_normalize(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[3.0, 4.0] for _ in texts])
        out = embed_texts(backend, ["x", ""], normalize=True)
        np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]], rtol=1e-6)

    def test_accepts_numpy_output(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: np.ones((len(texts), 4), dtype=np.float64))
        out = embed_texts(backend, ["a", "b"])
        assert out.dtype == np.float32 and out.shape == (2, 4)

    def test_wrong_row_count(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[1.0, 2.0]])
        with pytest.raises(EmbeddingBackendResponseError, match="1 vectors for 2 texts"):
            embed_texts(backend, ["a", "b"])

    def test_declared_dimension_is_enforced(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[1.0, 2.0] for _ in texts], dimension=3)
        with pytest.raises(EmbeddingBackendResponseError, match="expected 3"):
            embed_texts(backend, ["a"])

    def test_expected_dimension_is_enforced(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[1.0, 2.0] for _ in texts])
        with pytest.raises(EmbeddingBackendResponseError, match="expected 4"):
            embed_texts(backend, ["a"], expected_dimension=4)
        declared = CallableEmbeddingBackend(lambda texts: [[1.0] for _ in texts], dimension=1)
        with pytest.raises(EmbeddingBackendResponseError, match="1-dimensional"):
            embed_texts(declared, ["a"], expected_dimension=4)

    def test_width_must_be_consistent_across_chunks(self) -> None:
        widths = iter([2, 3])
        backend = CallableEmbeddingBackend(
            lambda texts: [[1.0] * next(widths) for _ in texts], max_batch_size=1
        )
        with pytest.raises(EmbeddingBackendResponseError):
            embed_texts(backend, ["a", "b"])

    def test_ragged_rows(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[1.0, 2.0], [1.0]])
        with pytest.raises(EmbeddingBackendResponseError, match="one width"):
            embed_texts(backend, ["a", "b"])

    def test_one_dimensional_output(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: np.ones(3))
        with pytest.raises(EmbeddingBackendResponseError, match="2-D"):
            embed_texts(backend, ["a"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite(self, bad: float) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[1.0, bad] for _ in texts])
        with pytest.raises(EmbeddingBackendResponseError, match="non-finite"):
            embed_texts(backend, ["a"])

    def test_zero_width_rows_rejected(self) -> None:
        backend = CallableEmbeddingBackend(lambda texts: [[] for _ in texts])
        with pytest.raises(EmbeddingBackendResponseError):
            embed_texts(backend, ["a"])

    def test_third_party_embed_errors_are_reduced_to_their_class(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(texts: list[str]) -> Any:
            raise RuntimeError("failed for user text CONTENT-CANARY")

        with caplog.at_level("DEBUG"), pytest.raises(EmbeddingBackendCallError) as excinfo:
            embed_texts(CallableEmbeddingBackend(boom, name="plugin"), ["hello"])
        assert "CONTENT-CANARY" not in str(excinfo.value)
        assert "RuntimeError" in str(excinfo.value) and "plugin" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert all("CONTENT-CANARY" not in r.getMessage() for r in caplog.records)

    def test_third_party_backend_errors_are_reduced_even_when_backend_errors(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)

        class PluginError(BackendError):
            pass

        class Plugin(_Backend):
            name = "plugin"

            def embed(self, texts: list[str]) -> Any:
                raise PluginError("POST https://u:sk-CANARY@emb.example/v1 failed")

        def callable_fn(texts: list[str]) -> Any:
            raise EmbeddingBackendResponseError("bad rows for CONTENT-CANARY")

        class OpenAISubclass(OpenAICompatibleEmbeddingBackend):
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise PluginError("subclass of a built-in sk-CANARY")

        subclass = OpenAISubclass("m", base_url="http://127.0.0.1:9/v1")
        for backend in (Plugin(), CallableEmbeddingBackend(callable_fn, name="plugin"), subclass):
            with caplog.at_level("DEBUG"), pytest.raises(EmbeddingBackendCallError) as excinfo:
                embed_texts(backend, ["hello"])
            message = str(excinfo.value)
            assert "CANARY" not in message
            assert "failed (" in message
            assert isinstance(excinfo.value.__cause__, BackendError)
        assert all("CANARY" not in r.getMessage() for r in caplog.records)

    def test_untrusted_backend_names_are_not_echoed(self) -> None:
        backend = CallableEmbeddingBackend(lambda t: [[1.0]], name="Evil\nname", dimension=2)
        with pytest.raises(EmbeddingBackendResponseError) as excinfo:
            embed_texts(backend, ["a"])
        assert "Evil" not in str(excinfo.value)

    def test_non_string_input(self) -> None:
        with pytest.raises(TypeError):
            embed_texts(_Backend(), ["a", None])  # type: ignore[list-item]

    def test_callable_backend_validates_its_numbers(self) -> None:
        with pytest.raises(ValueError):
            CallableEmbeddingBackend(lambda t: t, dimension=0)
        with pytest.raises(ValueError):
            CallableEmbeddingBackend(lambda t: t, max_batch_size=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            CallableEmbeddingBackend(42)  # type: ignore[arg-type]


# =============================================================================
# Environment settings
# =============================================================================


class TestEnvSettings:
    def test_unset_is_none(self) -> None:
        assert embedding_backend_settings_from_env("HEADROOM_X", {}) is None
        assert (
            embedding_backend_settings_from_env("HEADROOM_X", {"HEADROOM_X_PROVIDER": " "}) is None
        )

    def test_full_family(self) -> None:
        settings = embedding_backend_settings_from_env(
            "HEADROOM_X",
            {
                "HEADROOM_X_PROVIDER": " OpenAI ",
                "HEADROOM_X_MODEL": "env-model",
                "HEADROOM_X_BASE_URL": "http://127.0.0.1:9/v1",
                "HEADROOM_X_API_KEY": "secret-key",
                "HEADROOM_X_DIMENSION": "1024",
                "HEADROOM_X_HEADERS": "X-Team=search, X-Env=prod",
                "HEADROOM_X_OPTIONS": json.dumps({"timeout": 5, "dimension": 8}),
            },
        )
        assert settings is not None
        assert settings.name == "openai"
        assert settings.model == "env-model"
        assert settings.dimension == 1024
        assert settings.source == "HEADROOM_X_PROVIDER"
        assert settings.identity() == "openai:env-model"
        assert settings.backend_options() == {
            "timeout": 5,
            "dimension": 1024,
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "secret-key",
            "headers": {"X-Team": "search", "X-Env": "prod"},
            "model": "env-model",
        }
        assert "secret-key" not in repr(settings)

    @pytest.mark.parametrize(
        ("env", "var"),
        [
            ({"P_PROVIDER": "bad name"}, "P_PROVIDER"),
            ({"P_PROVIDER": "openai:model"}, "P_MODEL"),
            ({"P_PROVIDER": "x", "P_DIMENSION": "abc"}, "P_DIMENSION"),
            ({"P_PROVIDER": "x", "P_DIMENSION": "0"}, "P_DIMENSION"),
            ({"P_PROVIDER": "x", "P_OPTIONS": "{not json"}, "P_OPTIONS"),
            ({"P_PROVIDER": "x", "P_OPTIONS": "[1]"}, "P_OPTIONS"),
            ({"P_PROVIDER": "x", "P_OPTIONS": '{"model": "m"}'}, "P_MODEL"),
            ({"P_PROVIDER": "x", "P_HEADERS": "novalue"}, "P_HEADERS"),
            ({"P_PROVIDER": "x", "P_HEADERS": "A=1,broken"}, "P_HEADERS"),
            ({"P_PROVIDER": "x", "P_HEADERS": "=v"}, "P_HEADERS"),
        ],
    )
    def test_invalid_values_name_the_variable(self, env: dict[str, str], var: str) -> None:
        with pytest.raises(ValueError, match=var):
            embedding_backend_settings_from_env("P", env)

    @pytest.mark.parametrize(
        "env",
        [
            {"P_PROVIDER": "x", "P_OPTIONS": '{"api_key": "sk-leak"'},
            {"P_PROVIDER": "x", "P_HEADERS": "Authorization:sk-leak"},
            {"P_PROVIDER": "sk leak"},
        ],
    )
    def test_secret_values_are_not_echoed(self, env: dict[str, str]) -> None:
        with pytest.raises(ValueError) as excinfo:
            embedding_backend_settings_from_env("P", env)
        assert "sk-leak" not in str(excinfo.value) and "sk leak" not in str(excinfo.value)

    def test_create_and_unknown_provider_names_the_variable(self) -> None:
        register_embedding_backend("fake", _Backend)
        settings = embedding_backend_settings_from_env(
            "P", {"P_PROVIDER": "fake", "P_MODEL": "m", "P_DIMENSION": "3"}
        )
        assert settings is not None
        backend = settings.create()
        assert isinstance(backend, _Backend)
        assert backend.model_name == "m" and backend.dimension == 3

        unknown = embedding_backend_settings_from_env("P", {"P_PROVIDER": "sk-pasted-key"})
        assert unknown is not None
        with pytest.raises(UnknownBackendError) as excinfo:
            unknown.validate()
        assert "P_PROVIDER" in str(excinfo.value)
        assert "sk-pasted-key" not in str(excinfo.value)


class TestMemoryAndRelevanceEnv:
    def test_memory_default_is_none(self) -> None:
        assert memory_embedder_settings_from_env({}) is None
        assert memory_embedder_settings_from_env({"HEADROOM_EMBEDDER_RUNTIME": "cpu"}) is None
        assert (
            memory_embedder_settings_from_env({"HEADROOM_EMBEDDER_RUNTIME": "pytorch_mps"}) is None
        )

    def test_runtime_openai_is_an_alias_for_the_openai_provider(self) -> None:
        env = {
            "HEADROOM_EMBEDDER_RUNTIME": "OpenAI",
            "HEADROOM_EMBEDDER_MODEL": "m",
            "HEADROOM_EMBEDDER_DIMENSION": "8",
            "HEADROOM_EMBEDDER_BASE_URL": "http://127.0.0.1:9/v1",
        }
        alias = memory_embedder_settings_from_env(env)
        explicit = memory_embedder_settings_from_env(
            {**env, "HEADROOM_EMBEDDER_RUNTIME": "", "HEADROOM_EMBEDDER_PROVIDER": "openai"}
        )
        assert alias is not None and explicit is not None
        assert alias.backend_options() == explicit.backend_options()
        assert alias.name == explicit.name == "openai"
        both = memory_embedder_settings_from_env({**env, "HEADROOM_EMBEDDER_PROVIDER": "openai"})
        assert both is not None and both.name == "openai"

    @pytest.mark.parametrize(
        "env",
        [
            {"HEADROOM_EMBEDDER_RUNTIME": "openai", "HEADROOM_EMBEDDER_PROVIDER": "plugin"},
            {"HEADROOM_EMBEDDER_RUNTIME": "pytorch_mps", "HEADROOM_EMBEDDER_PROVIDER": "plugin"},
        ],
    )
    def test_runtime_and_provider_contradictions(self, env: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_RUNTIME"):
            memory_embedder_settings_from_env(env)

    def test_runtime_cpu_combines_with_a_provider(self) -> None:
        settings = memory_embedder_settings_from_env(
            {"HEADROOM_EMBEDDER_RUNTIME": "cpu", "HEADROOM_EMBEDDER_PROVIDER": "plugin"}
        )
        assert settings is not None and settings.name == "plugin"

    def test_relevance_family(self) -> None:
        settings = relevance_embedder_settings_from_env(
            {
                "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "fastembed",
                "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "m",
            }
        )
        assert settings is not None and (settings.name, settings.model) == ("fastembed", "m")
        assert relevance_embedder_settings_from_env({}) is None

    def test_relevance_reuses_memory_settings(self) -> None:
        env = {
            "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "Memory",
            "HEADROOM_EMBEDDER_PROVIDER": "openai",
            "HEADROOM_EMBEDDER_MODEL": "shared",
            "HEADROOM_EMBEDDER_API_KEY": "k",
        }
        settings = relevance_embedder_settings_from_env(env)
        assert settings is not None
        assert (
            settings.backend_options() == memory_embedder_settings_from_env(env).backend_options()
        )  # type: ignore[union-attr]

    def test_relevance_memory_needs_a_memory_provider(self) -> None:
        with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_PROVIDER"):
            relevance_embedder_settings_from_env({"HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "memory"})

    @pytest.mark.parametrize(
        ("env", "var"),
        [
            (
                {"HEADROOM_EMBEDDER_PROVIDER": "openai", "HEADROOM_EMBEDDER_DIMENSION": "1536"},
                "HEADROOM_EMBEDDER_API_KEY",
            ),
            (
                {"HEADROOM_EMBEDDER_RUNTIME": "openai", "HEADROOM_EMBEDDER_DIMENSION": "1536"},
                "HEADROOM_EMBEDDER_API_KEY",
            ),
            (
                {
                    "HEADROOM_EMBEDDER_PROVIDER": "openai",
                    "HEADROOM_EMBEDDER_BASE_URL": "https://API.openai.com/v1/",
                    "HEADROOM_EMBEDDER_API_KEY": "  ",
                },
                "HEADROOM_EMBEDDER_API_KEY",
            ),
            (
                {
                    "HEADROOM_EMBEDDER_PROVIDER": "openai",
                    "HEADROOM_EMBEDDER_OPTIONS": '{"api_key": null}',
                },
                "HEADROOM_EMBEDDER_API_KEY",
            ),
            (
                {"HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "openai"},
                "HEADROOM_RELEVANCE_EMBEDDER_API_KEY",
            ),
            (
                {
                    "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "memory",
                    "HEADROOM_EMBEDDER_PROVIDER": "openai",
                },
                "HEADROOM_EMBEDDER_API_KEY",
            ),
        ],
    )
    def test_env_selected_openai_never_uses_the_chat_key(
        self, env: dict[str, str], var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proxy-chat-credential")
        read = (
            relevance_embedder_settings_from_env
            if "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER" in env
            else memory_embedder_settings_from_env
        )
        with pytest.raises(ValueError, match=var) as excinfo:
            read(env)
        assert "sk-proxy-chat-credential" not in str(excinfo.value)

    def test_env_selected_openai_with_its_own_key_or_another_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proxy-chat-credential")
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        own = memory_embedder_settings_from_env(
            {"HEADROOM_EMBEDDER_PROVIDER": "openai", "HEADROOM_EMBEDDER_API_KEY": "sk-embed"}
        )
        assert own is not None
        assert own.create()._api_key == "sk-embed"  # type: ignore[attr-defined]
        via_options = relevance_embedder_settings_from_env(
            {
                "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "openai",
                "HEADROOM_RELEVANCE_EMBEDDER_OPTIONS": '{"api_key": "sk-embed"}',
            }
        )
        assert via_options is not None
        local = memory_embedder_settings_from_env(
            {
                "HEADROOM_EMBEDDER_PROVIDER": "openai",
                "HEADROOM_EMBEDDER_MODEL": "m",
                "HEADROOM_EMBEDDER_BASE_URL": "http://127.0.0.1:9/v1",
            }
        )
        assert local is not None
        assert local.create()._api_key is None  # type: ignore[attr-defined]

    def test_contradictions_are_reported_before_a_missing_key(self) -> None:
        with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_RUNTIME"):
            memory_embedder_settings_from_env(
                {"HEADROOM_EMBEDDER_RUNTIME": "pytorch_mps", "HEADROOM_EMBEDDER_PROVIDER": "openai"}
            )

    def test_a_replaced_openai_backend_needs_no_key(self) -> None:
        register_embedding_backend("openai", _Backend, replace=True)
        settings = memory_embedder_settings_from_env({"HEADROOM_EMBEDDER_PROVIDER": "openai"})
        assert settings is not None and isinstance(settings.create(), _Backend)

    def test_relevance_memory_rejects_stray_companions(self) -> None:
        with pytest.raises(ValueError, match="HEADROOM_RELEVANCE_EMBEDDER_MODEL"):
            relevance_embedder_settings_from_env(
                {
                    "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "memory",
                    "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "x",
                    "HEADROOM_EMBEDDER_PROVIDER": "openai",
                }
            )


# =============================================================================
# OpenAI-compatible backend, against a real loopback HTTP server
# =============================================================================


@pytest.fixture
def embedding_server(monkeypatch: pytest.MonkeyPatch):
    """A loopback OpenAI-compatible /v1/embeddings server with scripted responses."""
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[dict[str, Any]] = []
    scripted: list[tuple[int, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "authorization_all": self.headers.get_all("Authorization"),
                    "content_type_all": self.headers.get_all("Content-Type"),
                    "x_team": self.headers.get("X-Team"),
                    "body": body,
                }
            )
            if scripted:
                status, payload = scripted.pop(0)
            else:
                status = 200
                # Reversed order on purpose: the backend must restore it.
                payload = {
                    "data": [
                        {"index": i, "embedding": [float(len(text)), 1.0, 0.5]}
                        for i, text in reversed(list(enumerate(body["input"])))
                    ]
                }
            raw = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen, scripted
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(
        "headroom.models.backends.openai_compat.time.sleep", lambda s: slept.append(s)
    )
    return slept


class TestOpenAICompatibleBackend:
    def test_round_trip_restores_order_and_sends_auth_and_headers(self, embedding_server) -> None:
        url, seen, _ = embedding_server
        backend = create_embedding_backend(
            "openai:my-embed",
            base_url=url + "/",
            api_key="test-key",
            dimension=3,
            headers={"X-Team": "search"},
        )
        try:
            out = embed_texts(backend, ["a", "", "abc"])
        finally:
            backend.close()  # type: ignore[attr-defined]
        np.testing.assert_allclose(out, [[1, 1, 0.5], [0, 0, 0], [3, 1, 0.5]])
        assert seen[0]["path"] == "/v1/embeddings"
        assert seen[0]["authorization"] == "Bearer test-key"
        assert seen[0]["x_team"] == "search"
        assert seen[0]["body"] == {
            "model": "my-embed",
            "input": ["a", "abc"],
            "encoding_format": "float",
        }

    def test_custom_headers_replace_defaults_case_insensitively(self, embedding_server) -> None:
        url, seen, _ = embedding_server
        backend = OpenAICompatibleEmbeddingBackend(
            "m",
            base_url=url,
            api_key="KEY1",
            headers={"authorization": "Basic OTHER", "content-type": "application/json"},
        )
        backend.embed(["x"])
        assert seen[0]["authorization_all"] == ["Basic OTHER"]
        assert seen[0]["content_type_all"] == ["application/json"]

    def test_no_key_means_no_auth_header(self, embedding_server) -> None:
        url, seen, _ = embedding_server
        backend = OpenAICompatibleEmbeddingBackend("m", base_url=url)
        backend.embed(["x"])
        assert seen[0]["authorization"] is None

    def test_openai_api_key_is_never_sent_to_another_host(
        self, embedding_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, _ = embedding_server
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-only")
        OpenAICompatibleEmbeddingBackend("m", base_url=url).embed(["x"])
        assert seen[0]["authorization"] is None

    def test_openai_base_url_env_is_ignored(
        self, embedding_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, _ = embedding_server
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        backend = OpenAICompatibleEmbeddingBackend()
        assert backend.base_url == "https://api.openai.com/v1"
        assert backend.model_name == "text-embedding-3-small"
        assert seen == []

    def test_openai_host_requires_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            OpenAICompatibleEmbeddingBackend()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert OpenAICompatibleEmbeddingBackend()._api_key == "sk-env"

    def test_request_dimensions(self, embedding_server) -> None:
        url, seen, _ = embedding_server
        backend = OpenAICompatibleEmbeddingBackend(
            "m", base_url=url, dimension=3, request_dimensions=True
        )
        backend.embed(["x"])
        assert seen[0]["body"]["dimensions"] == 3

    def test_responses_without_index_keep_list_order(self, embedding_server) -> None:
        url, _, scripted = embedding_server
        scripted.append((200, {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]}))
        rows = OpenAICompatibleEmbeddingBackend("m", base_url=url).embed(["a", "b"])
        assert rows == [[1.0, 0.0], [0.0, 1.0]]

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"nope": []}, "no 'data' list"),
            ({"data": [{"index": 0, "embedding": [1.0]}]}, "1 vectors for 2 inputs"),
            (
                {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},
                "exactly once",
            ),
            ({"data": [{"index": 0, "embedding": [1.0]}, {"embedding": [1.0]}]}, "some vectors"),
            (
                {"data": [{"index": 0, "embedding": "AAAA"}, {"index": 1, "embedding": "AAAA"}]},
                "base64",
            ),
        ],
    )
    def test_malformed_responses(self, embedding_server, payload: Any, match: str) -> None:
        url, _, scripted = embedding_server
        scripted.append((200, payload))
        with pytest.raises(EmbeddingBackendResponseError, match=match):
            OpenAICompatibleEmbeddingBackend("m", base_url=url).embed(["a", "b"])

    def test_retries_transient_status_then_succeeds(self, embedding_server, no_sleep) -> None:
        url, seen, scripted = embedding_server
        scripted.extend([(503, {"error": "busy"}), (429, {"error": "slow down"})])
        rows = OpenAICompatibleEmbeddingBackend("m", base_url=url, max_retries=2).embed(["ab"])
        assert rows == [[2.0, 1.0, 0.5]]
        assert len(seen) == 3
        assert len(no_sleep) == 2

    def test_gives_up_after_max_retries(self, embedding_server, no_sleep) -> None:
        url, seen, scripted = embedding_server
        scripted.extend([(503, {})] * 3)
        with pytest.raises(EmbeddingBackendRequestError, match="HTTP 503"):
            OpenAICompatibleEmbeddingBackend("m", base_url=url, max_retries=1).embed(["ab"])
        assert len(seen) == 2

    def test_client_errors_are_not_retried_and_hide_the_body(
        self, embedding_server, no_sleep, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, scripted = embedding_server
        body = {"error": {"message": "bad key sk-secret-123 for input CONTENT-CANARY"}}
        scripted.append((401, body))
        backend = OpenAICompatibleEmbeddingBackend("m", base_url=url, api_key="sk-secret-123")
        with pytest.raises(EmbeddingBackendRequestError) as excinfo:
            backend.embed(["x"])
        assert isinstance(excinfo.value, ConnectionError)
        assert len(seen) == 1 and no_sleep == []
        assert "HTTP 401" in str(excinfo.value)
        assert "CONTENT-CANARY" not in str(excinfo.value)
        assert "sk-secret-123" not in str(excinfo.value)

        # With content logging opted in, the API's message is shown, key redacted.
        monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
        scripted.append((401, body))
        with pytest.raises(EmbeddingBackendRequestError) as excinfo:
            backend.embed(["x"])
        assert "CONTENT-CANARY" in str(excinfo.value)
        assert "sk-secret-123" not in str(excinfo.value)
        assert "[redacted]" in str(excinfo.value)

    def test_request_errors_pass_through_embed_texts(self, embedding_server, no_sleep) -> None:
        url, _, scripted = embedding_server
        scripted.append((400, {}))
        backend = OpenAICompatibleEmbeddingBackend("m", base_url=url)
        with pytest.raises(EmbeddingBackendRequestError, match="HTTP 400"):
            embed_texts(backend, ["x"])

    def test_transport_error_retried_then_reported_without_secrets(self, no_sleep) -> None:
        backend = OpenAICompatibleEmbeddingBackend(
            "m", base_url="http://127.0.0.1:9/v1", api_key="sk-x", max_retries=1, timeout=0.5
        )
        with pytest.raises(ConnectionError) as excinfo:
            backend.embed(["x"])
        assert "127.0.0.1" in str(excinfo.value)
        assert "sk-x" not in str(excinfo.value)
        assert len(no_sleep) == 1

    def test_offline_refuses_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        with pytest.raises(OfflineEgressBlocked):
            OpenAICompatibleEmbeddingBackend("m", base_url="http://127.0.0.1:9/v1")
        with pytest.raises(OfflineEgressBlocked):
            create_embedding_backend("openai:m", base_url="http://127.0.0.1:9/v1")

    def test_offline_refuses_before_building_a_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        backend = OpenAICompatibleEmbeddingBackend("m", base_url="http://127.0.0.1:9/v1")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        built: list[Any] = []
        monkeypatch.setattr(
            "headroom.models.backends.openai_compat.httpx.Client",
            lambda *a, **k: built.append(1),
        )
        with pytest.raises(OfflineEgressBlocked):
            backend.embed(["x"])
        assert built == []

    def test_offline_flip_after_first_use_refuses_the_next_request(
        self, embedding_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, _ = embedding_server
        backend = OpenAICompatibleEmbeddingBackend("m", base_url=url)
        backend.embed(["x"])
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        with pytest.raises(OfflineEgressBlocked):
            backend.embed(["y"])
        assert len(seen) == 1

    @pytest.mark.parametrize(
        "base_url",
        [
            "ftp://example.com/v1",
            "http:///v1",
            "https://user:pw@example.com/v1",
            "https://example.com/v1?x=1",
            "https://example.com/v1#frag",
        ],
    )
    def test_rejects_unsafe_base_urls(self, base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleEmbeddingBackend("m", base_url=base_url)

    @pytest.mark.parametrize(
        "base_url",
        ["http://127.0.0.1:80a0/v1", "http://127.0.0.1:99999/v1", "http://127.0.0.1:-1/v1"],
    )
    def test_rejects_a_bad_port_at_construction(
        self, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        with pytest.raises(ValueError, match="base_url") as excinfo:
            OpenAICompatibleEmbeddingBackend("m", base_url=base_url)
        assert base_url not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("kwargs", "option", "secret"),
        [
            ({"headers": {"X Api Key": "abc"}}, "headers", "X Api Key"),
            ({"headers": {"X:Y": "abc"}}, "headers", "X:Y"),
            ({"headers": {"": "abc"}}, "headers", None),
            ({"headers": {"X-Api-Key": "cl\u00e9-CANARY"}}, "headers", "CANARY"),
            ({"headers": {"X-Api-Key": " CANARY"}}, "headers", "CANARY"),
            ({"headers": {"X-Api-Key": "CANARY\t"}}, "headers", "CANARY"),
            ({"headers": {"X-Api-Key": "a\x01CANARY"}}, "headers", "CANARY"),
            ({"headers": {"X-A": "1", "x-a": "2"}}, "headers", None),
            ({"api_key": "sk-cl\u00e9-CANARY"}, "api_key", "CANARY"),
            ({"api_key": "sk-\x01CANARY"}, "api_key", "CANARY"),
        ],
    )
    def test_rejects_what_httpx_would_refuse_at_construction(
        self,
        kwargs: dict[str, Any],
        option: str,
        secret: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        with pytest.raises(ValueError, match=option) as excinfo:
            OpenAICompatibleEmbeddingBackend("m", base_url="http://127.0.0.1:9/v1", **kwargs)
        if secret is not None:
            assert secret not in str(excinfo.value)

    def test_valid_headers_still_work(self, embedding_server) -> None:
        url, seen, _ = embedding_server
        backend = OpenAICompatibleEmbeddingBackend(
            "m", base_url=url, headers={"X-Team": "search team", "X-Empty": ""}
        )
        backend.embed(["x"])
        assert seen[0]["x_team"] == "search team"

    def test_bad_env_headers_fail_the_startup_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        settings = memory_embedder_settings_from_env(
            {
                "HEADROOM_EMBEDDER_PROVIDER": "openai",
                "HEADROOM_EMBEDDER_MODEL": "m",
                "HEADROOM_EMBEDDER_BASE_URL": "http://127.0.0.1:9/v1",
                "HEADROOM_EMBEDDER_HEADERS": "X-Api-Key=cl\u00e9-CANARY",
            }
        )
        assert settings is not None
        with pytest.raises(BackendLoadError, match="headers") as excinfo:
            settings.create()
        assert "CANARY" not in str(excinfo.value)

    def test_model_required_off_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        with pytest.raises(ValueError, match="model is required"):
            OpenAICompatibleEmbeddingBackend(base_url="http://127.0.0.1:9/v1")
        default = OpenAICompatibleEmbeddingBackend(
            base_url="https://api.openai.com/v1", api_key="sk-test"
        )
        assert default.model_name == "text-embedding-3-small"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timeout": 0},
            {"timeout": True},
            {"max_retries": -1},
            {"dimension": 0},
            {"request_dimensions": True},
            {"headers": {"X": "a\r\nb"}},
            {"max_batch_size": 0},
        ],
    )
    def test_invalid_options(self, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        with pytest.raises(ValueError):
            OpenAICompatibleEmbeddingBackend("m", base_url="http://127.0.0.1:9/v1", **kwargs)

    def test_repr_has_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        backend = OpenAICompatibleEmbeddingBackend(
            "m", base_url="http://127.0.0.1:9/v1", api_key="sk-hidden"
        )
        assert "sk-hidden" not in repr(backend)


# =============================================================================
# sentence-transformers backend (model load stubbed)
# =============================================================================


class TestSentenceTransformerBackend:
    def test_uses_the_shared_registry_and_learns_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeModel:
            max_seq_length = 128

            def get_sentence_embedding_dimension(self) -> int:
                return 2

            def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
                assert kwargs["normalize_embeddings"] is False
                return np.array([[1.0, float(len(t))] for t in texts])

        requested: list[tuple[Any, Any]] = []

        def fake_get(model_name: str | None = None, device: str | None = None) -> FakeModel:
            requested.append((model_name, device))
            return FakeModel()

        monkeypatch.setattr(
            "headroom.models.ml_models.MLModelRegistry.get_sentence_transformer", fake_get
        )
        backend = create_embedding_backend("sentence-transformers:my-st", device="cpu")
        assert backend.dimension is None
        out = embed_texts(backend, ["abc"])
        np.testing.assert_allclose(out, [[1.0, 3.0]])
        assert backend.dimension == 2
        assert backend.max_tokens == 128  # type: ignore[attr-defined]
        assert requested == [("my-st", "cpu")]

    def test_dimension_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeModel:
            def get_sentence_embedding_dimension(self) -> int:
                return 384

        monkeypatch.setattr(
            "headroom.models.ml_models.MLModelRegistry.get_sentence_transformer",
            lambda *a, **k: FakeModel(),
        )
        backend = create_embedding_backend("sentence-transformers:secret-path", dimension=768)
        with pytest.raises(ValueError, match="384"):
            backend.embed(["x"])
        # Through embed_texts the built-in's explanation survives, not just a class.
        with pytest.raises(EmbeddingBackendResponseError, match="384-dimensional") as excinfo:
            embed_texts(backend, ["x"])
        assert "768" in str(excinfo.value)
        assert "secret-path" not in str(excinfo.value)

    def test_missing_package_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(BackendLoadError, match=r"headroom-ai\[memory\]"):
            create_embedding_backend("sentence-transformers")
        settings = memory_embedder_settings_from_env(
            {"HEADROOM_EMBEDDER_PROVIDER": "sentence-transformers"}
        )
        assert settings is not None
        with pytest.raises(BackendLoadError, match="requires sentence-transformers"):
            settings.create()

    def test_package_check_does_not_import_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.models.backends import sentence_transformer_backend as st

        monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
        monkeypatch.setattr(st.importlib.util, "find_spec", lambda name: object())
        st.SentenceTransformerBackend()
        assert "sentence_transformers" not in sys.modules
        monkeypatch.setattr(st.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(RuntimeError, match="requires sentence-transformers"):
            st.SentenceTransformerBackend()
        # A stub module put in place by a test counts as installed.
        monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace())
        st.SentenceTransformerBackend()
