"""Pluggable Kompress backends (``HEADROOM_KOMPRESS_PROVIDER``).

A backend supplies text in, text out. Everything Headroom promises around the
model has to survive the swap, so most of these tests are about what the
plugin does NOT get to decide: the word floor, the request ML budget, fail-open
pass-through, the proxy-local CCR store and its marker gate, the closed
``/health`` vocabulary, and the router's refusal to fall back to the local model
when the configured provider cannot be built.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import inspect
import json
import logging
import random
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from headroom.log_safety import WarnOnce
from headroom.models.backends import BackendError, BackendLoadError, UnknownBackendError
from headroom.models.backends import registry as registry_module
from headroom.offline import OfflineEgressBlocked, guard_egress
from headroom.transforms import kompress_remote
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _config_namespace,
)
from headroom.transforms.kompress_backend import (
    KOMPRESS_BACKENDS,
    KOMPRESS_ENTRY_POINT_GROUP,
    KOMPRESS_PROVIDER_ENV,
    KOMPRESS_PROVIDER_OPTIONS_ENV,
    KompressBackendCallError,
    KompressBackendResponseError,
    KompressModelBackend,
    available_kompress_backends,
    call_kompress_backend,
    create_kompress_backend,
    kompress_provider_settings_from_env,
    register_kompress_backend,
    unregister_kompress_backend,
    validate_kompress_provider_env,
)
from headroom.transforms.kompress_compressor import KompressConfig
from headroom.transforms.kompress_remote import (
    _CAPPED_CALL_WORKERS,
    KOMPRESS_ENDPOINT_ENV,
    ProviderKompressCompressor,
    RemoteKompressCompressor,
)

REMOTE_LOGGER = "headroom.transforms.kompress_remote"
LONG = " ".join(f"word{i}" for i in range(120))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    for var in (
        KOMPRESS_PROVIDER_ENV,
        KOMPRESS_PROVIDER_OPTIONS_ENV,
        KOMPRESS_ENDPOINT_ENV,
        "HEADROOM_OFFLINE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        kompress_remote, "_PROVIDER_FAILURE_WARNINGS", WarnOnce(64, "failing Kompress providers")
    )
    KOMPRESS_BACKENDS._reset_for_tests()
    kompress_remote._reset_provider_selection_for_tests()
    yield
    KOMPRESS_BACKENDS._reset_for_tests()
    kompress_remote._reset_provider_selection_for_tests()


class _Halver:
    """Keeps every other word. Records every call and the thread it ran on."""

    name = "halver"
    instances: list[_Halver] = []

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.calls: list[tuple[str, float | None, str | None]] = []
        self.threads: list[str] = []
        self.closed = 0
        _Halver.instances.append(self)

    def compress_text(
        self, content: str, *, target_ratio: float | None, question: str | None
    ) -> str:
        self.calls.append((content, target_ratio, question))
        self.threads.append(threading.current_thread().name)
        return " ".join(content.split()[::2])

    def close(self) -> None:
        self.closed += 1


class _Returns:
    """Returns a fixed value, or raises it when it is an exception."""

    name = "returns"

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def compress_text(
        self, content: str, *, target_ratio: float | None, question: str | None
    ) -> Any:
        self.calls += 1
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _Blocking:
    """Blocks every call until released, like a hung model server."""

    name = "blocking"

    def __init__(self) -> None:
        self.release = threading.Event()
        self.reached: list[str] = []
        self._lock = threading.Lock()

    def compress_text(
        self, content: str, *, target_ratio: float | None, question: str | None
    ) -> str:
        with self._lock:
            self.reached.append(content)
        self.release.wait(5.0)
        return " ".join(content.split()[::2])


@pytest.fixture
def halver() -> type[_Halver]:
    _Halver.instances = []
    register_kompress_backend("halver", _Halver)
    return _Halver


def _provider(backend: Any, **config: Any) -> ProviderKompressCompressor:
    return ProviderKompressCompressor(backend, provider="test", config=KompressConfig(**config))


def _free_permits(compressor: Any) -> int:
    return compressor._capped_slots._value  # type: ignore[attr-defined]


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    ends = time.monotonic() + timeout_s
    while time.monotonic() < ends:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeEntryPoint:
    def __init__(self, name: str, value: Any, group: str = KOMPRESS_ENTRY_POINT_GROUP) -> None:
        self.name = name
        self.value = f"fake.module:{name}"
        self.group = group
        self._target = value
        self.loads = 0

    def load(self) -> Any:
        self.loads += 1
        return self._target


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]) -> None:
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: [ep for ep in eps if ep.group == group],
    )


# =============================================================================
# Registry and contract
# =============================================================================


class TestRegistry:
    def test_there_are_no_builtins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [])
        assert available_kompress_backends() == []
        assert KOMPRESS_BACKENDS.entry_point_group == "headroom.kompress_backend"

    def test_register_create_and_unregister(self) -> None:
        register_kompress_backend("halver", _Halver)
        backend = create_kompress_backend("halver", endpoint="x", level=2)
        assert isinstance(backend, _Halver)
        assert backend.options == {"endpoint": "x", "level": 2}
        assert isinstance(backend, KompressModelBackend)
        with pytest.raises(ValueError, match="already registered"):
            register_kompress_backend("halver", _Halver)
        assert unregister_kompress_backend("halver") is True
        with pytest.raises(UnknownBackendError):
            create_kompress_backend("halver")

    def test_factory_returning_a_non_backend_is_rejected(self) -> None:
        register_kompress_backend("broken", lambda **kw: object())
        with pytest.raises(TypeError, match="KompressModelBackend"):
            create_kompress_backend("broken")

    def test_entry_point_is_loaded_only_when_selected(self, monkeypatch) -> None:
        wanted = _FakeEntryPoint("from-plugin", _Halver)
        other = _FakeEntryPoint("other", _Halver)
        unrelated = _FakeEntryPoint("from-plugin", object, group="headroom.embedding_backend")
        _patch_entry_points(monkeypatch, [wanted, other, unrelated])

        assert {"from-plugin", "other"} <= set(available_kompress_backends())
        assert wanted.loads == other.loads == 0
        assert isinstance(create_kompress_backend("from-plugin"), _Halver)
        assert (wanted.loads, other.loads, unrelated.loads) == (1, 0, 0)

    def test_entry_point_selected_through_the_router(self, monkeypatch) -> None:
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("from-plugin", _Halver)])
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "from-plugin")

        compressor = ContentRouter(ContentRouterConfig())._get_kompress()

        assert isinstance(compressor, ProviderKompressCompressor)
        assert isinstance(compressor.backend, _Halver)


class TestCallKompressBackend:
    def _call(self, value: Any) -> dict[str, Any]:
        return call_kompress_backend(
            _Returns(value), LONG, target_ratio=None, question=None, label="test"
        )

    def test_a_string_is_the_compressed_text(self) -> None:
        assert self._call("short") == {"compressed": "short"}

    def test_a_mapping_keeps_known_checked_fields_and_drops_none(self) -> None:
        fields = self._call(
            {
                "compressed": "short",
                "original_tokens": 120,
                "compressed_tokens": 1,
                "compression_ratio": 1,
                "model_used": "m",
                "unknown": "ignored",
                "extra": None,
            }
        )
        assert fields == {
            "compressed": "short",
            "original_tokens": 120,
            "compressed_tokens": 1,
            "compression_ratio": 1.0,
            "model_used": "m",
        }
        assert self._call({"compressed": "s", "original_tokens": None}) == {"compressed": "s"}

    @pytest.mark.parametrize(
        "value",
        [
            42,
            None,
            ["short"],
            {},
            {"compressed": 7},
            {"compressed": ""},
            {"compressed": "  \n "},
            {"compressed": "s", "original_tokens": True},
            {"compressed": "s", "original_tokens": -1},
            {"compressed": "s", "compressed_tokens": 1.5},
            {"compressed": "s", "compression_ratio": float("nan")},
            {"compressed": "s", "compression_ratio": "0.5"},
            {"compressed": "s", "model_used": 3},
        ],
    )
    def test_malformed_results_are_rejected(self, value: Any) -> None:
        with pytest.raises(KompressBackendResponseError):
            self._call(value)

    def test_backend_errors_are_reduced_to_their_class(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        with pytest.raises(KompressBackendCallError) as excinfo:
            self._call(RuntimeError("upstream said sk-live-secret"))
        assert "RuntimeError" in str(excinfo.value)
        assert "sk-live-secret" not in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "sk-live-secret" not in caplog.text

    def test_offline_refusal_is_not_wrapped(self) -> None:
        refusal = OfflineEgressBlocked("plugin", "https://k.example.com")
        with pytest.raises(OfflineEgressBlocked):
            self._call(refusal)


# =============================================================================
# Environment selection
# =============================================================================


class TestEnv:
    def test_unset_or_blank_is_none(self) -> None:
        assert kompress_provider_settings_from_env({}) is None
        assert kompress_provider_settings_from_env({KOMPRESS_PROVIDER_ENV: "  "}) is None
        # The options variable alone selects nothing.
        assert kompress_provider_settings_from_env({KOMPRESS_PROVIDER_OPTIONS_ENV: "{}"}) is None
        assert validate_kompress_provider_env({}) is None

    def test_name_and_options(self) -> None:
        settings = kompress_provider_settings_from_env(
            {
                KOMPRESS_PROVIDER_ENV: " My-Plugin ",
                KOMPRESS_PROVIDER_OPTIONS_ENV: '{"base_url": "http://x", "level": 3}',
            }
        )
        assert settings == ("my-plugin", {"base_url": "http://x", "level": 3})

    def test_kompress_backend_is_not_the_selector(self) -> None:
        """HEADROOM_KOMPRESS_BACKEND already picks the ONNX/PyTorch runtime."""
        assert KOMPRESS_PROVIDER_ENV == "HEADROOM_KOMPRESS_PROVIDER"
        assert kompress_provider_settings_from_env({"HEADROOM_KOMPRESS_BACKEND": "onnx"}) is None

    @pytest.mark.parametrize(
        ("env", "var"),
        [
            ({KOMPRESS_PROVIDER_ENV: "sk live key"}, KOMPRESS_PROVIDER_ENV),
            ({KOMPRESS_PROVIDER_ENV: "plugin:model"}, KOMPRESS_PROVIDER_ENV),
            (
                {
                    KOMPRESS_PROVIDER_ENV: "x",
                    KOMPRESS_PROVIDER_OPTIONS_ENV: '{"api_key": "sk-leak"',
                },
                KOMPRESS_PROVIDER_OPTIONS_ENV,
            ),
            (
                {KOMPRESS_PROVIDER_ENV: "x", KOMPRESS_PROVIDER_OPTIONS_ENV: '["sk-leak"]'},
                KOMPRESS_PROVIDER_OPTIONS_ENV,
            ),
        ],
    )
    def test_malformed_values_name_the_variable_not_the_value(
        self, env: dict[str, str], var: str
    ) -> None:
        with pytest.raises(ValueError, match=var) as excinfo:
            kompress_provider_settings_from_env(env)
        assert "sk-leak" not in str(excinfo.value)
        assert "sk live key" not in str(excinfo.value)
        with pytest.raises(ValueError, match=var):
            validate_kompress_provider_env(env)

    def test_validate_resolves_without_constructing(self) -> None:
        built: list[dict[str, Any]] = []
        register_kompress_backend("fake", lambda **kw: built.append(kw) or _Halver())
        env = {KOMPRESS_PROVIDER_ENV: "fake", KOMPRESS_PROVIDER_OPTIONS_ENV: '{"a": 1}'}

        assert validate_kompress_provider_env(env) == "fake"
        assert built == []

    def test_validate_unknown_names_the_variable_not_the_value(self) -> None:
        with pytest.raises(UnknownBackendError) as excinfo:
            validate_kompress_provider_env({KOMPRESS_PROVIDER_ENV: "sk-pasted-key"})
        assert KOMPRESS_PROVIDER_ENV in str(excinfo.value)
        assert "sk-pasted-key" not in str(excinfo.value)

    def test_validate_reports_an_unimportable_factory(self) -> None:
        register_kompress_backend("lazy", "headroom_no_such_module.x:Factory")
        with pytest.raises(BackendLoadError):
            validate_kompress_provider_env({KOMPRESS_PROVIDER_ENV: "lazy"})


# =============================================================================
# ProviderKompressCompressor: the shared post-processing
# =============================================================================


class TestProviderCompressor:
    def test_short_blocks_never_reach_the_backend(self) -> None:
        backend = _Halver()
        compressor = _provider(backend)
        short = " ".join(["word"] * 63)  # under min_input_words=64

        result = compressor.compress(short)

        assert result.compressed == short
        assert backend.calls == []

    def test_compresses_and_forwards_target_ratio_and_question(self) -> None:
        backend = _Halver()
        compressor = _provider(backend, enable_ccr=False)

        result = compressor.compress(LONG, question="which word?", target_ratio=0.3)

        assert result.compressed == " ".join(LONG.split()[::2])
        assert result.original_tokens == 120
        assert result.compressed_tokens == 60
        assert backend.calls == [(LONG, 0.3, "which word?")]
        compressor.close()
        assert backend.closed == 1

    def test_mapping_metadata_is_reported(self) -> None:
        backend = _Returns(
            {
                "compressed": "short result",
                "original_tokens": 20,
                "compressed_tokens": 2,
                "compression_ratio": 0.1,
                "model_used": "plugin-model",
            }
        )
        result = _provider(backend, enable_ccr=False).compress(LONG)
        assert (
            result.compressed,
            result.original_tokens,
            result.compressed_tokens,
            result.compression_ratio,
            result.model_used,
        ) == ("short result", 20, 2, 0.1, "plugin-model")

    def test_health_label_is_the_fixed_literal_never_the_plugins_name(self) -> None:
        backend = _Halver()
        backend.name = "Pwned\r\nX-Injected: 1"  # type: ignore[misc]
        compressor = ProviderKompressCompressor(backend)

        assert compressor.is_ready() is True
        assert compressor.ready_backend() == "custom"
        assert compressor.preload() == "custom"
        assert compressor.ensure_background_load() is None
        # Log lines fall back to the class name for a non-identifier name.
        assert "Pwned" not in compressor.provider

    def test_rejects_a_non_backend_and_a_bad_timeout(self) -> None:
        with pytest.raises(TypeError):
            ProviderKompressCompressor(object())  # type: ignore[arg-type]
        for bad in (0, -1, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                ProviderKompressCompressor(_Halver(), timeout=bad)

    def test_accepts_every_keyword_the_router_passes(self) -> None:
        """The drop-in surface: the router calls every Kompress through one seam."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        def _kwargs(fn: Any) -> set[str]:
            return {
                name
                for name, p in inspect.signature(fn).parameters.items()
                if name != "self"
                and not name.startswith("_")
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }

        missing = _kwargs(KompressCompressor.compress) - _kwargs(
            ProviderKompressCompressor.compress
        )
        assert not missing
        assert ProviderKompressCompressor.accepts_time_budget_cap is True
        assert not getattr(ProviderKompressCompressor, "shares_request_deadline", False)


class TestFailOpen:
    @pytest.mark.parametrize(
        "value",
        [
            RuntimeError("upstream said sk-live-secret"),
            KeyError("sk-live-secret"),
            {"compressed": None},
            ["sk-live-secret"],
            "",
        ],
    )
    def test_any_failure_passes_the_block_through(self, value: Any, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger=REMOTE_LOGGER)
        backend = _Returns(value)

        result = _provider(backend).compress(LONG)

        assert result.compressed == LONG
        assert result.cache_key is None
        assert result.compression_ratio == 1.0
        assert backend.calls == 1
        assert "sk-live-secret" not in caplog.text

    def test_failures_warn_once_by_class_and_detail_at_debug(self, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger=REMOTE_LOGGER)
        compressor = _provider(_Returns(ConnectionError("https://user:sk-live@host refused")))

        for _ in range(3):
            assert compressor.compress(LONG).compressed == LONG

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "a backend failing on every block must warn once"
        assert "ConnectionError" in warnings[0].getMessage()
        assert "passing through" in warnings[0].getMessage()
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("ConnectionError at" in line for line in debug), debug
        assert "sk-live" not in caplog.text

    def test_a_different_failure_still_warns(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger=REMOTE_LOGGER)
        backend = _Returns(ConnectionError("x"))
        compressor = _provider(backend)
        compressor.compress(LONG)
        backend.value = 42  # now a malformed result
        compressor.compress(LONG)
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    def test_an_air_gap_refusal_from_the_backend_propagates(self) -> None:
        """Fail-open must not turn a refusal into "compression just stopped"."""
        compressor = _provider(_Returns(OfflineEgressBlocked("plugin", "https://k.example.com")))
        with pytest.raises(OfflineEgressBlocked):
            compressor.compress(LONG)
        # The worker gave its permit back on the way out.
        assert _wait_until(lambda: _free_permits(compressor) == _CAPPED_CALL_WORKERS)
        compressor.close()


class _PluginError(BackendError):
    """A plugin's own error class: Headroom's public base, the plugin's message."""


_CALL_SECRET = "POST https://api.example.com/v1?key=sk-SECRET-123 failed: 401"
_FACTORY_SECRET = "could not reach https://u:sk-SECRET-456@host/"


class TestPluginErrorsAreNeverLogged:
    """``BackendError`` is public, so a plugin may raise a subclass of it (or one
    of Headroom's own classes). Its message is still the plugin's: it is never
    logged, whatever the class, from ``compress_text`` or from the factory."""

    @pytest.mark.parametrize(
        "error_class", [_PluginError, BackendLoadError, KompressBackendCallError]
    )
    def test_from_compress_text(self, caplog, error_class: type[BackendError]) -> None:
        caplog.set_level(logging.DEBUG)
        backend = _Returns(error_class(_CALL_SECRET))

        with pytest.raises(KompressBackendCallError) as excinfo:
            call_kompress_backend(backend, LONG, target_ratio=None, question=None, label="t")
        assert "sk-SECRET-123" not in str(excinfo.value)
        assert error_class.__name__ in str(excinfo.value)
        assert excinfo.value.__cause__ is backend.value

        compressor = _provider(backend)
        assert compressor.compress(LONG).compressed == LONG
        compressor.close()

        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert any("passing through" in line for line in loud), loud
        assert not [line for line in loud if "sk-SECRET" in line]
        assert "sk-SECRET" not in caplog.text

    @pytest.mark.parametrize("error_class", [_PluginError, BackendLoadError, UnknownBackendError])
    def test_from_the_factory(self, monkeypatch, caplog, error_class: type[BackendError]) -> None:
        def factory(**_options: Any) -> Any:
            raise error_class(_FACTORY_SECRET)

        register_kompress_backend("leaky", factory)
        caplog.set_level(logging.DEBUG)

        with pytest.raises(BackendLoadError) as excinfo:
            create_kompress_backend("leaky")
        assert type(excinfo.value) is BackendLoadError
        assert "sk-SECRET-456" not in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, error_class)

        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "leaky")
        assert ContentRouter(ContentRouterConfig())._get_kompress() is None

        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
        assert not [line for line in loud if "sk-SECRET" in line]
        assert "sk-SECRET" not in caplog.text

    def test_from_the_factory_at_proxy_startup(self, monkeypatch, caplog, capsys) -> None:
        from headroom.proxy import server
        from headroom.proxy.models import ProxyConfig

        def factory(**_options: Any) -> Any:
            raise _PluginError(_FACTORY_SECRET)

        register_kompress_backend("leaky", factory)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "leaky")
        caplog.set_level(logging.DEBUG)

        with pytest.raises(SystemExit) as excinfo:
            server._preflight_model_backends(ProxyConfig(optimize=True, memory_enabled=False))

        assert excinfo.value.code == 78
        assert "sk-SECRET" not in capsys.readouterr().err
        assert "sk-SECRET" not in caplog.text


# =============================================================================
# CCR: marker, gate and ccr_original, at parity with remote Kompress
# =============================================================================


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def post(self, url, headers=None, json=None):  # noqa: A002, ANN001
        return _FakeResponse(self._payload)

    def close(self) -> None:
        return None


def _store(original: str, compressed: str, original_tokens: int) -> str:
    return hashlib.sha256(original.encode()).hexdigest()[:24]


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []

    def _fake_store(original: str, compressed: str, original_tokens: int) -> str:
        seen.append((original, original_tokens))
        return _store(original, compressed, original_tokens)

    monkeypatch.setattr(kompress_remote, "store_kompress_in_ccr", _fake_store)
    return seen


def _pair(payload: dict, *, enable_ccr: bool) -> tuple[Any, Any]:
    remote = RemoteKompressCompressor("https://ml.example.invalid")
    remote._client = _FakeClient(payload)  # type: ignore[assignment]
    remote.config.enable_ccr = enable_ccr
    remote.config.min_input_words = 10
    provider = ProviderKompressCompressor(
        _Returns(dict(payload)),
        config=KompressConfig(enable_ccr=enable_ccr, min_input_words=10),
        timeout=None,
    )
    return remote, provider


_SMALL = " ".join(["alpha"] * 99 + ["nfs"])
_RETOKENIZED = " ".join(["alpha"] * 36 + ["bureaucratic"] + ["alpha"] * 63)
_BIG = " ".join(["alpha"] * 299 + ["nfs"])


class TestCcrParity:
    @pytest.mark.parametrize("enable_ccr", [True, False])
    @pytest.mark.parametrize(
        ("source", "drop", "extra"),
        [
            (_SMALL, 41, {}),
            (_RETOKENIZED, 36, {}),
            (_BIG, 60, {}),
            (_BIG, 60, {"original_tokens": 999, "compressed_tokens": 5}),
            (_BIG, 0, {}),  # unchanged text: nothing to store
        ],
    )
    def test_results_match_remote_kompress_field_for_field(
        self, stored, source: str, drop: int, extra: dict, enable_ccr: bool
    ) -> None:
        """Whatever the tokenizer, the provider must give exactly what the remote
        endpoint gives for the same model output: same marker, same gate, same
        accounting."""
        payload = {"compressed": " ".join(source.split()[drop:]), **extra}
        remote, provider = _pair(payload, enable_ccr=enable_ccr)

        from_remote = remote.compress(source)
        from_provider = provider.compress(source)

        assert dataclasses.asdict(from_provider) == dataclasses.asdict(from_remote)

    def test_ccr_original_is_stored_at_parity(self, stored) -> None:
        original = "real secret block " * 60
        placeholder = "{{HEADROOM_TAG_0}} " * 60
        remote, provider = _pair({"compressed": "short"}, enable_ccr=True)

        from_remote = remote.compress(placeholder, ccr_original=original)
        from_provider = provider.compress(placeholder, ccr_original=original)

        assert dataclasses.asdict(from_provider) == dataclasses.asdict(from_remote)
        assert stored[0] == stored[1] == (original, len(original.split()))


class TestCcrPolicy:
    """Concrete outcomes, with a word-count tokenizer so they hold without tiktoken."""

    @pytest.fixture(autouse=True)
    def _word_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kompress_remote, "payload_tokens", lambda text: len(text.split()))

    def test_a_saving_result_is_stored_and_marked(self, stored) -> None:
        source = " ".join(f"w{i}" for i in range(100))
        kept = " ".join(source.split()[:50])
        result = _provider(_Returns(kept)).compress(source)

        key = _store(source, kept, 100)
        assert result.cache_key == key
        assert result.compressed.startswith(kept)
        assert f"Retrieve more: hash={key}]" in result.compressed
        assert stored == [(source, 100)]
        assert result.original_tokens == 100
        assert result.compressed_tokens == len(result.compressed.split()) < 100

    def test_a_result_that_cannot_pay_for_its_marker_passes_through(self, stored) -> None:
        source = " ".join(f"w{i}" for i in range(100))
        kept = " ".join(source.split()[:95])
        result = _provider(_Returns(kept)).compress(source)

        assert result.compressed == source
        assert result.cache_key is None

    def test_no_ccr_mode_never_stores_or_marks(self, stored) -> None:
        source = " ".join(f"w{i}" for i in range(100))
        kept = " ".join(source.split()[:50])
        result = _provider(_Returns(kept), enable_ccr=False).compress(source)

        assert result.compressed == kept
        assert result.cache_key is None
        assert stored == []

    def test_tag_protected_content_stores_the_real_text(self, stored) -> None:
        original = "real secret block " * 60
        placeholder = "{{HEADROOM_TAG_0}} " * 60
        # 60 words: under the production floor (64), so lower it to the clamp.
        result = _provider(_Returns("short"), min_input_words=10).compress(
            placeholder, ccr_original=original
        )

        assert stored == [(original, len(original.split()))]
        assert "HEADROOM_TAG" not in stored[0][0]
        assert result.cache_key is not None


# =============================================================================
# The request ML budget
# =============================================================================


class TestBudget:
    def test_a_call_that_cannot_fit_the_budget_is_never_made(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger=REMOTE_LOGGER)
        backend = _Halver()
        result = _provider(backend).compress(LONG, _time_budget_cap_seconds=0.01)

        assert result.compressed == LONG
        assert backend.calls == []
        assert "Kompress provider declined" in caplog.text

    def test_the_call_is_capped_and_its_permit_held_until_it_returns(self) -> None:
        backend = _Blocking()
        compressor = _provider(backend)
        try:
            started = time.monotonic()
            result = compressor.compress(LONG, _time_budget_cap_seconds=0.1)
            took = time.monotonic() - started

            assert result.compressed == LONG, "an over-budget call passes through"
            assert took < 0.1 + 0.15, f"took {took:.3f}s against a 100 ms cap"
            assert backend.reached == [LONG]
            assert _free_permits(compressor) == _CAPPED_CALL_WORKERS - 1
        finally:
            backend.release.set()
        assert _wait_until(lambda: _free_permits(compressor) == _CAPPED_CALL_WORKERS)
        compressor.close()

    def test_saturated_calls_fail_open_without_queueing_their_payloads(self) -> None:
        backend = _Blocking()
        compressor = _provider(backend)
        try:
            with concurrent.futures.ThreadPoolExecutor(_CAPPED_CALL_WORKERS) as callers:
                held = list(
                    callers.map(
                        lambda i: compressor.compress(
                            f"held{i} {LONG}", _time_budget_cap_seconds=0.2
                        ),
                        range(_CAPPED_CALL_WORKERS),
                    )
                )
            assert all(out.compressed == f"held{i} {LONG}" for i, out in enumerate(held))
            assert len(backend.reached) == _CAPPED_CALL_WORKERS
            assert _free_permits(compressor) == 0

            for i in range(5):
                late = f"late{i} {LONG}"
                started = time.monotonic()
                out = compressor.compress(late, _time_budget_cap_seconds=0.06)
                assert out.compressed == late
                assert time.monotonic() - started < 0.06 + 0.1
                assert compressor._capped_pool._work_queue.qsize() == 0
            assert len(backend.reached) == _CAPPED_CALL_WORKERS
        finally:
            backend.release.set()

        assert _wait_until(lambda: _free_permits(compressor) == _CAPPED_CALL_WORKERS)
        recovered = compressor.compress(LONG, _time_budget_cap_seconds=2.0)
        assert recovered.compressed != LONG, "capacity returns once the held calls end"
        compressor.close()

    def test_an_uncapped_call_is_bounded_by_the_provider_timeout(self) -> None:
        backend = _Blocking()
        compressor = ProviderKompressCompressor(backend, timeout=0.1)
        try:
            started = time.monotonic()
            assert compressor.compress(LONG).compressed == LONG
            assert time.monotonic() - started < 0.1 + 0.15
        finally:
            backend.release.set()
        compressor.close()

    def test_the_provider_timeout_also_bounds_a_larger_ml_budget(self) -> None:
        backend = _Blocking()
        compressor = ProviderKompressCompressor(backend, timeout=0.1)
        try:
            started = time.monotonic()
            out = compressor.compress(LONG, _time_budget_cap_seconds=5.0)
            assert out.compressed == LONG
            assert time.monotonic() - started < 0.1 + 0.15
        finally:
            backend.release.set()
        compressor.close()

    def test_the_default_uncapped_call_runs_on_a_bounded_worker(self) -> None:
        backend = _Halver()
        compressor = ProviderKompressCompressor(backend)
        assert compressor.compress(LONG).compressed != LONG
        assert backend.threads[0].startswith("kompress-provider")
        compressor.close()

    def test_timeout_none_runs_inline(self) -> None:
        backend = _Halver()
        compressor = ProviderKompressCompressor(backend, timeout=None)
        assert compressor.compress(LONG).compressed != LONG
        assert backend.threads == [threading.current_thread().name]
        assert compressor._capped_pool is None


# =============================================================================
# ContentRouter selection
# =============================================================================


def _local_kompress_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    import headroom.transforms.kompress_compressor as kc

    available = MagicMock(return_value=True)
    local = MagicMock()
    monkeypatch.setattr(kc, "is_kompress_available", available)
    monkeypatch.setattr(kc, "KompressCompressor", local)
    return available, local


class TestRouterSelection:
    def test_provider_is_built_once_and_cached_in_the_remote_slot(
        self, monkeypatch, halver
    ) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        monkeypatch.setenv(KOMPRESS_PROVIDER_OPTIONS_ENV, '{"level": 2}')
        _available, local = _local_kompress_spies(monkeypatch)
        router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))

        first = router._get_kompress()
        assert isinstance(first, ProviderKompressCompressor)
        assert router._kompress_remote is first
        assert router._get_remote_kompress() is first
        assert router._get_kompress() is first
        assert [b.options for b in halver.instances] == [{"level": 2}]
        assert first.config.enable_ccr is False
        assert first.provider == "halver"
        local.assert_not_called()

    def test_custom_model_requests_use_the_provider_too(self, monkeypatch, halver) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        _available, local = _local_kompress_spies(monkeypatch)
        router = ContentRouter(ContentRouterConfig())
        router._runtime_kompress_model = "acme/kompress-finance"

        assert isinstance(router._get_kompress(), ProviderKompressCompressor)
        local.assert_not_called()

    def test_a_changed_selection_rebuilds_it(self, monkeypatch, halver) -> None:
        register_kompress_backend("other", _Halver)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        router = ContentRouter(ContentRouterConfig())
        first = router._get_kompress()

        monkeypatch.setenv(KOMPRESS_PROVIDER_OPTIONS_ENV, '{"level": 3}')
        second = router._get_kompress()
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "other")
        third = router._get_kompress()

        assert len({id(first), id(second), id(third)}) == 3
        assert second.backend.options == {"level": 3}
        assert third.provider == "other"

    def test_concurrent_first_callers_build_one_backend(self, monkeypatch, halver) -> None:
        """The blocks of one request compress in parallel; each used to build
        (and all but one to drop, unclosed) its own backend."""

        class SlowLoad(_Halver):
            name = "slowload"

            def __init__(self, **options: Any) -> None:
                time.sleep(0.3)  # loading a model
                super().__init__(**options)

        register_kompress_backend("slowload", SlowLoad)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "slowload")
        router = ContentRouter(ContentRouterConfig())
        barrier = threading.Barrier(8)

        def first_use(_: int) -> Any:
            barrier.wait(5)
            return router._get_kompress()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            got = list(pool.map(first_use, range(8)))

        assert len(halver.instances) == 1, "one construction"
        assert all(compressor is router._kompress_remote for compressor in got)
        assert halver.instances[0].closed == 0
        router._kompress_remote.close()

    def test_a_compressor_the_slot_drops_is_closed(self, monkeypatch, halver) -> None:
        register_kompress_backend("other", _Halver)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        router = ContentRouter(ContentRouterConfig())
        first = router._get_kompress()

        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "other")
        second = router._get_kompress()
        assert (first.backend.closed, second.backend.closed) == (1, 0)
        assert router._get_kompress() is second, "a cache hit closes nothing"
        assert second.backend.closed == 0

        # Unsetting the provider hands the slot to the endpoint...
        monkeypatch.delenv(KOMPRESS_PROVIDER_ENV)
        monkeypatch.setenv(KOMPRESS_ENDPOINT_ENV, "https://kompress.example")
        remote = router._get_kompress()
        assert isinstance(remote, RemoteKompressCompressor)
        assert second.backend.closed == 1
        assert not remote._client.is_closed

        # ...and a provider that cannot be built empties it.
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "missing")
        assert router._get_remote_kompress() is None
        assert router._kompress_remote is None
        assert remote._client.is_closed

    def test_provider_wins_over_the_endpoint_with_one_warning(
        self, monkeypatch, halver, caplog
    ) -> None:
        caplog.set_level(logging.WARNING, logger=REMOTE_LOGGER)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        monkeypatch.setenv(KOMPRESS_ENDPOINT_ENV, "https://kompress.example")

        for router in (ContentRouter(ContentRouterConfig()), ContentRouter(ContentRouterConfig())):
            assert isinstance(router._get_kompress(), ProviderKompressCompressor)
            assert isinstance(router._get_kompress(), ProviderKompressCompressor)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert KOMPRESS_PROVIDER_ENV in warnings[0] and KOMPRESS_ENDPOINT_ENV in warnings[0]
        assert "kompress.example" not in warnings[0]

    def test_unsetting_the_provider_falls_back_to_the_endpoint(self, monkeypatch, halver) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        monkeypatch.setenv(KOMPRESS_ENDPOINT_ENV, "https://kompress.example")
        router = ContentRouter(ContentRouterConfig())
        assert isinstance(router._get_kompress(), ProviderKompressCompressor)

        monkeypatch.delenv(KOMPRESS_PROVIDER_ENV)
        remote = router._get_kompress()
        try:
            assert isinstance(remote, RemoteKompressCompressor)
            assert router._kompress_remote is remote
            assert router._get_kompress() is remote
        finally:
            remote.close()

    def test_disabled_kompress_never_builds_the_provider(self, monkeypatch) -> None:
        built: list[object] = []
        register_kompress_backend("spy", lambda **kw: built.append(kw) or _Halver())
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "spy")

        router = ContentRouter(ContentRouterConfig(enable_kompress=False))
        assert router._get_kompress() is None

        router = ContentRouter(ContentRouterConfig())
        router._runtime_kompress_model = "disabled"
        assert router._get_kompress() is None
        assert built == []

    @pytest.mark.parametrize(
        ("env", "must_not_leak"),
        [
            ({KOMPRESS_PROVIDER_ENV: "sk-pasted-key"}, "sk-pasted-key"),
            ({KOMPRESS_PROVIDER_ENV: "sk pasted key"}, "sk pasted key"),
            (
                {KOMPRESS_PROVIDER_ENV: "halver", KOMPRESS_PROVIDER_OPTIONS_ENV: '{"k": "sk-x"'},
                "sk-x",
            ),
            ({KOMPRESS_PROVIDER_ENV: "raises"}, "sk-from-factory"),
            ({KOMPRESS_PROVIDER_ENV: "not-a-backend"}, "sk-never"),
        ],
    )
    def test_an_unusable_provider_disables_kompress_with_one_error(
        self, monkeypatch, halver, caplog, env: dict[str, str], must_not_leak: str
    ) -> None:
        def _raises(**_kw: Any) -> Any:
            raise RuntimeError("factory saw sk-from-factory")

        register_kompress_backend("raises", _raises)
        register_kompress_backend("not-a-backend", lambda **kw: object())
        caplog.set_level(logging.DEBUG)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        _available, local = _local_kompress_spies(monkeypatch)

        routers = [ContentRouter(ContentRouterConfig()), ContentRouter(ContentRouterConfig())]
        for router in routers:
            for _ in range(3):
                assert router._get_kompress() is None
                assert router._get_remote_kompress() is None
            assert router._kompress_remote is None
            assert router._kompress_model_ready() is True  # nothing is loading

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, errors
        assert KOMPRESS_PROVIDER_ENV in errors[0] and "disabled" in errors[0]
        assert must_not_leak not in caplog.text
        local.assert_not_called()
        _available.assert_not_called()

    def test_the_router_seam_reads_only_the_stubbed_attributes(self, monkeypatch, halver) -> None:
        """The offline tests drive ``_get_remote_kompress`` on a stub holding only
        ``config.ccr_inject_marker`` and ``_kompress_remote``."""
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")

        class _Stub:
            config = SimpleNamespace(ccr_inject_marker=True)
            _kompress_remote = None
            _get_remote_kompress = ContentRouter._get_remote_kompress

        stub = _Stub()
        compressor = stub._get_remote_kompress()
        assert isinstance(compressor, ProviderKompressCompressor)
        assert stub._get_remote_kompress() is compressor

    def test_an_air_gap_refusal_at_construction_propagates(self, monkeypatch) -> None:
        def _guarded(**_kw: Any) -> _Halver:
            guard_egress("acme Kompress", "https://kompress.acme.example")
            return _Halver()

        register_kompress_backend("guarded", _guarded)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "guarded")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        router = ContentRouter(ContentRouterConfig())

        for _ in range(2):  # not cached as an ordinary failure
            with pytest.raises(OfflineEgressBlocked):
                router._get_remote_kompress()

    def test_compression_runs_through_the_router(self, monkeypatch, halver) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
        prose = "\n".join(
            "the deployment pipeline reported elevated latency in the inference workspace "
            f"during step {i} while the telemetry boundary stayed healthy"
            for i in range(40)
        )

        out, _tokens = router._try_ml_compressor(prose, "context", question="why slow?")

        backend = halver.instances[0]
        assert backend.calls == [(prose, None, "why slow?")]
        assert out == " ".join(prose.split()[::2])

    def test_apply_compresses_a_tool_result_with_the_provider(self, monkeypatch, halver) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
        # Varied prose: repeated templated lines would be routed as a log.
        rng = random.Random(7)
        words = (
            "analysis deployment configuration throughput latency pipeline compression "
            "artifact identifier resolution boundary telemetry inference workspace"
        ).split()
        prose = "\n".join(" ".join(rng.choice(words) for _ in range(14)) for _ in range(150))
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": prose}],
            },
        ]

        class _Tokenizer:
            def count_text(self, content: str) -> int:
                return len(content.split())

        result = router.apply(
            messages, _Tokenizer(), frozen_message_count=1, min_tokens_to_compress=1
        )

        assert halver.instances and halver.instances[0].calls, "the provider never ran"
        assert result.transforms_applied == ["router:tool_result:text"]
        out = result.messages[-1]["content"][0]["content"]
        assert out == " ".join(prose.split()[::2])


class TestCacheIdentity:
    def test_the_provider_variables_are_part_of_the_cache_namespace(self, monkeypatch) -> None:
        """Results made by one compressor must never be served for another."""
        config = ContentRouterConfig()
        baseline = _config_namespace(config)

        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "alpha")
        alpha = _config_namespace(config)
        monkeypatch.setenv(KOMPRESS_PROVIDER_OPTIONS_ENV, '{"model": "small"}')
        alpha_small = _config_namespace(config)
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "beta")
        beta_small = _config_namespace(config)

        assert len({baseline, alpha, alpha_small, beta_small}) == 4
        assert ContentRouter(config)._cache._namespace == beta_small

        monkeypatch.delenv(KOMPRESS_PROVIDER_ENV)
        monkeypatch.delenv(KOMPRESS_PROVIDER_OPTIONS_ENV)
        assert _config_namespace(config) == baseline


# =============================================================================
# Startup status and /health
# =============================================================================


class TestHealth:
    def test_eager_load_reports_the_custom_backend(self, monkeypatch, halver) -> None:
        from headroom.proxy.warmup import WarmupRegistry

        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
        router = ContentRouter(
            ContentRouterConfig(enable_code_aware=False, enable_smart_crusher=False)
        )
        prefetch = MagicMock(return_value=True)
        monkeypatch.setattr(router, "_prefetch_kompress_artifacts_async", prefetch)

        status = router.eager_load_compressors()

        assert status["kompress"] == "enabled"
        assert status["kompress_backend"] == "custom"
        prefetch.assert_not_called()  # no local model to fetch
        registry = WarmupRegistry()
        registry.merge_transform_status(status)
        assert registry.kompress.status == "loaded"
        assert registry.kompress.info["backend"] == "custom"

    def test_eager_load_unusable_provider_is_unavailable(self, monkeypatch) -> None:
        monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "missing")
        router = ContentRouter(
            ContentRouterConfig(enable_code_aware=False, enable_smart_crusher=False)
        )
        assert router.eager_load_compressors()["kompress"] == "unavailable"

    def test_readyz_finds_the_provider_and_reports_custom(self, monkeypatch) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
        app = create_app(ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False))
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()
        router = proxy.anthropic_pipeline.transforms[-1]
        backend = _Halver()
        backend.name = "Pwned\r\nX: 1"  # type: ignore[misc]
        router._kompress = None
        router._kompress_remote = ProviderKompressCompressor(backend)

        payload = TestClient(app).get("/readyz").json()

        assert payload["checks"]["kompress"] == {
            "enabled": True,
            "ready": True,
            "status": "healthy",
            "optional": True,
            "backend": "custom",
            "detail": "loaded",
        }
        assert "Pwned" not in json.dumps(payload)


# =============================================================================
# OpenAI reasoning compaction
# =============================================================================


def test_openai_reasoning_compaction_uses_the_provider(monkeypatch, halver) -> None:
    """``proxy/handlers/openai.py`` asks ``_get_remote_kompress() or _get_kompress()``;
    the provider lives in that slot, so compaction must use it."""
    pytest.importorskip("fastapi")
    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_HTTP2", "false")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_THINKING_COMPACT", "1")
    monkeypatch.setenv("HEADROOM_THINKING_COMPACT_KEEP_LAST", "0")
    monkeypatch.setenv(KOMPRESS_PROVIDER_ENV, "halver")
    _available, local = _local_kompress_spies(monkeypatch)
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
    )
    forwarded: list[dict] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "model": "kimi-k2.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
            },
        )

    reasoning = " ".join(f"reason{i}" for i in range(100))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        proxy = app.state.proxy

        async def install_transport() -> None:
            await proxy.http_client.aclose()
            if proxy.http_client_h1 is not None:
                await proxy.http_client_h1.aclose()
            proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            proxy.http_client_h1 = None

        client.portal.call(install_transport)
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer local-fixture"},
            json={
                "model": "kimi-k2.7",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer", "reasoning_content": reasoning},
                    {"role": "user", "content": "continue"},
                ],
            },
        )
        assert response.status_code == 200
        readyz = client.get("/readyz").json()

    assert any(call[0] == reasoning for b in halver.instances for call in b.calls), (
        "reasoning compaction did not reach the provider"
    )
    assert len(forwarded) == 1
    sent = forwarded[0]["messages"][1]["reasoning_content"]
    assert sent != reasoning
    assert sent.startswith(" ".join(reasoning.split()[::2]))
    local.assert_not_called()
    # Startup reported the provider, not the remote endpoint or a local model.
    assert readyz["checks"]["kompress"]["backend"] == "custom"
    assert readyz["checks"]["kompress"]["ready"] is True
