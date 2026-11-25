"""Remote Kompress: offload ML compression to a hosted ``/compress`` endpoint.

Lets a sandboxed proxy — installed WITHOUT the ``[ml]`` extra (no torch/onnx) —
still run Kompress by calling a remote endpoint over HTTP. The class mirrors
:class:`~headroom.transforms.kompress_compressor.KompressCompressor`'s public
surface (``is_ready`` / ``preload`` / ``ensure_background_load`` / ``compress``),
so it is a drop-in at the ContentRouter seam.

Only the model inference is remote. The CCR store + retrieval marker stay
proxy-local (the endpoint is stateless, ``enable_ccr=False``), so
``headroom_retrieve`` keeps working and original content never persists off-box.

Enabled by ``HEADROOM_KOMPRESS_ENDPOINT`` — see ``ContentRouter._get_kompress``.

The same post-processing serves a Kompress *backend*: a plugin selected by
``HEADROOM_KOMPRESS_PROVIDER`` that supplies only text in, text out (see
:mod:`headroom.transforms.kompress_backend`). :class:`DelegatedKompressCompressor`
holds everything both share — the word floor, the request ML-budget floor and
cap, fail-open pass-through, the proxy-local CCR store, retrieval marker and
the marker-must-save-tokens gate, ``ccr_original`` — and the two subclasses
differ only in how the compressed text is obtained:
:class:`RemoteKompressCompressor` POSTs it, :class:`ProviderKompressCompressor`
calls the backend.

# Bring-your-own deployment

The endpoint does not have to be Headroom Labs'. An org can pull the Kompress
weights from HuggingFace, serve them on its own stack (vLLM, TorchServe,
SageMaker, KServe, a bare FastAPI box) and point Headroom at it. Nothing about
this class is Modal-specific, and no credential is required — auth is whatever
the operator's own infrastructure expects, including none at all:

    HEADROOM_KOMPRESS_ENDPOINT          https://ml.internal.acme.com
    HEADROOM_KOMPRESS_ENDPOINT_PATH     /compress   (default; set empty to use
                                        the endpoint URL verbatim)
    HEADROOM_KOMPRESS_ENDPOINT_TOKEN    optional; sent as `Authorization: Bearer`
    HEADROOM_KOMPRESS_ENDPOINT_HEADERS  optional; `k=v,k2=v2`, applied last so it
                                        can replace the Authorization header for
                                        stacks that want `x-api-key` or similar

Both new knobs default to today's behaviour: with only
``HEADROOM_KOMPRESS_ENDPOINT`` set, the request is byte-identical to before —
``POST <endpoint>/compress`` with an optional Bearer token. Existing Modal
deployments need no change.

# The HTTP contract

Deliberately small, so a shim in front of an existing inference server is a few
lines. ``POST <endpoint><path>``:

    request   {"content": "<text>", "target_ratio": 0.5 | null}
    response  {"compressed": "<text>",          # REQUIRED, must be a string
               "original_tokens": int,          # optional, defaults to word count
               "compressed_tokens": int,        # optional, defaults to word count
               "compression_ratio": float,      # optional, defaults to 1.0
               "model_used": str}               # optional

``compressed`` is the only required field; every other value is derived if
absent. Any non-2xx, timeout, malformed field, or missing ``compressed`` makes
this pass the content through verbatim — a broken endpoint costs compression,
never correctness.
"""

from __future__ import annotations

import abc
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import httpx

from ..log_safety import WarnOnce, describe_class, describe_exception
from ..models.backends.registry import NAME_RE, BackendError
from ..offline import guard_egress
from .kompress_backend import (
    KOMPRESS_PROVIDER_ENV,
    KOMPRESS_PROVIDER_OPTIONS_ENV,
    KompressModelBackend,
    call_kompress_backend,
    create_kompress_backend,
    kompress_provider_settings_from_env,
    resolve_kompress_provider,
)
from .kompress_compressor import (
    KompressConfig,
    KompressResult,
    ccr_retrieval_marker,
    payload_tokens,
    store_kompress_in_ccr,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Appended to the configured endpoint unless overridden. An operator whose stack
# already serves a full path (``/v1/models/kompress:predict``) sets
# HEADROOM_KOMPRESS_ENDPOINT_PATH="" and gives the complete URL instead.
DEFAULT_ENDPOINT_PATH = "/compress"

#: Selects remote Kompress. Read by ``ContentRouter._get_remote_kompress``.
KOMPRESS_ENDPOINT_ENV = "HEADROOM_KOMPRESS_ENDPOINT"

# Upper bound on one backend call when the caller passes no ML budget (direct
# callers such as reasoning compaction). The same 20s that bounds the remote
# endpoint's HTTP phases and local Kompress's cooperative deadline, so no
# Kompress variant can hold a thread longer than the others by default.
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 20.0


def parse_endpoint_headers(raw: str | None) -> dict[str, str]:
    """Parse ``k=v,k2=v2`` into a header dict.

    Same format as ``HEADROOM_OTEL_METRICS_HEADERS`` so operators meet one
    convention. Reimplemented rather than imported from
    :mod:`headroom.observability.metrics`, which pulls in opentelemetry at module
    scope — remote Kompress exists precisely so a proxy can run without heavy
    optional deps, so it must not drag one in through a parsing helper.
    """
    pairs: dict[str, str] = {}
    for item in (raw or "").split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            pairs[key] = value
    return pairs


# Below this word count local Kompress passes through verbatim (KompressCompressor
# .compress); mirror it so we never pay a round-trip on a trivially small block.
_MIN_WORDS = 10

# Accept-any-shrink CCR gate, identical to KompressCompressor.compress: only
# store + mark when the shrink is worth the retrieval marker's own cost.

# A remote call is declined outright when the caller's request-scoped ML budget
# has this little left: no HTTP round trip (connect + write + server inference +
# read) fits in it, so starting one only burns the budget and leaves a worker
# behind. The block passes through unchanged instead.
_MIN_REMOTE_BUDGET_SECONDS = 0.05

# Workers for budget-capped calls, and the number of admission permits. A call
# whose caller times out is abandoned, not killed (a sync httpx request cannot be
# interrupted from another thread), so the worker keeps its permit until the
# request itself ends -- and every request ends at an absolute deadline (see
# ``_capped_request``). With permits == workers nothing ever waits in the
# executor's (unbounded) queue: a saturated caller waits for a permit, not with
# its payload enqueued, and fails open when its own deadline passes.
_CAPPED_CALL_WORKERS = 16


class _BudgetExceeded(TimeoutError):
    """A capped call ran out of time in Headroom's own admission or wait.

    Its message is written here and names only counts and durations, so it is
    safe to log even where a backend's own exception text is not.
    """


class DelegatedKompressCompressor(abc.ABC):
    """Kompress whose model runs outside this class: the shared post-processing.

    Subclasses supply :meth:`_infer`, which returns the remote endpoint's
    response fields for one block. Everything around it is here so the remote
    endpoint and plugin backends cannot drift apart on an invariant: the word
    floor, the request ML budget (declined below a floor, capped otherwise),
    fail-open pass-through, and the proxy-local CCR store with its retrieval
    marker and marker-must-save-tokens gate.

    Fails OPEN: any error from :meth:`_infer` returns the content verbatim so a
    broken model degrades compression rather than breaking the proxy.
    """

    name = "kompress_compressor"

    # ``compress()`` accepts ``_time_budget_cap_seconds``: what the router's
    # request-scoped ML budget has left. Unlike local Kompress this class does
    # not take ``_deadline_started_at``, so it does not set
    # ``shares_request_deadline``; the router checks this flag separately.
    accepts_time_budget_cap: bool = True

    #: What ``ready_backend()`` and ``preload()`` report, which ``/health`` and
    #: ``/readyz`` serve unauthenticated. A fixed literal per subclass, never a
    #: value a plugin or an operator supplied.
    backend_label = ""
    # Names used in log lines and budget messages; fixed per subclass.
    _log_name = "Kompress"
    _workers_noun = "Kompress"
    _answerer = "model"
    _worker_thread_prefix = "kompress"

    def __init__(self, config: KompressConfig | None = None) -> None:
        self.config = config or KompressConfig()
        self._capped_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._capped_pool_lock = threading.Lock()
        self._capped_slots = threading.BoundedSemaphore(_CAPPED_CALL_WORKERS)

    # Nothing to load locally; short-circuit the router straight to compress().
    def is_ready(self) -> bool:
        return True

    def ready_backend(self) -> str | None:
        return self.backend_label

    def preload(self, *, allow_download: bool = True) -> str:
        return self.backend_label

    def ensure_background_load(self) -> None:
        return None

    def _passthrough(self, content: str, n_words: int) -> KompressResult:
        return KompressResult(
            compressed=content,
            original=content,
            original_tokens=n_words,
            compressed_tokens=n_words,
            compression_ratio=1.0,
            model_used=self.config.model_id,
        )

    # ------------------------------------------------------------ subclass hooks

    def _check_egress(self) -> None:
        """Refuse before any work when this compressor's model is off-host and the
        process is air-gapped. Raises; never returns a pass-through.

        The default guards nothing: a Kompress backend that opens connections
        must call ``guard_egress`` itself, which only it can place correctly.
        """
        return None

    @abc.abstractmethod
    def _infer(
        self,
        content: str,
        *,
        target_ratio: float | None,
        question: str | None,
        cap: float | None,
    ) -> Mapping[str, Any]:
        """Return the response fields for ``content``; ``compressed`` must be a str.

        ``cap`` is what is left of the caller's ML budget, or ``None``. Any
        exception makes :meth:`compress` pass the block through.
        """

    @abc.abstractmethod
    def _log_failure(self, exc: Exception) -> None:
        """Log one failed call; :meth:`compress` then passes the block through."""

    # ------------------------------------------------------------------ compress

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | None = None,
        *,
        allow_download: bool = True,
        ccr_original: str | None = None,
        _time_budget_cap_seconds: float | None = None,
    ) -> KompressResult:
        """Compress via the delegated model.

        ``_time_budget_cap_seconds`` is what is left of the caller's
        request-scoped ML budget (ContentRouter passes it). It bounds the model
        call AS A WHOLE -- for the endpoint: connect, write, server time and
        read together, not each phase -- and a call that runs out passes the
        block through unchanged. When the remainder is at or below
        :data:`_MIN_REMOTE_BUDGET_SECONDS` no call is made at all. ``None``
        (direct callers) keeps the subclass's own default bound: the endpoint's
        configured per-phase timeout, 20s by default.

        ``ccr_original`` mirrors :meth:`KompressCompressor.compress`: text to
        store in CCR instead of ``content``, used when ``content`` is a
        tag-protected placeholder intermediate ({{HEADROOM_TAG_N}}). It is
        accepted here because this class promises to be a DROP-IN for the local
        compressor at the ContentRouter seam (see the module docstring) — and it
        was not. ContentRouter passes the kwarg whenever custom tags are
        protected, so on any deployment with HEADROOM_KOMPRESS_ENDPOINT set,
        every such request raised

            TypeError: RemoteKompressCompressor.compress() got an unexpected
            keyword argument 'ccr_original'

        which ContentRouter caught with a broad ``except Exception`` and logged
        as ``Kompress failed: ...``. Compression silently degraded to zero on the
        whole deployment while the proxy kept reporting success.
        """
        n_words = len(content.split())
        # Same floor contract as the in-process compressor: lossy
        # word-dropping below config.min_input_words is a net loss (the
        # retrieval marker alone is ~20 words) and garbles short
        # instruction-like blocks. _MIN_WORDS stays the hard clamp.
        if n_words < max(_MIN_WORDS, self.config.min_input_words):
            return self._passthrough(content, n_words)

        # Outside the fail-open try on purpose: the except below turns every
        # exception into a silent pass-through, and a policy refusal that
        # degrades silently is exactly the hole the air-gap guard closes.
        self._check_egress()

        if _time_budget_cap_seconds is not None and (
            _time_budget_cap_seconds <= _MIN_REMOTE_BUDGET_SECONDS
        ):
            logger.info(
                "%s declined: %.3fs of request ML budget left (floor %.3fs); "
                "passing through unchanged",
                self._log_name,
                max(0.0, _time_budget_cap_seconds),
                _MIN_REMOTE_BUDGET_SECONDS,
            )
            return self._passthrough(content, n_words)

        try:
            data = self._infer(
                content,
                target_ratio=target_ratio,
                question=question,
                cap=_time_budget_cap_seconds,
            )
            compressed = data["compressed"]
            # Coerce the numeric/string metadata fields inside the fail-open guard.
            # A 200 response with a malformed field (e.g. a non-numeric string, or
            # an explicit JSON null: data.get returns None for a present key, and
            # float(None)/int(None) raise) would otherwise escape uncaught and break
            # the proxy request, defeating the fail-open contract this class promises.
            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=int(data.get("original_tokens", n_words)),
                compressed_tokens=int(data.get("compressed_tokens", len(compressed.split()))),
                compression_ratio=float(data.get("compression_ratio", 1.0)),
                model_used=str(data.get("model_used", self.config.model_id)),
            )
        except Exception as e:  # fail OPEN — never break the proxy on a bad model
            self._log_failure(e)
            return self._passthrough(content, n_words)

        # CCR stays PROXY-LOCAL: the model is stateless (enable_ccr=False), so we
        # store the mapping + append the retrieval marker here — same policy and
        # marker format as KompressCompressor.compress.
        if self.config.enable_ccr and compressed != content:
            # Same gate as KompressCompressor: the marked payload must save
            # tokens, and a result that cannot pay for the marker passes through.
            # Store the PRE-protection text when the caller supplied it. ``content``
            # may be the tag-protected placeholder intermediate, and storing that
            # makes a later full retrieval hand back {{HEADROOM_TAG_N}} instead of
            # the real block — the exact loss ``ccr_original`` exists to prevent.
            # Same resolution order as KompressCompressor.compress.
            ccr_source = ccr_original if ccr_original is not None else content
            # The model's ``original_tokens`` describes ``content``, so it does
            # not describe a different ``ccr_source``; count that one locally.
            # Unchanged on the common path where no override was passed.
            ccr_source_tokens = (
                len(ccr_source.split()) if ccr_original is not None else result.original_tokens
            )
            cache_key = store_kompress_in_ccr(ccr_source, compressed, ccr_source_tokens)
            if cache_key:
                # Report the source line span so a reader can tell content was
                # compressed away rather than absent (#2586).
                marked = compressed + ccr_retrieval_marker(
                    result.original_tokens, result.compressed_tokens, ccr_source, cache_key
                )
                # Whole original against whole marked candidate, one unit,
                # and the accounting reports that measurement.
                original_tokens = payload_tokens(content)
                compressed_tokens = payload_tokens(marked)
                if compressed_tokens >= original_tokens:
                    return self._passthrough(content, n_words)
                result.cache_key = cache_key
                result.compressed = marked
                result.original_tokens = original_tokens
                result.compressed_tokens = compressed_tokens
                result.compression_ratio = (
                    compressed_tokens / original_tokens if original_tokens else 1.0
                )

        return result

    # ------------------------------------------------------------ capped calls

    def _call_within(
        self,
        work: Callable[[float], _T],
        cap: float,
        *,
        budget: str = "request ML budget",
    ) -> _T:
        """Run ``work(deadline)`` on a worker and return its result within *cap* seconds.

        A worker cannot be interrupted, so three things together bound the call:

        * **Admission.** A permit (one per worker) is taken before submitting,
          waiting at most until the caller's deadline. Saturated means fail
          open, with the payload never enqueued -- ``future.cancel()`` does not
          free a queued payload, so an unbounded queue would retain every
          timed-out prompt.
        * **Caller deadline.** The caller waits for the worker with one overall
          deadline and gets :class:`_BudgetExceeded` when it runs out.
        * **Worker permit.** The worker keeps its permit until ``work`` actually
          returns (released in the worker's own ``finally``, never on caller
          timeout), so abandoned calls cannot pile up past the worker count.
          ``work`` receives the absolute deadline to stop by, where it can.
        """
        deadline = time.monotonic() + cap
        if not self._capped_slots.acquire(timeout=max(0.0, cap)):
            raise _BudgetExceeded(
                f"all {_CAPPED_CALL_WORKERS} {self._workers_noun} workers busy for the "
                f"{budget} ({cap:.3f}s)"
            )
        try:
            future = self._get_capped_pool().submit(self._run_holding_permit, work, deadline)
        except BaseException:
            self._capped_slots.release()  # never reached the worker that would
            raise
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            raise _BudgetExceeded(
                f"exceeded the {budget} ({cap:.3f}s) before the {self._answerer} answered"
            ) from None

    def _run_holding_permit(self, work: Callable[[float], _T], deadline: float) -> _T:
        """The worker side of :meth:`_call_within`. Owns one admission permit."""
        try:
            return work(deadline)
        finally:
            self._capped_slots.release()

    def _get_capped_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._capped_pool_lock:
            if self._capped_pool is None:
                self._capped_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CAPPED_CALL_WORKERS,
                    thread_name_prefix=self._worker_thread_prefix,
                )
            return self._capped_pool

    def _shutdown_pool(self) -> None:
        with self._capped_pool_lock:
            pool, self._capped_pool = self._capped_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def close(self) -> None:
        self._shutdown_pool()


class RemoteKompressCompressor(DelegatedKompressCompressor):
    """Drop-in for KompressCompressor that POSTs to a hosted ``/compress`` endpoint.

    Fails OPEN: any network/HTTP error returns the content verbatim so a flaky
    endpoint degrades compression rather than breaking the proxy.
    """

    backend_label = "remote"
    _log_name = "Remote Kompress"
    _workers_noun = "remote Kompress"
    _answerer = "endpoint"
    _worker_thread_prefix = "remote-kompress"

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        config: KompressConfig | None = None,
        timeout: float = 20.0,
        path: str | None = DEFAULT_ENDPOINT_PATH,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        # Default keeps the pre-existing behaviour exactly: <endpoint>/compress.
        # An empty (or None) path means the caller has supplied a complete URL —
        # needed because real inference servers do not serve at /compress
        # (TorchServe /predictions/<model>, KServe /v1/models/<name>:predict,
        # SageMaker /invocations), and appending to those yields a 404.
        if path:
            suffix = path if path.startswith("/") else "/" + path
            self._url = endpoint.rstrip("/") + suffix
        else:
            self._url = endpoint
        self._headers = {"content-type": "application/json"}
        if token:
            self._headers["authorization"] = f"Bearer {token}"
        # Applied last on purpose: lets an operator replace `authorization` with
        # whatever their gateway wants (x-api-key, a signed header, a tenant id)
        # without needing a separate auth-scheme setting.
        if headers:
            self._headers.update(headers)
        # Air-gap chokepoint. HEADROOM_KOMPRESS_ENDPOINT points at a box that is
        # by definition off this host, so under HEADROOM_OFFLINE this class must
        # not exist at all. Guarding at construction (not at the POST) means the
        # connection pool is never created and the operator hears about the
        # contradictory configuration the first time the router reaches for
        # Kompress, rather than discovering it in a packet capture.
        guard_egress("remote Kompress inference", self._url)
        # httpx.Client is safe to share across the proxy's worker threads.
        # ``transport`` exists for tests (httpx.MockTransport); None is the
        # normal network transport.
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)

    @property
    def url(self) -> str:
        """The resolved POST target.

        Logged when the router builds this, so a wrong ``_PATH`` shows up as a
        visibly odd URL at startup rather than as silent pass-through later —
        the fail-open contract means a 404 never surfaces as an error.
        """
        return self._url

    def _check_egress(self) -> None:
        # Re-check on every call. __init__ already refused if the flag was set
        # then, but the env can be flipped after this object was cached on the
        # ContentRouter (the router builds it once per instance and reuses it for
        # the process lifetime), and the security property we owe is "no
        # socket", not "no construction".
        guard_egress("remote Kompress inference", self._url)

    def _infer(
        self,
        content: str,
        *,
        target_ratio: float | None,
        question: str | None,
        cap: float | None,
    ) -> Mapping[str, Any]:
        # ``question`` is not part of the HTTP contract; the request stays
        # byte-identical to what deployed endpoints already accept.
        payload = {"content": content, "target_ratio": target_ratio}
        if cap is None:
            data = self._post(payload)
        else:
            data = self._post_within(payload, cap)
        compressed = data["compressed"]
        if not isinstance(compressed, str):
            raise TypeError("remote Kompress response field 'compressed' must be a string")
        fields: Mapping[str, Any] = data
        return fields

    def _log_failure(self, exc: Exception) -> None:
        logger.warning("Remote Kompress failed (%s); passing through", exc)

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> Any:
        # No ``timeout`` kwarg on the default path: the client's own (20s)
        # applies, and the call is exactly what it was before the cap existed.
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        resp = self._client.post(self._url, headers=self._headers, json=payload, **extra)
        resp.raise_for_status()
        return resp.json()

    def _post_within(self, payload: dict[str, Any], cap: float) -> Any:
        """POST and read the response within *cap* seconds in total.

        httpx timeouts are per phase (connect, write, pool, and each read
        between chunks), so ``timeout=cap`` alone neither bounds the call nor
        stops a slow-drip body. :meth:`_call_within` bounds admission and the
        caller's wait; the worker (:meth:`_capped_request`) finishes by an
        absolute deadline: per-phase timeouts are capped to what is left, and
        the body is streamed and cut off at the deadline, so a dripping
        response cannot hold a permit indefinitely.
        """
        return self._call_within(lambda deadline: self._capped_request(payload, deadline), cap)

    def _capped_request(self, payload: dict[str, Any], deadline: float) -> Any:
        """The HTTP work of :meth:`_post_within`, run on a permit-holding worker.

        Lifetime is bounded by *deadline* plus at most one per-phase timeout
        (itself no longer than what was left at start): every httpx phase is
        capped to the time remaining, and the body is read chunk by chunk with
        the absolute deadline checked after each one.
        """
        left = deadline - time.monotonic()
        if left <= 0.0:
            raise TimeoutError("request ML budget spent before the request started")
        with self._client.stream(
            "POST",
            self._url,
            headers=self._headers,
            json=payload,
            timeout=min(left, self._timeout),
        ) as resp:
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_bytes():
                body += chunk
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote Kompress response outlived the request ML budget")
        return json.loads(bytes(body))

    def close(self) -> None:
        self._shutdown_pool()
        self._client.close()


# One WARNING per (provider, failure kind) per hour, so a backend failing on
# every block warns once and a different failure still warns; the rest go to
# DEBUG. Remote Kompress keeps its per-call WARNING unchanged.
_PROVIDER_FAILURE_WARNINGS = WarnOnce(64, "failing Kompress providers")


def _backend_label(backend: object) -> str:
    name = getattr(backend, "name", None)
    if isinstance(name, str) and NAME_RE.match(name):
        return name
    return describe_class(type(backend))


class ProviderKompressCompressor(DelegatedKompressCompressor):
    """Kompress backed by a :class:`~headroom.transforms.kompress_backend.KompressModelBackend`.

    The backend supplies the compressed text; everything else is the shared
    post-processing of :class:`DelegatedKompressCompressor`. The backend call
    runs on a bounded worker pool with admission permits, capped by the
    tighter of the request's ML budget and ``timeout``, and a call that runs
    over passes the block through.

    Backend failures pass the block through. They are logged as a bounded
    WARNING naming the exception class (the message can quote the payload or a
    credential-bearing URL), with :func:`~headroom.log_safety.describe_exception`
    at DEBUG.

    Args:
        backend: The model.
        provider: Name for log lines, normally the registry name it was selected
            by. Defaults to the backend's ``name`` when that is a plain
            identifier.
        config: Kompress settings; only ``enable_ccr`` and ``min_input_words``
            apply.
        timeout: Upper bound in seconds on one backend call; a request's ML
            budget can only lower it. ``None`` sets no bound of its own, and a
            call made without a budget then runs inline, unbounded.
        selection_key: Opaque id of the configuration this was built from, so
            the router can tell when the configuration changed.
    """

    backend_label = "custom"
    _log_name = "Kompress provider"
    _workers_noun = "Kompress provider"
    _answerer = "provider"
    _worker_thread_prefix = "kompress-provider"

    def __init__(
        self,
        backend: KompressModelBackend,
        *,
        provider: str | None = None,
        config: KompressConfig | None = None,
        timeout: float | None = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        selection_key: str | None = None,
    ) -> None:
        if not isinstance(backend, KompressModelBackend):
            raise TypeError(
                f"{describe_class(type(backend))} does not implement KompressModelBackend "
                "(needs name and compress_text())"
            )
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number of seconds or None")
        super().__init__(config)
        self.backend = backend
        self.provider = provider if provider is not None else _backend_label(backend)
        self.selection_key = selection_key
        self._timeout = timeout

    def _infer(
        self,
        content: str,
        *,
        target_ratio: float | None,
        question: str | None,
        cap: float | None,
    ) -> Mapping[str, Any]:
        def work(_deadline: float) -> dict[str, Any]:
            return call_kompress_backend(
                self.backend,
                content,
                target_ratio=target_ratio,
                question=question,
                label=self.provider,
            )

        # The tighter of the request's ML budget and the provider's own bound.
        if self._timeout is not None and (cap is None or self._timeout < cap):
            return self._call_within(work, self._timeout, budget="Kompress provider timeout")
        if cap is not None:
            return self._call_within(work, cap)
        return work(math.inf)

    def _log_failure(self, exc: Exception) -> None:
        logger.debug("Kompress provider %r failed: %s", self.provider, describe_exception(exc))
        # Headroom's own errors are worded to be safe to log; anything else is
        # named by its class only.
        if isinstance(exc, (BackendError, _BudgetExceeded)):
            detail = str(exc)
        else:
            detail = describe_class(type(exc))
        cause = exc.__cause__
        key = (
            self.provider,
            describe_class(type(exc)),
            describe_class(type(cause)) if cause is not None else "",
        )
        first = _PROVIDER_FAILURE_WARNINGS.first(key, logger)
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "Kompress provider %r failed (%s); passing through",
            self.provider,
            detail,
        )

    def close(self) -> None:
        """Stop the worker pool and close the backend, if it has ``close()``."""
        self._shutdown_pool()
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                logger.debug(
                    "Closing Kompress provider %r failed: %s",
                    self.provider,
                    describe_exception(exc),
                )


# --------------------------------------------------------------------------- selection

# Process-wide, keyed by selection: the proxy runs one router per pipeline, and
# a configuration error is one fact, so it is reported once rather than per
# router or per block.
_selection_lock = threading.Lock()
# Selections that could not be built. They are not retried: a factory that
# fails at construction is a configuration error, and retrying it per block
# would rerun failing plugin code and repeat the ERROR on every request.
_failed_selections: set[str] = set()
_endpoint_overridden_warned: set[str] = set()


def provider_selection_key(env: Mapping[str, str] | None = None) -> str:
    """An opaque digest of the provider variables, credential-free by construction."""
    env = os.environ if env is None else env
    digest = hashlib.blake2b(digest_size=16)
    for var in (KOMPRESS_PROVIDER_ENV, KOMPRESS_PROVIDER_OPTIONS_ENV):
        digest.update((env.get(var) or "").strip().encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def provider_kompress_from_env(
    cached: object,
    *,
    enable_ccr: bool,
    env: Mapping[str, str] | None = None,
) -> ProviderKompressCompressor | None:
    """The compressor ``HEADROOM_KOMPRESS_PROVIDER`` selects, for the router's Kompress slot.

    Args:
        cached: What the router holds now. Returned as is when it was built
            from the same variables, so the backend and its worker pool are
            reused across requests; a changed selection builds a new one.
        enable_ccr: The router's marker policy (``ccr_inject_marker``).

    Returns:
        The compressor, or ``None`` when the selection cannot be used: a
        malformed variable, an unknown name, or a factory that fails. That is
        logged once per selection at ERROR, and the caller must then run no ML
        compression at all: falling back to the local model would silently run
        a different compressor than the operator configured.

    An air-gap refusal raised by the backend's constructor
    (:class:`~headroom.offline.OfflineEgressBlocked`) propagates, as it does
    for remote Kompress: a refusal must not degrade into "compression stopped".
    """
    env = os.environ if env is None else env
    key = provider_selection_key(env)
    if isinstance(cached, ProviderKompressCompressor) and cached.selection_key == key:
        return cached
    with _selection_lock:
        if key in _failed_selections:
            return None
        warn_endpoint = (
            bool((env.get(KOMPRESS_ENDPOINT_ENV) or "").strip())
            and key not in _endpoint_overridden_warned
        )
        if warn_endpoint:
            _endpoint_overridden_warned.add(key)
    if warn_endpoint:
        logger.warning(
            "Kompress: %s and %s are both set; the provider is used and the endpoint is not",
            KOMPRESS_PROVIDER_ENV,
            KOMPRESS_ENDPOINT_ENV,
        )
    try:
        settings = kompress_provider_settings_from_env(env)
        if settings is None:
            return None
        name, options = settings
        resolve_kompress_provider(name)
        compressor = ProviderKompressCompressor(
            create_kompress_backend(name, **options),
            provider=name,
            config=KompressConfig(enable_ccr=enable_ccr),
            selection_key=key,
        )
    except Exception as exc:
        with _selection_lock:
            first = key not in _failed_selections
            _failed_selections.add(key)
        if first:
            logger.debug("Kompress provider selection failed: %s", describe_exception(exc))
            # Every exception raised above is Headroom's own and worded without
            # option values or third-party messages (the registry wraps factory
            # errors); anything else is named by its class.
            detail = (
                str(exc)
                if isinstance(exc, (BackendError, ValueError, TypeError))
                else describe_class(type(exc))
            )
            logger.error(
                "Kompress: %s is set but cannot be used (%s). ML compression (Kompress) is "
                "disabled; fix or unset %s.",
                KOMPRESS_PROVIDER_ENV,
                detail,
                KOMPRESS_PROVIDER_ENV,
            )
        return None
    logger.info("Kompress: using provider %r", name)
    return compressor


def _reset_provider_selection_for_tests() -> None:
    """Forget reported selection failures and warnings."""
    with _selection_lock:
        _failed_selections.clear()
        _endpoint_overridden_warned.clear()
