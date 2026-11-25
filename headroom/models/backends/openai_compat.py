"""Built-in ``openai`` embedding backend: any OpenAI-compatible ``/embeddings`` API.

One synchronous HTTP client covers OpenAI itself and the self-hosted servers
that speak the same wire format: vLLM, Hugging Face TEI, LM Studio, llama.cpp,
LocalAI, Ollama's ``/v1`` endpoint and most gateways. It needs only ``httpx``,
a core dependency, and not the ``openai`` SDK.

Air gap: :func:`headroom.offline.guard_egress` runs at construction, so an
air-gapped process refuses at startup rather than on the first request, before
the HTTP client is built, and before every request. Loopback endpoints are not
exempt, matching the policy documented on ``guard_egress``: with
``HEADROOM_OFFLINE`` set, use a local model backend instead.

Ambient configuration is deliberately ignored: ``OPENAI_BASE_URL`` is never
read (``headroom wrap`` points it at the Headroom proxy itself), and
``OPENAI_API_KEY`` is used only when the endpoint is api.openai.com, so an
OpenAI key is never sent to another host.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from headroom.log_safety import content_logging_enabled
from headroom.offline import guard_egress

from .embedding import EmbeddingBackendResponseError, _positive_int_or_none
from .registry import BackendError


class EmbeddingBackendRequestError(BackendError, ConnectionError):
    """The embedding API could not be reached, or answered with an HTTP error."""


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_OPENAI_HOST = "api.openai.com"
_PURPOSE = "OpenAI-compatible embedding API"
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 8.0
# What httpx and h11 accept when the request is sent: a header name is an
# RFC 7230 token; a value is ASCII (httpx encodes str values as ASCII), without
# control characters or leading or trailing whitespace.
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_VALUE_RE = re.compile(r"(?:[\x21-\x7e](?:[\t\x20-\x7e]*[\x21-\x7e])?)?")
_BASE_URL_RULE = (
    "base_url must be an http(s) URL with a host, a valid port and no credentials, "
    "query or fragment"
)


def _validate_base_url(base_url: str) -> str:
    try:
        parts = urlsplit(base_url)
        _ = parts.port  # raises ValueError on a non-numeric or out-of-range port
        httpx.URL(base_url)
    except (ValueError, httpx.InvalidURL):
        raise ValueError(_BASE_URL_RULE) from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(_BASE_URL_RULE)
    return base_url.rstrip("/")


def _validate_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in (headers or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("headers must map strings to strings")
        if not _HEADER_NAME_RE.fullmatch(key):
            raise ValueError(
                "headers: each name must be an HTTP token (ASCII letters, digits and "
                "!#$%&'*+-.^_`|~)"
            )
        if not _HEADER_VALUE_RE.fullmatch(value):
            raise ValueError(
                "headers: each value must be printable ASCII without control characters "
                "or leading or trailing whitespace"
            )
        if key.lower() in seen:
            raise ValueError("headers must not repeat a name; header names are case-insensitive")
        seen.add(key.lower())
        out[key] = value
    return out


def _merge_headers(defaults: dict[str, str], custom: dict[str, str]) -> dict[str, str]:
    """``defaults`` overridden by ``custom``, matching header names case-insensitively."""
    overridden = {key.lower() for key in custom}
    merged = {key: value for key, value in defaults.items() if key.lower() not in overridden}
    merged.update(custom)
    return merged


class OpenAICompatibleEmbeddingBackend:
    """Embeddings from an OpenAI-compatible ``POST {base_url}/embeddings`` endpoint.

    Args:
        model: Model id sent to the server. Required unless the endpoint is
            OpenAI's own, where it defaults to ``text-embedding-3-small``.
        base_url: API root including any ``/v1`` prefix. Defaults to
            ``https://api.openai.com/v1``. ``OPENAI_BASE_URL`` is not read.
        api_key: Sent as ``Authorization: Bearer``. For api.openai.com it falls
            back to ``OPENAI_API_KEY`` and is required; for any other host it is
            optional (omit it for unauthenticated local servers) and never taken
            from the environment.
        dimension: Expected output width. Every response is checked against it.
            Memory requires it; relevance learns it from the first response.
        request_dimensions: Also send ``"dimensions": dimension`` in the request,
            asking models that support it (OpenAI ``text-embedding-3-*``) to
            shorten their output. Off by default.
        headers: Extra request headers. A header with the same name as
            ``Authorization`` or ``Content-Type``, in any case, replaces it.
            Names must be HTTP tokens and values printable ASCII.
        timeout: Per-request timeout in seconds.
        max_retries: Retries after the first attempt, for connection errors,
            timeouts, 429 and 5xx responses.
        max_batch_size: Most texts sent in one request.
        max_tokens: The model's input window, exposed as the ``max_tokens``
            attribute. Headroom does not trim inputs to it; callers keep
            texts within the window. Defaults to 8191, OpenAI's limit.

    Raises:
        ValueError: An option is malformed. The message names the option,
            never its value.
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
        request_dimensions: bool = False,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_batch_size: int = 256,
        max_tokens: int = 8191,
    ) -> None:
        resolved = base_url.strip() if base_url and base_url.strip() else DEFAULT_BASE_URL
        self.base_url = _validate_base_url(resolved)
        self.host = urlsplit(self.base_url).hostname or ""
        is_openai = self.host == _OPENAI_HOST

        model = (model or "").strip()
        if not model:
            if not is_openai:
                raise ValueError(
                    "model is required for an OpenAI-compatible endpoint other than api.openai.com"
                )
            model = DEFAULT_OPENAI_MODEL
        self.model_name = model

        if api_key is None and is_openai:
            api_key = os.environ.get("OPENAI_API_KEY") or None
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        if is_openai and self._api_key is None:
            raise ValueError("api_key is required for api.openai.com (or set OPENAI_API_KEY)")
        if self._api_key is not None and not _HEADER_VALUE_RE.fullmatch(self._api_key):
            raise ValueError("api_key must be printable ASCII")

        self.dimension = _positive_int_or_none(dimension, "dimension")
        if request_dimensions and self.dimension is None:
            raise ValueError("request_dimensions=True needs a dimension")
        self.request_dimensions = bool(request_dimensions)
        self._headers = _validate_headers(headers)
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        self.timeout = float(timeout)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.max_retries = max_retries
        self.max_batch_size = _positive_int_or_none(max_batch_size, "max_batch_size")
        self.max_tokens = _positive_int_or_none(max_tokens, "max_tokens")

        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        # Refuse at construction too, so an air-gapped process fails at startup
        # instead of on the first request that needs an embedding.
        guard_egress(_PURPOSE, self.host)

    # ------------------------------------------------------------------ client

    def _get_client(self) -> httpx.Client:
        client = self._client
        if client is not None:
            return client
        with self._lock:
            if self._client is None:
                guard_egress(_PURPOSE, self.host)
                defaults = {"Content-Type": "application/json"}
                if self._api_key:
                    defaults["Authorization"] = f"Bearer {self._api_key}"
                self._client = httpx.Client(
                    base_url=self.base_url,
                    headers=_merge_headers(defaults, self._headers),
                    timeout=self.timeout,
                )
            return self._client

    def close(self) -> None:
        """Close the HTTP client. A later ``embed`` opens a new one."""
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    # ------------------------------------------------------------------ embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in one request and return rows in input order."""
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self.request_dimensions:
            payload["dimensions"] = self.dimension
        body = self._post_with_retry(payload)
        return self._parse(body, len(texts))

    def _post_with_retry(self, payload: dict[str, Any]) -> Any:
        # Re-checked per request: HEADROOM_OFFLINE can be switched on after this
        # backend (and its pooled client) was built and cached.
        guard_egress(_PURPOSE, self.host)
        client = self._get_client()
        attempt = 0
        while True:
            try:
                response = client.post("/embeddings", json=payload)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingBackendRequestError(
                        f"Embedding request to {self.host} failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}"
                    ) from exc
                self._sleep_before_retry(attempt, None)
                attempt += 1
                continue

            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise EmbeddingBackendResponseError(
                        f"Embedding API at {self.host} returned a non-JSON response"
                    ) from exc

            if response.status_code in _RETRY_STATUS and attempt < self.max_retries:
                self._sleep_before_retry(attempt, response.headers.get("retry-after"))
                attempt += 1
                continue
            raise EmbeddingBackendRequestError(
                f"Embedding API at {self.host} returned HTTP {response.status_code}"
                f"{_error_detail(response, self._api_key)}"
            )

    @staticmethod
    def _sleep_before_retry(attempt: int, retry_after: str | None) -> None:
        delay = min(_MAX_BACKOFF_SECONDS, 0.5 * (2**attempt)) * (0.5 + random.random() / 2)
        if retry_after:
            try:
                delay = min(_MAX_BACKOFF_SECONDS, max(delay, float(retry_after)))
            except ValueError:
                pass
        time.sleep(delay)

    def _parse(self, body: Any, count: int) -> list[list[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingBackendResponseError(
                f"Embedding API at {self.host} returned no 'data' list"
            )
        if len(data) != count:
            raise EmbeddingBackendResponseError(
                f"Embedding API at {self.host} returned {len(data)} vectors for {count} inputs"
            )
        has_index = [isinstance(item, dict) and "index" in item for item in data]
        if all(has_index):
            indices = [item["index"] for item in data]
            if sorted(indices) != list(range(count)):
                raise EmbeddingBackendResponseError(
                    f"Embedding API at {self.host} returned indices that do not cover every "
                    "input exactly once"
                )
            ordered = sorted(data, key=lambda item: item["index"])
        elif not any(has_index):
            ordered = data
        else:
            raise EmbeddingBackendResponseError(
                f"Embedding API at {self.host} returned 'index' on some vectors but not others"
            )
        rows: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbeddingBackendResponseError(
                    f"Embedding API at {self.host} returned an item without a float 'embedding' "
                    "list (base64 responses are not supported)"
                )
            rows.append(vector)
        return rows

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleEmbeddingBackend(model={self.model_name!r}, "
            f"base_url={self.base_url!r}, dimension={self.dimension!r})"
        )


def _error_detail(response: httpx.Response, api_key: str | None) -> str:
    """The API's own error message, key redacted, when content logging is on.

    Off by default: servers can echo request input in their errors, and the
    input is prompt or tool content.
    """
    if not content_logging_enabled():
        return ""
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str) or not message.strip():
        return ""
    message = message.strip()
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return f": {message[:200]}"
