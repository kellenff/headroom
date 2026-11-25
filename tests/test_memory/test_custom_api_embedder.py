"""Custom embedding APIs must work through the SDK, factory, and project router.

The proxy reaches them through the model-backend registry's ``openai`` backend
(``HEADROOM_EMBEDDER_PROVIDER=openai`` or the older ``RUNTIME=openai``); the
``OpenAIEmbedder`` tests cover ``EmbedderBackend.OPENAI`` for library callers.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from headroom.memory.adapters.embedders import OpenAIEmbedder
from headroom.memory.config import EmbedderBackend, MemoryConfig
from headroom.memory.embedder_identity import read_embedder_identity
from headroom.memory.factory import _create_embedder, _reset_embedder_cache_for_tests
from headroom.models.backends import BackendError
from headroom.offline import OfflineEgressBlocked
from headroom.proxy.memory_handler import MemoryConfig as ProxyMemoryConfig
from headroom.proxy.memory_handler import MemoryHandler


@pytest.fixture
def embedding_api(monkeypatch):
    """Serve synthetic vectors over real HTTP without contacting a provider."""
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    requests = []
    responses = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            data = (
                responses.pop(0)
                if responses
                else [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.0, 2.0, 0.0] if "gpu" in text.lower() else [3.0, 0.0, 0.0],
                    }
                    for i, text in reversed(list(enumerate(body["input"])))
                ]
            )
            payload = json.dumps(
                {
                    "object": "list",
                    "model": body["model"],
                    "data": data,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_custom_api_preserves_input_order_and_dimensions(embedding_api):
    url, requests, _ = embedding_api
    embedder = OpenAIEmbedder(
        api_key="test-embedding-key", model_name="custom-embedding", base_url=url, dimension=3
    )
    try:
        single = await embedder.embed("proxy")
        batch = await embedder.embed_batch(["proxy", "", "gpu"])
        np.testing.assert_allclose(single, [1, 0, 0])
        np.testing.assert_allclose(batch, [[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        assert all(vector.dtype == np.float32 for vector in [single, *batch])
        assert len(requests) == 2
        for path, auth, body in requests:
            assert path == "/v1/embeddings"
            assert auth == "Bearer test-embedding-key"
            assert body["model"] == "custom-embedding"
            assert body["encoding_format"] == "float"
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([], "indices"),
        ([{"index": 1, "embedding": [1, 0, 0]}], "indices"),
        ([{"index": 0, "embedding": [1, 0]}], "dimension"),
        ([{"index": 0, "embedding": [float("nan"), 0, 0]}], "finite"),
    ],
)
async def test_custom_api_rejects_invalid_response_before_indexing(embedding_api, data, error):
    url, requests, responses = embedding_api
    responses.append(data)
    embedder = OpenAIEmbedder(api_key="test-key", base_url=url, dimension=3)
    try:
        with pytest.raises(ConnectionError, match=error):
            await embedder.embed("proxy")
        assert len(requests) == 1
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    "override",
    [
        {"openai_base_url": "http://second.invalid/v1"},
        {"openai_api_key": "second-test-key"},
        {"vector_dimension": 4},
    ],
)
def test_custom_api_cache_isolates_endpoint_credentials_and_dimensions(override):
    settings = {
        "embedder_backend": EmbedderBackend.OPENAI,
        "embedder_model": "custom-embedding",
        "openai_base_url": "http://first.invalid/v1",
        "openai_api_key": "first-test-key",
        "vector_dimension": 3,
    }
    _reset_embedder_cache_for_tests()
    try:
        first = _create_embedder(MemoryConfig(**settings))
        assert _create_embedder(MemoryConfig(**settings)) is first
        assert _create_embedder(MemoryConfig(**(settings | override))) is not first
    finally:
        _reset_embedder_cache_for_tests()


@pytest.mark.parametrize(
    "selector",
    [
        {"HEADROOM_EMBEDDER_RUNTIME": "openai"},  # the first release's spelling
        {"HEADROOM_EMBEDDER_PROVIDER": "openai"},
    ],
)
async def test_proxy_custom_api_reaches_project_backends(
    embedding_api, tmp_path, monkeypatch, selector
):
    url, requests, _ = embedding_api
    for name, value in {
        **selector,
        "HEADROOM_EMBEDDER_MODEL": "custom-embedding",
        "HEADROOM_EMBEDDER_BASE_URL": url,
        "HEADROOM_EMBEDDER_API_KEY": "test-proxy-key",
        "HEADROOM_EMBEDDER_DIMENSION": "3",
    }.items():
        monkeypatch.setenv(name, value)
    _reset_embedder_cache_for_tests()
    handler = MemoryHandler(ProxyMemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db")))
    project = None
    try:
        await handler._ensure_initialized()
        assert handler.init_error is None
        # One implementation: the runtime alias and the provider both route
        # through the registry's "openai" backend.
        config = handler._backend._config
        assert config.embedder_backend == "external"
        assert config.embedder_backend_name == "openai"
        assert config.vector_dimension == 3
        assert config.embedder_options["model"] == "custom-embedding"
        project = handler._router._get_or_create_backend(tmp_path / "project" / "memory.db")
        saved = await project.save_memory("proxy settings", user_id="test-user")
        await project.save_memory("gpu settings", user_id="test-user")
        results = await project.search_memories("proxy", user_id="test-user", top_k=1)
        assert results[0].memory.id == saved.id
        assert all(auth == "Bearer test-proxy-key" for _, auth, _ in requests)
        assert all(body["model"] == "custom-embedding" for _, _, body in requests)
        assert requests
        # Both databases are stamped with the embedding space they hold.
        for db in (tmp_path / "memory.db", tmp_path / "project" / "memory.db"):
            assert read_embedder_identity(db) == "openai:custom-embedding:3"
    finally:
        if project is not None:
            await project.close()
        await handler.close()
        _reset_embedder_cache_for_tests()


async def test_proxy_custom_api_key_is_optional_off_openai(embedding_api, tmp_path, monkeypatch):
    """An unauthenticated local server needs no key, and none is invented."""
    url, requests, _ = embedding_api
    for name, value in {
        "HEADROOM_EMBEDDER_PROVIDER": "openai",
        "HEADROOM_EMBEDDER_MODEL": "custom-embedding",
        "HEADROOM_EMBEDDER_BASE_URL": url,
        "HEADROOM_EMBEDDER_DIMENSION": "3",
        # Never sent to another host, even when set.
        "OPENAI_API_KEY": "ambient-openai-key",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HEADROOM_EMBEDDER_API_KEY", raising=False)
    _reset_embedder_cache_for_tests()
    handler = MemoryHandler(ProxyMemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db")))
    try:
        await handler._ensure_initialized()
        assert handler.initialized
        await handler._backend.save_memory("proxy settings", user_id="test-user")
        assert requests
        assert all(auth is None for _, auth, _ in requests)
    finally:
        await handler.close()
        _reset_embedder_cache_for_tests()


@pytest.mark.parametrize("dimension", ["0", "-1", "invalid", ""])
async def test_proxy_rejects_invalid_api_dimension(tmp_path, monkeypatch, dimension):
    monkeypatch.setenv("HEADROOM_EMBEDDER_RUNTIME", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", dimension)
    handler = MemoryHandler(ProxyMemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db")))
    with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_DIMENSION"):
        await handler._init_backend_locked()
    assert not (tmp_path / "memory.db").exists()


def test_custom_api_offline_guard_reports_actual_destination(monkeypatch):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    embedder = OpenAIEmbedder(api_key="test-key", base_url="https://embedding.example/v1")
    with pytest.raises(OfflineEgressBlocked) as error:
        _ = embedder._async_client
    assert error.value.destination == "embedding.example"


@pytest.mark.parametrize(
    "settings",
    [
        {"base_url": "file:///tmp/embeddings"},
        {"base_url": "https://"},
        {"base_url": "https://secret@embedding.example/v1"},
        {"base_url": "https://embedding.example/v1?key=secret"},
        {"dimension": 0},
    ],
)
def test_custom_api_rejects_invalid_connection_settings(settings):
    with pytest.raises(ValueError):
        OpenAIEmbedder(api_key="test-key", **settings)


@pytest.mark.parametrize(
    ("unset", "base_url", "error"),
    [
        # Any server other than OpenAI's needs the model named.
        ("HEADROOM_EMBEDDER_MODEL", "http://embedding.invalid/v1", "model is required"),
        # OpenAI's own endpoint needs a key (HEADROOM_EMBEDDER_API_KEY or OPENAI_API_KEY).
        ("HEADROOM_EMBEDDER_API_KEY", None, "api_key is required"),
    ],
)
async def test_proxy_missing_api_settings_never_falls_back(
    tmp_path, monkeypatch, unset, base_url, error
):
    for name, value in {
        "HEADROOM_EMBEDDER_RUNTIME": "openai",
        "HEADROOM_EMBEDDER_DIMENSION": "3",
        "HEADROOM_EMBEDDER_MODEL": "custom-embedding",
        "HEADROOM_EMBEDDER_API_KEY": "test-key",
    }.items():
        monkeypatch.setenv(name, value)
    if base_url is None:
        monkeypatch.delenv("HEADROOM_EMBEDDER_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_EMBEDDER_BASE_URL", base_url)
    monkeypatch.delenv(unset)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    _reset_embedder_cache_for_tests()
    handler = MemoryHandler(ProxyMemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db")))
    try:
        with pytest.raises(BackendError, match=error):
            await handler._init_backend_locked()
        # The embedder is built before the store, so nothing was created.
        assert not (tmp_path / "memory.db").exists()
    finally:
        _reset_embedder_cache_for_tests()
