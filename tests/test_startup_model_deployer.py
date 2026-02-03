"""Tests for the startup model deployer.

This file contains:
1) Pure unit tests (safe to run anywhere)
2) Optional integration test that calls real endpoints.

The integration test uses your environment variables, so it can run without a
special test server as long as the configured services are reachable.
"""

from __future__ import annotations

import importlib
import os
import asyncio

import pytest


def _maybe_load_dotenv() -> None:
    """Load .env from repo root for local test runs.

    Note: pytest reads only process environment variables. The `.env` file is not
    automatically applied unless you `source` it or a tool (docker compose, dotenv)
    loads it. We load it here to match user expectations during local testing.
    """

    if os.getenv("RUN_STARTUP_MODEL_DEPLOY_E2E") is not None:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    except Exception:
        # If python-dotenv isn't available, we just fall back to the real env.
        return


_maybe_load_dotenv()

_gpustack_client = importlib.import_module("memmachine.common.gpustack_client")
_startup_deployer = importlib.import_module("memmachine.server.startup_model_deployer")

build_gpustack_model_payload = _gpustack_client.build_gpustack_model_payload
pick_model_id = _gpustack_client.pick_model_id
maybe_deploy_init_model = _startup_deployer.maybe_deploy_init_model


def test_pick_model_id_match() -> None:
    models = [
        {"id": 57, "modelName": "Qwen3-Embedding-0.6B", "version": "1.0.0"},
        {"id": 55, "modelName": "Qwen3-4B", "version": "1.0.1"},
    ]
    assert (
        pick_model_id(models, model_name="Qwen3-Embedding-0.6B", version="1.0.0") == 57
    )


def test_pick_model_id_no_match() -> None:
    models = [{"id": 1, "modelName": "A", "version": "0"}]
    assert pick_model_id(models, model_name="B", version="0") is None


def test_build_payload_omits_none_fields() -> None:
    payload = build_gpustack_model_payload(model_name="m", download_url="http://x")

    # explicit None keys should NOT exist
    assert "backend_version" not in payload
    assert "worker_selector" not in payload
    assert "gpu_selector" not in payload

    assert payload["name"] == "m"
    assert payload["download_url"] == "http://x"


@pytest.fixture
def _e2e_env_guard() -> None:
    """Guard to prevent accidental real API calls."""

    if os.getenv("RUN_STARTUP_MODEL_DEPLOY_E2E") != "1":
        pytest.skip("Set RUN_STARTUP_MODEL_DEPLOY_E2E=1 to run this test")


def test_startup_model_deploy_e2e(_e2e_env_guard: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real integration test against your configured endpoints.

    Requirements (env vars):
      - INIT_EMBEDDING_MODEL
      - INIT_EMBEDDING_MODEL_VERSION
      - MODEL_SYSTEM_BASE_URL
      - MODEL_SYSTEM_LIST
      - MODEL_SYSTEM_DOWNLOAD
      - GPUSTACK_SERVICE_HOST
      - GPUSTACK_SERVICE_PORT
      - GPUSTACK_USERNAME
      - GPUSTACK_PASSWORD
      - GPUSTACK_DEPLOY_URL

    Notes:
      - This will create/deploy a model in GPUStack (side-effect!).
      - If your deploy endpoint is not idempotent, repeated runs may fail.
    """

    required = [
        "INIT_EMBEDDING_MODEL",
        "INIT_EMBEDDING_MODEL_VERSION",
        "MODEL_SYSTEM_BASE_URL",
        "MODEL_SYSTEM_LIST",
        "MODEL_SYSTEM_DOWNLOAD",
        "GPUSTACK_SERVICE_HOST",
        "GPUSTACK_SERVICE_PORT",
        "GPUSTACK_USERNAME",
        "GPUSTACK_PASSWORD",
        "GPUSTACK_DEPLOY_URL",
    ]

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing required env vars: {missing}")

    # Make it fail fast in tests
    monkeypatch.setenv("STARTUP_MODEL_DEPLOY_ENABLED", "true")
    monkeypatch.setenv("FAIL_ON_STARTUP_MODEL_DEPLOY", "true")
    monkeypatch.setenv(
        "STARTUP_MODEL_DEPLOY_RETRIES", os.getenv("STARTUP_MODEL_DEPLOY_RETRIES", "1")
    )
    monkeypatch.setenv(
        "STARTUP_MODEL_DEPLOY_BACKOFF_SECONDS",
        os.getenv("STARTUP_MODEL_DEPLOY_BACKOFF_SECONDS", "0"),
    )

    asyncio.run(maybe_deploy_init_model())
