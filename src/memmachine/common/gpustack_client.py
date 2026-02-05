"""Small async client for GPUStack and external model registry.

This module is intentionally lightweight and only depends on httpx.
It is used during service startup to optionally auto-deploy a model into GPUStack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRegistryConfig:
    base_url: str
    list_path: str
    download_path_template: str


@dataclass(frozen=True)
class GPUStackConfig:
    base_url: str
    username: str | None
    password: str | None
    deploy_path: str


class GPUStackClient:
    """Async GPUStack client that stores auth cookies inside an httpx client."""

    def __init__(
            self,
            conf: GPUStackConfig,
            *,
            timeout_seconds: float = 30.0,
            verify: bool = True,
    ) -> None:
        self.conf = conf
        self._timeout = httpx.Timeout(timeout_seconds)
        self._verify = verify
        self._client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, verify=self._verify)
        return self._client

    async def login(self) -> bool:
        """Login to GPUStack and persist cookies in the session."""
        if not self.conf.username or not self.conf.password:
            logger.warning("GPUStack username/password not provided; skip login")
            return False

        try:
            client = await self._ensure_client()
            login_url = f"{self.conf.base_url}/auth/login"
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {"username": self.conf.username, "password": self.conf.password}
            resp = await client.post(login_url, headers=headers, data=data)
            if resp.status_code == 200:
                cookies_dict = dict(client.cookies)
                if cookies_dict:
                    cookie_names = list(cookies_dict.keys())
                    logger.info("GPUStack login ok; saved cookies: %s", cookie_names)
                else:
                    logger.warning("GPUStack login ok but no cookies returned")
                return True

            logger.warning("GPUStack login failed: HTTP %s", resp.status_code)
            return False
        except Exception as e:
            logger.warning("GPUStack login error: %s", e)
            return False

    async def deploy_model(self, payload: dict[str, Any]) -> httpx.Response:
        """Backward-compatible alias: deploy/create a model."""
        return await self.create_model(payload)

    async def create_model(self, payload: dict[str, Any]) -> httpx.Response:
        client = await self._ensure_client()
        url = f"{self.conf.base_url}{self.conf.deploy_path}"
        return await client.post(url, json=payload)

    async def list_models(self) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        url = f"{self.conf.base_url}{self.conf.deploy_path}"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def get_model(self, model_id: int | str) -> dict[str, Any] | None:
        client = await self._ensure_client()
        url = f"{self.conf.base_url}{self.conf.deploy_path}/{model_id}"
        resp = await client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None

    async def delete_model(self, model_id: int | str) -> bool:
        client = await self._ensure_client()
        url = f"{self.conf.base_url}{self.conf.deploy_path}/{model_id}"
        resp = await client.delete(url)
        return resp.status_code < 400

    async def get_model_instances(self, model_id: int | str) -> list[dict[str, Any]]:
        """Return instances for a given model.

        GPUStack commonly exposes: /v2/models/{id}/instances
        """
        client = await self._ensure_client()
        url = f"{self.conf.base_url}{self.conf.deploy_path}/{model_id}/instances"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def watch_model_instances(
            self,
            *,
            model_id: int | str,
            timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Watch model instances via SSE.

        GPUStack can stream events from `GET /v2/model-instances?watch=true`.

        Each yielded item is the parsed JSON dict of a single event.
        The event may either be:
        - an SSE line:   data: {"type": ..., "data": {...}}
        - a raw JSON line: {"type": ..., "data": {...}}

        Note: we currently don't filter by `model_id` at the HTTP level since the
        watch endpoint is global; callers can filter by `event['data']['model_id']`.
        """

        client = await self._ensure_client()
        url = f"{self.conf.base_url}/v2/model-instances?watch=true"
        if model_id is not None:
            url += f"&model_id={model_id}"

        headers = {"Accept": "text/event-stream"}

        async with client.stream("GET", url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                raw = line.strip()
                if raw.startswith("data:"):
                    raw = raw[len("data:"):].strip()

                if not raw:
                    continue

                try:
                    event = httpx.Response(200, content=raw).json()
                except Exception:
                    continue

                if isinstance(event, dict):
                    # Optional: filter stream for given model_id when possible.
                    # Event shape: {"type": 1, "data": {"model_id": 13, ...}}
                    d = event.get("data")
                    if isinstance(d, dict) and d.get("model_id") is not None:
                        try:
                            if str(d.get("model_id")) != str(model_id):
                                continue
                        except Exception:
                            pass
                    yield event


async def fetch_model_list(
        conf: ModelRegistryConfig,
        *,
        timeout_seconds: float = 30.0,
        verify: bool = True,
) -> list[dict[str, Any]]:
    url = f"{conf.base_url}{conf.list_path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), verify=verify) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("Model list response is not a JSON list")
        # ensure every item is a dict
        return [x for x in data if isinstance(x, dict)]


async def fetch_model_download_url(
        conf: ModelRegistryConfig,
        model_id: int,
        *,
        timeout_seconds: float = 30.0,
        verify: bool = True,
) -> str:
    path = conf.download_path_template.format(id=model_id)
    url = f"{conf.base_url}{path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), verify=verify) as client:
        resp = await client.get(url)
        resp.raise_for_status()

        # The model system may return a raw UTF-8 text body containing the URL.
        # Prefer text parsing first.
        text = (resp.text or "").strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text

        # Fallback: some deployments return JSON.
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(
                "Download URL response is neither a URL text body nor valid JSON"
            ) from e

        # Support common shapes: {"url": "..."} or plain string
        if isinstance(data, str):
            data = data.strip()
            if data:
                return data
        if isinstance(data, dict):
            for key in ("download_url", "url", "data"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        raise ValueError("Download URL response does not contain a usable URL")


def build_gpustack_model_payload(*, model_name: str, download_url: str) -> dict[str, Any]:
    """Build the GPUStack deploy payload.

    Note: keys with value None are omitted to satisfy strict schema validators.
    """

    payload: dict[str, Any] = {
        "name": model_name,
        "source": "download_url",
        "cluster_id": 1,
        "backend": "vLLM",
        "replicas": 1,
        "extended_kv_cache": {"enabled": False},
        "speculative_config": {"enabled": False},
        "categories": ["imported"],
        "backend_parameters": ["--gpu-memory-utilization=0.4"],
        "distributed_inference_across_workers": True,
        "restart_on_error": True,
        "generic_proxy": False,
        "download_url_model_name": model_name,
        "download_url": download_url,
        "placement_strategy": "spread",
    }

    # only set these keys when not None / not empty
    # (some API servers reject explicit nulls)
    # backend_version
    # worker_selector
    # gpu_selector

    return payload


def pick_model_id(
        models: list[dict[str, Any]], *, model_name: str, version: str
) -> int | None:
    for m in models:
        if str(m.get("modelName")) == model_name and str(m.get("version")) == version:
            try:
                return int(m.get("id"))
            except Exception:
                return None
    return None
