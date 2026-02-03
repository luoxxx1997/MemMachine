"""Startup-time model deployer.

Reads environment variables to:
- query the external model registry
- resolve a model download URL
- login to GPUStack
- deploy the model to GPUStack

This is best-effort by default: failures are logged and won't block MemMachine
startup unless `FAIL_ON_STARTUP_MODEL_DEPLOY=true` is set.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..common.gpustack_client import (
    GPUStackClient,
    GPUStackConfig,
    ModelRegistryConfig,
    build_gpustack_model_payload,
    fetch_model_download_url,
    fetch_model_list,
    pick_model_id,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


async def maybe_deploy_init_model() -> None:
    """Best-effort init model deployment.

    Contract:
    - If INIT_EMBEDDING_MODEL(+_VERSION) is set, deploy it to GPUStack.
    - If INIT_LLM_MODEL(+_VERSION) is set, deploy it to GPUStack.
    - If none are set, no-op.

    Note: by default, failures are logged and startup continues unless
    FAIL_ON_STARTUP_MODEL_DEPLOY=true.
    """

    enabled = _env_bool("STARTUP_MODEL_DEPLOY_ENABLED", default=True)
    fail_hard = _env_bool("FAIL_ON_STARTUP_MODEL_DEPLOY", default=False)

    logger.info(
        "Startup model deploy step entered: enabled=%s fail_hard=%s",
        enabled,
        fail_hard,
    )

    if not enabled:
        logger.info("Startup model deploy disabled (STARTUP_MODEL_DEPLOY_ENABLED=false)")
        return

    # ---- entry: deploy embedding + llm (if configured) ----
    targets: list[tuple[str, str, str]] = []

    emb_name = _env("INIT_EMBEDDING_MODEL")
    emb_ver = _env("INIT_EMBEDDING_MODEL_VERSION")
    if emb_name and emb_ver:
        targets.append(("embedding", emb_name, emb_ver))

    llm_name = _env("INIT_LLM_MODEL")
    llm_ver = _env("INIT_LLM_MODEL_VERSION")
    if llm_name and llm_ver:
        targets.append(("llm", llm_name, llm_ver))

    if not targets:
        logger.info(
            "Startup model deploy skipped (no INIT_EMBEDDING_MODEL/INIT_LLM_MODEL configured)"
        )
        return

    logger.info(
        "Startup model deploy targets: %s",
        ", ".join([f"{k}:{n}:{v}" for k, n, v in targets]),
    )

    async def _deploy_one(*, model_name: str, model_version: str) -> None:
        registry_base = _env("MODEL_SYSTEM_BASE_URL")
        list_path = _env("MODEL_SYSTEM_LIST")
        download_path_tpl = _env("MODEL_SYSTEM_DOWNLOAD", "/models/{id}/download")

        gpustack_host = _env("GPUSTACK_SERVICE_HOST")
        gpustack_port = _env("GPUSTACK_SERVICE_PORT", "80")
        gpustack_user = _env("GPUSTACK_USERNAME")
        gpustack_pass = _env("GPUSTACK_PASSWORD")
        gpustack_deploy = _env("GPUSTACK_DEPLOY_URL", "/v2/models")

        if not registry_base or not list_path:
            raise RuntimeError(
                "MODEL_SYSTEM_BASE_URL or MODEL_SYSTEM_LIST not set; cannot deploy init model"
            )
        if not gpustack_host:
            raise RuntimeError("GPUSTACK_SERVICE_HOST not set; cannot deploy init model")

        registry_conf = ModelRegistryConfig(
            base_url=registry_base.rstrip("/"),
            list_path=list_path,
            download_path_template=download_path_tpl,
        )

        gpustack_base_url = f"http://{gpustack_host}:{gpustack_port}".rstrip("/")
        gpustack_conf = GPUStackConfig(
            base_url=gpustack_base_url,
            username=gpustack_user,
            password=gpustack_pass,
            deploy_path=gpustack_deploy,
        )

        logger.info(
            "Startup model deploy config: registry_base=%s list_path=%s download_path_tpl=%s gpustack_base_url=%s deploy_path=%s",
            registry_base,
            list_path,
            download_path_tpl,
            gpustack_base_url,
            gpustack_deploy,
        )

        retries = int(_env("STARTUP_MODEL_DEPLOY_RETRIES", "3") or 3)
        backoff = float(_env("STARTUP_MODEL_DEPLOY_BACKOFF_SECONDS", "2") or 2)
        watch_timeout = float(_env("GPUSTACK_WATCH_TIMEOUT_SECONDS", "600") or 600)

        client = GPUStackClient(gpustack_conf)
        try:
            last_err: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    logger.info(
                        "Startup model deploy attempt %s/%s: resolve %s:%s",
                        attempt,
                        retries,
                        model_name,
                        model_version,
                    )

                    models = await fetch_model_list(registry_conf)
                    mids = pick_model_id(models, model_name=model_name, version=model_version)
                    if mids is None:
                        raise RuntimeError(
                            f"Model not found in registry: modelName={model_name} version={model_version}"
                        )

                    download_url = await fetch_model_download_url(registry_conf, mids)
                    payload = build_gpustack_model_payload(
                        model_name=model_name, download_url=download_url
                    )

                    ok = await client.login()
                    if not ok:
                        raise RuntimeError("GPUStack login failed")

                    resp = await client.deploy_model(payload)
                    if resp.status_code >= 400:
                        raise RuntimeError(f"GPUStack deploy failed: HTTP {resp.status_code}")

                    # Extract model id (best-effort): JSON {id: ...} or Location header.
                    deployed_model_id = None
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("id") is not None:
                            deployed_model_id = data.get("id")
                    except Exception:
                        deployed_model_id = None

                    if deployed_model_id is None:
                        loc = resp.headers.get("Location")
                        if loc and loc.rstrip("/").split("/")[-1].isdigit():
                            deployed_model_id = int(loc.rstrip("/").split("/")[-1])

                    logger.info(
                        "Startup model deploy request accepted for %s:%s (model_id=%s)",
                        model_name,
                        model_version,
                        deployed_model_id,
                    )

                    if deployed_model_id is not None:
                        async for event in client.watch_model_instances(
                            model_id=deployed_model_id, timeout=watch_timeout
                        ):
                            payload_data = event.get("data") if isinstance(event, dict) else None
                            if isinstance(payload_data, dict):
                                state = str(payload_data.get("state", "")).lower()
                                state_message = str(payload_data.get("state_message", "") or "")
                            else:
                                state = str(event.get("state", "")).lower()
                                state_message = str(event.get("state_message", "") or "")

                            if state == "running":
                                logger.info(
                                    "Startup model is running: %s:%s (model_id=%s)",
                                    model_name,
                                    model_version,
                                    deployed_model_id,
                                )
                                return
                            if state == "error":
                                raise RuntimeError(
                                    f"GPUStack instance deployment error (model_id={deployed_model_id}): {state_message}"
                                )

                    # If we can't watch, consider accepted.
                    logger.info(
                        "Startup model deploy completed (no model_id to watch): %s:%s",
                        model_name,
                        model_version,
                    )
                    return

                except Exception as e:
                    last_err = e
                    logger.warning("Startup model deploy attempt failed: %s", e)
                    if attempt < retries:
                        await asyncio.sleep(backoff * attempt)

            raise RuntimeError(
                f"Startup model deploy failed after {retries} attempts: {last_err}"
            ) from last_err
        finally:
            await client.aclose()

    for kind, name, ver in targets:
        try:
            logger.info("Startup deploy %s model: %s:%s", kind, name, ver)
            await _deploy_one(model_name=name, model_version=ver)
        except Exception as e:
            if fail_hard:
                raise
            logger.warning("Startup deploy %s model failed: %s", kind, e)

    return
