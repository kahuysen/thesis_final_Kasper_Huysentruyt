"""Provider/model settings — read & write `.env` (and `os.environ`).

The pipeline already reads every credential and model list from environment
variables (`pipeline/config.py`). This module exposes a structured view of
those values to the UI and persists updates back to:

  1. `os.environ` — so the very next /api/health and /api/runs pick them up,
  2. `5 App/.env`  — so the change survives a server restart.

Schema (returned by `get_settings()`):
    {
      "default_provider": "azure",
      "providers": {
        "anthropic": {
          "label": "...", "endpoint_field": null,
          "api_key_set": true, "api_key_preview": "•••• 12ab",
          "default_model": "claude-opus-4-7",
          "default_model_fallback": "claude-opus-4-7",
          "models": [{"id":"...","label":"..."}, ...],
        },
        "azure": { ..., "endpoint": "https://...", ... },
        "gemini": { ... },
        "openrouter": { ..., "base_url": "https://openrouter.ai/api/v1", ... },
      }
    }
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import set_key, unset_key

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Per-provider mapping: logical field -> env-var name.
PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic (direct API)",
        "endpoint_field": None,
        "fields": {
            "api_key":       "ANTHROPIC_API_KEY",
            "default_model": "DEFAULT_MODEL",
            "models":        "ANTHROPIC_MODELS",
        },
        "default_model_fallback": "claude-opus-4-7",
    },
    "azure": {
        "label": "Azure-hosted Claude",
        "endpoint_field": "endpoint",
        "fields": {
            "api_key":       "AZURE_ANTHROPIC_API_KEY",
            "endpoint":      "AZURE_ANTHROPIC_ENDPOINT",
            "default_model": "AZURE_DEPLOYMENT_NAME",
            "models":        "AZURE_DEPLOYMENTS",
        },
        "default_model_fallback": "claude-opus-4-7",
    },
    "gemini": {
        "label": "Google Gemini",
        "endpoint_field": None,
        "fields": {
            "api_key":       "GEMINI_API_KEY",
            "default_model": "GEMINI_MODEL",
            "models":        "GEMINI_MODELS",
        },
        "default_model_fallback": "gemini-2.5-pro",
    },
    "openrouter": {
        "label": "OpenRouter",
        "endpoint_field": "base_url",
        "fields": {
            "api_key":       "OPENROUTER_API_KEY",
            "base_url":      "OPENROUTER_BASE_URL",
            "default_model": "OPENROUTER_MODEL",
            "models":        "OPENROUTER_MODELS",
        },
        "default_model_fallback": "anthropic/claude-opus-4-7",
    },
}


def _mask_key(val: str) -> str:
    if not val:
        return ""
    if len(val) <= 6:
        return "•" * len(val)
    return f"•••• {val[-4:]}"


def _parse_models(raw: str) -> list[dict]:
    """Parse `id=Label,id2=Label2` -> [{id, label}, ...]."""
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            mid, label = part.split("=", 1)
            mid, label = mid.strip(), label.strip()
        else:
            mid, label = part, part
        if mid:
            out.append({"id": mid, "label": label or mid})
    return out


def _serialize_models(models: list[dict]) -> str:
    parts = []
    for m in models or []:
        mid = (m.get("id") or "").strip()
        if not mid:
            continue
        label = (m.get("label") or "").strip()
        parts.append(f"{mid}={label}" if (label and label != mid) else mid)
    return ",".join(parts)


def get_settings() -> dict:
    out: dict = {
        "default_provider": (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower(),
        "providers": {},
    }
    for prov, spec in PROVIDERS.items():
        info: dict = {
            "label": spec["label"],
            "endpoint_field": spec["endpoint_field"],
            "default_model_fallback": spec["default_model_fallback"],
        }
        for field, env_name in spec["fields"].items():
            val = os.environ.get(env_name, "") or ""
            if field == "api_key":
                info["api_key_set"]     = bool(val)
                info["api_key_preview"] = _mask_key(val)
            elif field == "models":
                info["models"] = _parse_models(val)
            else:
                info[field] = val
        out["providers"][prov] = info
    return out


def _write_env(env_name: str, value: Optional[str]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    if value is None or value == "":
        unset_key(str(ENV_PATH), env_name)
        os.environ.pop(env_name, None)
    else:
        set_key(str(ENV_PATH), env_name, str(value), quote_mode="always")
        os.environ[env_name] = str(value)


def update_provider(provider: str, payload: dict) -> dict:
    """Update one provider. Payload fields:
        api_key (str)            — only written when non-empty
        clear_api_key (bool)     — explicitly remove the stored key
        endpoint (str)           — Azure only
        base_url (str)           — OpenRouter only
        default_model (str)
        models ([{id, label}])
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    spec = PROVIDERS[provider]

    if payload.get("clear_api_key"):
        _write_env(spec["fields"]["api_key"], None)
    elif (payload.get("api_key") or "").strip():
        _write_env(spec["fields"]["api_key"], payload["api_key"].strip())

    for field, env_name in spec["fields"].items():
        if field == "api_key":
            continue
        if field == "models":
            if "models" in payload:
                raw = _serialize_models(payload["models"] or [])
                _write_env(env_name, raw or None)
            continue
        if field in payload:
            val = (payload[field] or "").strip()
            _write_env(env_name, val or None)

    return get_settings()


def set_default_provider(provider: str) -> dict:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    _write_env("LLM_PROVIDER", provider)
    return get_settings()
