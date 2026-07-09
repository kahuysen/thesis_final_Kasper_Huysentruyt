"""Centralized client + model config.

Reads from `.env` (via python-dotenv). Three backends; any subset can be
configured at the same time and the UI lets the user pick per-request:

  - direct Anthropic API:     ANTHROPIC_API_KEY  (+ ANTHROPIC_MODELS / DEFAULT_MODEL)
  - Azure-hosted Claude:      AZURE_ANTHROPIC_ENDPOINT
                              AZURE_ANTHROPIC_API_KEY
                              AZURE_DEPLOYMENT_NAME / AZURE_DEPLOYMENTS
  - Google Gemini:            GEMINI_API_KEY  (+ GEMINI_MODEL / GEMINI_MODELS)

LLM_PROVIDER picks the *default* provider used when a request doesn't
specify one. It does NOT restrict which providers are available — the UI
shows every provider whose credentials are present in .env.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env once on import. Idempotent — safe under pytest/Flask reloads.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FALLBACK_MODEL = "claude-opus-4-7"
FALLBACK_GEMINI_MODEL = "gemini-2.5-pro"

ANTHROPIC = "anthropic"
AZURE = "azure"
GEMINI = "gemini"
OPENROUTER = "openrouter"
FALLBACK_OPENROUTER_MODEL = "anthropic/claude-opus-4-7"


def _default_provider() -> str:
    return (os.environ.get("LLM_PROVIDER") or ANTHROPIC).strip().lower()


def current_provider() -> str:
    """Default provider used when a request omits one. One of: anthropic, azure, gemini."""
    return _default_provider()


# ---------- credential checks ----------

def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_azure() -> bool:
    return bool(
        os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        and os.environ.get("AZURE_ANTHROPIC_API_KEY")
    )


def _has_gemini() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _has_openrouter() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


# ---------- client construction ----------

def build_client_for(provider: str) -> Any:
    """Construct an SDK client for a specific provider.

    Returns an Anthropic client for anthropic/azure, or a `google.genai.Client`
    for gemini. Raises RuntimeError if that provider's credentials are missing.
    """
    provider = provider.strip().lower()

    if provider == AZURE:
        if not _has_azure():
            raise RuntimeError(
                "Azure not configured: set AZURE_ANTHROPIC_ENDPOINT and "
                "AZURE_ANTHROPIC_API_KEY in .env"
            )
        endpoint = os.environ["AZURE_ANTHROPIC_ENDPOINT"]
        key = os.environ["AZURE_ANTHROPIC_API_KEY"]
        return Anthropic(
            base_url=endpoint.rstrip("/") + "/",
            api_key=key,
            default_headers={"api-key": key},
        )

    if provider == GEMINI:
        if not _has_gemini():
            raise RuntimeError("Gemini not configured: set GEMINI_API_KEY in .env")
        from google import genai  # lazy import: anthropic-only setups don't need it
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    if provider == ANTHROPIC:
        if not _has_anthropic():
            raise RuntimeError(
                "Anthropic direct API not configured: set ANTHROPIC_API_KEY in .env"
            )
        return Anthropic()

    if provider == OPENROUTER:
        if not _has_openrouter():
            raise RuntimeError(
                "OpenRouter not configured: set OPENROUTER_API_KEY in .env"
            )
        from openai import OpenAI  # lazy import; openrouter uses the OpenAI SDK
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    raise RuntimeError(f"Unknown provider: {provider!r}")


def build_client() -> Any:
    """Build a client for the default provider (LLM_PROVIDER)."""
    return build_client_for(_default_provider())


# ---------- model lists ----------

def _parse_model_list(raw: str) -> list[tuple[str, str]]:
    """Parse a comma-separated `id=label` list. `id` alone is fine too."""
    out: list[tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            mid, label = part.split("=", 1)
            mid, label = mid.strip(), label.strip()
        else:
            mid, label = part, part
        if mid:
            out.append((mid, label or mid))
    return out


def _models_for(
    provider: str, env_list: str, env_default: str, fallback: str
) -> list[dict]:
    """Build the full model list for one provider, including the default."""
    raw = os.environ.get(env_list, "")
    pairs = _parse_model_list(raw)
    default = os.environ.get(env_default) or fallback
    if default and not any(mid == default for mid, _ in pairs):
        pairs.insert(0, (default, default))
    return [{"id": mid, "label": label, "provider": provider} for mid, label in pairs]


def default_model() -> str:
    """Model id that the default provider's loop will use when none is requested."""
    p = _default_provider()
    if p == AZURE:
        return os.environ.get("AZURE_DEPLOYMENT_NAME") or FALLBACK_MODEL
    if p == GEMINI:
        return os.environ.get("GEMINI_MODEL") or FALLBACK_GEMINI_MODEL
    return os.environ.get("DEFAULT_MODEL") or FALLBACK_MODEL


def available_models() -> list[dict]:
    """Every model from every provider whose credentials are present in .env.

    Each entry is `{id, label, provider}`. The current default provider's
    models come first so the picker's first option is the natural default.
    """
    blocks = []
    if _has_anthropic():
        blocks.append(
            (ANTHROPIC, _models_for(ANTHROPIC, "ANTHROPIC_MODELS", "DEFAULT_MODEL", FALLBACK_MODEL))
        )
    if _has_azure():
        blocks.append(
            (AZURE, _models_for(AZURE, "AZURE_DEPLOYMENTS", "AZURE_DEPLOYMENT_NAME", FALLBACK_MODEL))
        )
    if _has_gemini():
        blocks.append(
            (GEMINI, _models_for(GEMINI, "GEMINI_MODELS", "GEMINI_MODEL", FALLBACK_GEMINI_MODEL))
        )
    if _has_openrouter():
        blocks.append(
            (OPENROUTER, _models_for(OPENROUTER, "OPENROUTER_MODELS", "OPENROUTER_MODEL", FALLBACK_OPENROUTER_MODEL))
        )

    default = _default_provider()
    blocks.sort(key=lambda kv: 0 if kv[0] == default else 1)
    return [m for _, ms in blocks for m in ms]


def is_model_allowed(model_id: str, provider: Optional[str] = None) -> bool:
    """Validate a (model, provider) pair against what .env declares."""
    for m in available_models():
        if m["id"] != model_id:
            continue
        if provider is None or m["provider"] == provider:
            return True
    return False


def provider_for_model(model_id: str) -> Optional[str]:
    """Reverse lookup: which provider does a given model id belong to?

    If the same id is declared for multiple providers (e.g. you list the
    same Claude model under both `anthropic` and `azure`), the default
    provider wins.
    """
    matches = [m for m in available_models() if m["id"] == model_id]
    if not matches:
        return None
    default = _default_provider()
    for m in matches:
        if m["provider"] == default:
            return m["provider"]
    return matches[0]["provider"]


# ---------- diagnostics ----------

def describe() -> str:
    """One-line description of the current default backend, for logging."""
    p = _default_provider()
    if p == AZURE:
        ep = os.environ.get("AZURE_ANTHROPIC_ENDPOINT", "(unset)")
        return f"azure: {ep}  (deployment={default_model()})"
    if p == GEMINI:
        return f"gemini: generativelanguage.googleapis.com  (model={default_model()})"
    return f"anthropic: api.anthropic.com  (model={default_model()})"
