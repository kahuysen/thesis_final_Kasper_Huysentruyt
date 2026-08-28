"""Single-image ChemEagle runner, adapted to OpenRouter.

Identical to `_run_one_image.py` (usage tracking via patched
Completions.create) plus two documented baseline adaptations, made
without touching the vendored code:

1. `main.AzureOpenAI` is replaced by a factory returning an OpenAI
   client pointed at OpenRouter (no Azure credentials exist anymore).
2. Bare model ids (`gpt-5-mini`) are rewritten to OpenRouter form
   (`openai/gpt-5-mini`) inside the patched `create` call, so the same
   engine family ChemEagle ships with serves the pipeline.

Usage: .venv-chemeagle/bin/python3 _run_one_image_openrouter.py <img> <out.json>
Env:   OPENROUTER_API_KEY (read from ../../approaches/single_agent_sdk/.env
       if not already set).
"""
import sys, json, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) or "."))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── credentials: reuse the project .env ─────────────────────────────────
if not os.getenv("OPENROUTER_API_KEY"):
    env_path = os.path.join("..", "..", "approaches", "single_agent_sdk", ".env")
    for line in open(env_path):
        if line.startswith("OPENROUTER_API_KEY="):
            os.environ["OPENROUTER_API_KEY"] = line.strip().split("=", 1)[1]
# main.py insists on these being set; values are unused after the patch.
os.environ.setdefault("API_KEY", "unused-patched")
os.environ.setdefault("AZURE_ENDPOINT", "https://unused.invalid")
os.environ.setdefault("API_VERSION", "unused")

_ACCUM = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "models": []}

from openai.resources.chat.completions import Completions, AsyncCompletions
_orig_sync = Completions.create
_orig_async = AsyncCompletions.create


def _rewrite(kwargs):
    m = kwargs.get("model")
    if m and "/" not in m:
        kwargs["model"] = f"openai/{m}"
    return kwargs


def _record(resp, kwargs):
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        _ACCUM["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _ACCUM["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        _ACCUM["calls"] += 1
        if kwargs.get("model") not in _ACCUM["models"]:
            _ACCUM["models"].append(kwargs.get("model"))
    except Exception:
        pass


def _wrapped_sync(self, *args, **kwargs):
    resp = _orig_sync(self, *args, **_rewrite(kwargs))
    _record(resp, kwargs)
    return resp


async def _wrapped_async(self, *args, **kwargs):
    resp = await _orig_async(self, *args, **_rewrite(kwargs))
    _record(resp, kwargs)
    return resp


Completions.create = _wrapped_sync
AsyncCompletions.create = _wrapped_async

# ── replace the Azure client with OpenRouter, SDK-wide ──────────────────
# Six vendored modules each do `from openai import AzureOpenAI` and build
# their own client; patching the attribute on the openai package BEFORE any
# ChemEagle import means every one of them binds this factory instead.
import openai
from openai import OpenAI


def _openrouter_client(**_ignored):
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=600,
    )


openai.AzureOpenAI = _openrouter_client

from main import ChemEagle  # noqa: E402

img = sys.argv[1]
out_path = sys.argv[2]
usage_path = out_path[:-5] + ".usage.json" if out_path.endswith(".json") else out_path + ".usage.json"
try:
    result = ChemEagle(img)
    with open(out_path, "w") as f:
        json.dump(result if isinstance(result, dict) else {"_raw": str(result)},
                  f, indent=2, ensure_ascii=False, default=str)
    status = "ok"
except Exception as e:
    status = f"error: {type(e).__name__}: {e}"
    print(status, file=sys.stderr)
finally:
    with open(usage_path, "w") as f:
        json.dump({**_ACCUM, "status": status, "image": img}, f, indent=2)
sys.exit(0 if status == "ok" else 1)
