
import sys, json, os
sys.path.insert(0, os.path.abspath("."))

# Patch openai's Completions.create BEFORE ChemEagle imports it, so every
# chat-completion call accumulates token usage into _ACCUM. Captures both
# sync and async clients (Azure subclasses these same Completions classes).
_ACCUM = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "model": None}
try:
    from openai.resources.chat.completions import Completions, AsyncCompletions
    _orig_sync = Completions.create
    _orig_async = AsyncCompletions.create

    def _record(resp, kwargs):
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return
            _ACCUM["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            _ACCUM["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            _ACCUM["calls"] += 1
            if _ACCUM["model"] is None:
                _ACCUM["model"] = kwargs.get("model")
        except Exception:
            pass

    def _wrapped_sync(self, *args, **kwargs):
        resp = _orig_sync(self, *args, **kwargs)
        _record(resp, kwargs)
        return resp

    async def _wrapped_async(self, *args, **kwargs):
        resp = await _orig_async(self, *args, **kwargs)
        _record(resp, kwargs)
        return resp

    Completions.create = _wrapped_sync
    AsyncCompletions.create = _wrapped_async
except Exception as _e:
    print(f"[usage-tracker] disabled: {type(_e).__name__}: {_e}")

from main import ChemEagle

img = sys.argv[1]
out_path = sys.argv[2]
usage_path = out_path + ".usage.json" if not out_path.endswith(".json") else out_path[:-5] + ".usage.json"
try:
    result = ChemEagle(img)
    with open(out_path, "w") as f:
        json.dump(result if isinstance(result, dict) else {"_raw": str(result)}, f, indent=2, ensure_ascii=False, default=str)
    print(f"OK {out_path}")
except Exception as e:
    import traceback
    err = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    with open(out_path, "w") as f:
        json.dump(err, f, indent=2)
    print(f"ERR {out_path}: {type(e).__name__}: {e}")
    raise
finally:
    try:
        with open(usage_path, "w") as f:
            json.dump({
                "model": _ACCUM["model"],
                "prompt_tokens": _ACCUM["prompt_tokens"],
                "completion_tokens": _ACCUM["completion_tokens"],
                "total_tokens": _ACCUM["prompt_tokens"] + _ACCUM["completion_tokens"],
                "calls": _ACCUM["calls"],
            }, f, indent=2)
    except Exception:
        pass
