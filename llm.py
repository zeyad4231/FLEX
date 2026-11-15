"""LLM wrapper: every GPT call goes through chat(). Caches answers and tracks cost."""
import os
import json
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Default model for debugging. Final runs: set FLEX_MODEL=gpt-4o in .env
MODEL = os.getenv("FLEX_MODEL", "gpt-4o-mini")

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# USD per 1M tokens (input, output). Check against OpenAI's pricing page.
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}

usage = {"api_calls": 0, "cache_hits": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0}
_client = None


def _client_get():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def chat(prompt, system=None, model=None, temperature=0.0, sample_id=0):
    """Send one prompt, return the text answer.

    sample_id: use different values to get several independent samples of the
    same prompt (needed for improvement 3); each sample is cached separately.
    """
    model = model or MODEL
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]

    key_data = {"model": model, "messages": messages, "temperature": temperature, "sample_id": sample_id}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        usage["cache_hits"] += 1
        return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

    for attempt in range(5):  # retry on network / rate-limit errors
        try:
            r = _client_get().chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"[llm] error: {e} -> retrying in {wait}s")
            time.sleep(wait)

    text = r.choices[0].message.content
    tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    pin, pout = next((PRICES[k] for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), (0.0, 0.0))
    cost = (tin * pin + tout * pout) / 1e6

    usage["api_calls"] += 1
    usage["in_tokens"] += tin
    usage["out_tokens"] += tout
    usage["cost_usd"] += cost

    cache_file.write_text(json.dumps({
        "model_requested": model,
        "model_served": r.model,  # exact snapshot, for the report
        "messages": messages,
        "temperature": temperature,
        "sample_id": sample_id,
        "response": text,
        "tokens": {"in": tin, "out": tout},
        "cost_usd": cost,
    }, indent=2), encoding="utf-8")
    return text


def print_usage():
    u = usage
    print(f"[llm] model={MODEL} | api calls={u['api_calls']} | cache hits={u['cache_hits']} | "
          f"tokens in/out={u['in_tokens']}/{u['out_tokens']} | cost=${u['cost_usd']:.5f}")


if __name__ == "__main__":
    print(chat("Reply with exactly: FLEX ready"))
    print_usage()
    print(chat("Reply with exactly: FLEX ready"))  # same prompt -> should come from cache
    print_usage()