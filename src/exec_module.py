"""
Execution Module — manages API interactions with LLM providers.

Handles authentication, request submission, and response parsing
for OpenAI, DeepSeek, and Google Gemini.

Includes:
  - Provider-specific run functions (run_openai, run_deepseek, run_gemini)
  - Shared execute_single() for single-turn prompt execution
  - Automatic retry with backoff for connection errors
"""

import os
import sys
import time
import functools
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Lazy import to avoid circular dependency at module level
import src.formatter as fmt

client_openai = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=300.0,
)
client_deepseek = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
    timeout=300.0,
)
client_gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    http_options={"timeout": 300_000},  # milliseconds
)


# ---------------------------------------------------------------------------
# Retry decorator for connection errors
# ---------------------------------------------------------------------------

# Exception types that indicate a connection/network problem
_CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Try to import provider-specific connection exceptions
try:
    from openai import APIConnectionError, APITimeoutError
    _CONNECTION_ERRORS = _CONNECTION_ERRORS + (APIConnectionError, APITimeoutError)
except ImportError:
    pass

# Also catch provider-specific connection errors by message
_CONNECTION_KEYWORDS = (
    "connection error",
    "connection reset",
    "connection refused",
    "connection aborted",
    "timed out",
    "timeout",
    "network",
    "unreachable",
    "temporary failure",
    "name resolution",
    "eof occurred",
    "ssl",
    "429",
    "rate limit",
    "too many requests",
)

RETRY_WAIT_SECONDS = 300  # 5 minutes between retries
MAX_RETRIES = 50           # ~4 hours max total wait time


def _is_connection_error(exc: Exception) -> bool:
    """Check if an exception is a connection/network error."""
    if isinstance(exc, _CONNECTION_ERRORS):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _CONNECTION_KEYWORDS)


def _log_retry(msg: str):
    """Write retry message so it's visible even in subprocess."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def with_retry(func):
    """Decorator that retries on connection errors with 5-min backoff."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not _is_connection_error(e):
                    raise  # non-connection error, propagate immediately

                _log_retry(
                    f"  ⚠ Connection error: {e}\n"
                    f"    Retry {attempt}/{MAX_RETRIES} — "
                    f"waiting {RETRY_WAIT_SECONDS}s..."
                )
                time.sleep(RETRY_WAIT_SECONDS)

        # exhausted all retries
        raise ConnectionError(
            f"Failed after {MAX_RETRIES} retries "
            f"({MAX_RETRIES * RETRY_WAIT_SECONDS}s total)"
        )

    return wrapper


# ---------------------------------------------------------------------------
# Provider execution functions (low-level)
# ---------------------------------------------------------------------------

@with_retry
def run_openai(model: str, input_items, **params):
    resp = client_openai.responses.create(model=model, input=input_items, **params)
    text = getattr(resp, "output_text", None) or ""
    return {"provider": "openai", "model": model, "raw": resp, "text": text}


@with_retry
def run_deepseek(model: str, messages, **params):
    resp = client_deepseek.chat.completions.create(model=model, messages=messages, **params)
    text = resp.choices[0].message.content if resp.choices else ""
    return {"provider": "deepseek", "model": model, "raw": resp, "text": text}


@with_retry
def run_gemini(model: str, messages, **params):
    resp = client_gemini.models.generate_content(
        model=model,
        contents=messages,
        config=params
    )
    text = getattr(resp, "text", None) or ""
    return {"provider": "gemini", "model": model, "raw": resp, "text": text}


# ---------------------------------------------------------------------------
# Shared execution helper (high-level)
# ---------------------------------------------------------------------------

def execute_single(
    prompt_text: str,
    msg_type: str,
    provider: str,
    model: str,
    params: dict,
) -> dict:
    """Send a single prompt through the formatting and execution pipeline.

    This is the shared entry point used by RF1, RF2, RF4, RT1, and RT2
    campaign scripts, eliminating per-module execute_prompt() duplication.

    Parameters
    ----------
    prompt_text : str
        The prompt text to send.
    msg_type : str
        Message type for formatting (e.g. 'counterfactual', 'benefit_request').
        Must be a key in formatter.TYPE_TO_OPENAI_ROLE.
    provider : str
        Provider key ('openai', 'deepseek', 'gemini').
    model : str
        Model identifier for the provider.
    params : dict
        Must contain 'system_prompt' key. May contain provider-specific
        sub-dicts ('openai', 'deepseek', 'gemini') with extra parameters.

    Returns
    -------
    dict with keys: provider, model, raw, text
    """
    messages = [{"type": msg_type, "text": prompt_text}]
    turns = fmt.normalize_to_turns(messages, system_prompt=params.get("system_prompt"))

    if provider == "openai":
        payload = fmt.format_openai(model, turns, **params.get("openai", {}))
        return run_openai(model, payload["input"], **params.get("openai", {}))

    if provider == "deepseek":
        payload = fmt.format_deepseek(model, turns, **params.get("deepseek", {}))
        return run_deepseek(model, payload["messages"], **params.get("deepseek", {}))

    if provider == "gemini":
        payload = fmt.format_gemini(model, turns, **params.get("gemini", {}))
        return run_gemini(model, payload["contents"], **params.get("gemini", {}))

    raise ValueError(f"Unknown provider: {provider}")