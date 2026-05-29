"""Langfuse tracing integration for the LangChain backend."""

from __future__ import annotations

import logging
import os
from typing import Optional

from .config import LANGFUSE_PUBLIC_KEY_ENV, LANGFUSE_SECRET_KEY_ENV

log = logging.getLogger("tracing.lc")

_handler: Optional[object] = None
_checked: bool = False


def get_langfuse_handler() -> Optional[object]:
    """Return a shared Langfuse CallbackHandler, or None if not configured.

    Reads LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from the environment.
    The handler is safe to reuse across requests — it reads the current
    trace context (set by @observe) at invocation time.
    """
    global _handler, _checked

    if _checked:
        return _handler

    _checked = True

    public_key = os.environ.get(LANGFUSE_PUBLIC_KEY_ENV)
    secret_key = os.environ.get(LANGFUSE_SECRET_KEY_ENV)

    if not public_key or not secret_key:
        log.info("Langfuse not configured — set %s and %s to enable tracing",
                 LANGFUSE_PUBLIC_KEY_ENV, LANGFUSE_SECRET_KEY_ENV)
        return None

    try:
        from langfuse.langchain import CallbackHandler
        _handler = CallbackHandler()
        log.info("Langfuse tracing enabled")
        return _handler
    except ImportError:
        log.warning("langfuse package not installed, tracing disabled")
        return None
    except Exception as exc:
        log.warning("Failed to initialize Langfuse: %s", exc)
        return None
