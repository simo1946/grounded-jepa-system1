"""Keep an instrument's stdout for its JSON result only.

osaudit run parses an instrument's stdout as one JSON object, but some libraries log to stdout (loguru sinks
bound to sys.stdout at import time). ``quiet_stdout`` sends file descriptor 1 to stderr while the work runs,
and ``emit`` writes the result to the real stdout afterwards.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys


@contextlib.contextmanager
def quiet_stdout():
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=1) + "\n")
    sys.stdout.flush()
