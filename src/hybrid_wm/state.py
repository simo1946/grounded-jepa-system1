"""Streaming state helpers.

Recurrent state is a nested ``dict`` of tensors. Keeping it explicit (instead
of hidden module buffers) makes the model functional: the same weights can
serve many independent streams, and planners can fork a state cheaply.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import torch

State = Dict[str, Any]


def map_state(state: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply ``fn`` to every tensor in a nested state, preserving structure."""
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, dict):
        return {k: map_state(v, fn) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(map_state(v, fn) for v in state)
    return state  # ints, floats, None


def clone_state(state: Any) -> Any:
    return map_state(state, lambda t: t.clone())


def repeat_state(state: Any, n: int) -> Any:
    """Repeat each batch element ``n`` times (batch must be dim 0):
    [B, ...] -> [B*n, ...] with element order b0,b0,..,b1,b1,..."""
    return map_state(state, lambda t: t.repeat_interleave(n, dim=0))


def detach_state(state: Any) -> Any:
    """Truncate backprop-through-time between training segments."""
    return map_state(state, lambda t: t.detach())
