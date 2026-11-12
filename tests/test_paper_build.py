"""The paper builder prints numbers only from results values and tells osaudit how each one was derived."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import build  # noqa: E402

VALUES = {"v": {"share": 70.512, "neg": -19.44, "count": 42, "p_small": 3e-5, "p": 0.0061, "one": 1.0,
                "whole": 55.0, "frac": 54.44, "list": [3, 4], "odd key": {"x": 2.0}}}


def fmt(text: str) -> tuple[str, build.Ledger]:
    ledger = build.Ledger()
    return build.Formatter(VALUES, ledger).text(text), ledger


def test_formats_and_derive_entries():
    out, led = fmt("{{pct1: v.share}} {{sgn1: v.neg}} {{dec1: v.neg}} {{int: v.count}} {{num1: v.frac}} {{num0: v.whole}}")
    assert out == "70.5% −19.4 −19.4 42 54.4 55"
    derive = dict(led.derive)
    assert derive[("70.5%", "v.share")] is None
    assert ("19.4", "0 - (v.neg)") in derive               # osaudit reads printed numbers without their sign
    assert ("42", "v.count") in derive


def test_p_values_floor_and_one():
    out, led = fmt("{{p: v.p_small}}; {{p: v.p}}; {{p: v.one}}")
    assert out == "< 0.001; 0.006; 1"
    assert ("0.001", "reporting floor for p-values (smaller values are printed as < 0.001)", "") in led.allow


def test_expressions_indices_and_quoted_keys():
    out, _ = fmt('{{int: v.list[1]}} {{num1: v["odd key"].x * 2}} {{num1: v.whole - v.frac}}')
    assert out == "4 4.0 0.6"


def test_rejects_rounding_a_non_integer_to_a_plain_integer():
    with pytest.raises(ValueError, match="integral"):
        fmt("{{num0: v.frac}}")


def test_rejects_negative_with_unsigned_format():
    with pytest.raises(ValueError, match="negative"):
        fmt("{{num1: v.neg}}")


def test_settings_and_cited_numbers_become_allow_entries():
    out, led = fmt("{{set: 64 | frame size}} {{ext: 42 | [22] | their drop}} {{fact: 3,000 | log}}")
    assert out == "64 42 3,000"
    assert ("64", "setting of the study: frame size", "") in led.allow
    assert ("42", "reported by the cited work: their drop", "[22]") in led.allow
    assert ("3,000", "log", "") in led.allow


def test_missing_value_fails_unless_draft():
    with pytest.raises(KeyError):
        fmt("{{int: nofile.x}}")
    f = build.Formatter(VALUES, build.Ledger(), draft=True)
    assert f.text("{{int: nofile.x}}") == "[pending: nofile.x]"
    assert f.pending == ["nofile.x"]


def test_rounding_is_half_up():
    assert build._dec(0.125, 2) == "0.13"
    assert build._dec(16.65, 1) == "16.7"
    assert build._dec(2.0, 1) == "2.0"
