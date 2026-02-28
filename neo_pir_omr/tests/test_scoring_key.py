from __future__ import annotations

from pathlib import Path

from neo_pir_omr.core.engine import load_scoring_key_from_bytes


def test_scoring_key_has_240_items():
    p = Path(__file__).resolve().parents[1] / "data" / "scoring_key.csv"
    key = load_scoring_key_from_bytes(p.read_bytes())
    assert len(key) == 240
    assert 1 in key and 240 in key
