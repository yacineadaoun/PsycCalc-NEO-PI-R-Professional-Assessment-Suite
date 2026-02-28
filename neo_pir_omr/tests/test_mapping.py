from __future__ import annotations

from neo_pir_omr.core.engine import item_to_facette


def test_item_mapping_complete():
    # 240 items should map to a facette
    missing = [i for i in range(1, 241) if i not in item_to_facette]
    assert missing == []
