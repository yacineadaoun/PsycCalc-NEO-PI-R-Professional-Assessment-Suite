from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from neo_pir_omr.core.engine import (
    OMRScanner,
    OMRConfig,
    load_scoring_key_from_bytes,
    binarize_inv,
    find_table_bbox_soft,
    split_grid_micro_adjust,
    split_grid_uniform,
    item_id_from_rc,
)
from neo_pir_omr.core.security import SecurityPolicy, validate_file_bytes


st.set_page_config(page_title="NEO PI‑R OMR Scanner", page_icon="🧾", layout="wide")


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bytes_download_button(label: str, data: bytes, file_name: str, mime: str):
    st.download_button(label, data=data, file_name=file_name, mime=mime)


st.title("🧾 NEO PI‑R OMR Scanner — Professional Edition")
st.caption("Scan robuste + validation humaine assistée (stylo bleu/noir).")

policy = SecurityPolicy(max_upload_mb=15)

with st.sidebar:
    st.header("Paramètres")
    mark_threshold = st.slider("Seuil marque (mark_threshold)", 0.5, 6.0, 1.7, 0.1)
    ambiguity_gap = st.slider("Seuil ambiguïté (gap)", 0.1, 6.0, 0.9, 0.1)

    detect_blue = st.checkbox("Détecter encre bleue", value=True)
    detect_black = st.checkbox("Détecter encre noire", value=True)
    black_dark_thresh = st.slider("Seuil noir (dark_thresh)", 60, 180, 110, 1)
    black_baseline_quantile = st.slider("Quantile baseline noir", 0.0, 50.0, 15.0, 1.0)

    st.divider()
    st.subheader("Clé de scoring")
    key_file = st.file_uploader("Importer scoring_key.csv", type=["csv"])


colA, colB = st.columns([1, 1])
with colA:
    img_file = st.file_uploader("Importer une photo/scanner de la feuille", type=["jpg", "jpeg", "png", "webp"])

with colB:
    st.info(
        "Conseils: photo nette, feuille complète, lumière uniforme, bordures visibles.\n\n"
        "Le système marque en **orange** les cellules ambiguës et en **rouge** les vides."
    )


if not img_file:
    st.stop()

validate_file_bytes(img_file.name, img_file.size, policy)

pil_img = Image.open(io.BytesIO(img_file.getvalue()))

# scoring key
if key_file is not None:
    scoring_key = load_scoring_key_from_bytes(key_file.getvalue())
else:
    default_key_path = Path(__file__).resolve().parents[1] / "data" / "scoring_key.csv"
    scoring_key = load_scoring_key_from_bytes(default_key_path.read_bytes())

cfg = OMRConfig(
    mark_threshold=float(mark_threshold),
    ambiguity_gap=float(ambiguity_gap),
    detect_blue=bool(detect_blue),
    detect_black=bool(detect_black),
    black_dark_thresh=int(black_dark_thresh),
    black_baseline_quantile=float(black_baseline_quantile),
)

scanner = OMRScanner(cfg=cfg)

with st.spinner("Scan en cours..."):
    result = scanner.scan_pil(pil_img, scoring_key)

st.success(f"Scan terminé ✅  (scan_id: {result.scan_id})")

# Visuals
v1, v2 = st.columns([1, 1])
with v1:
    st.subheader("Overlay (détection)")
    st.image(bgr_to_rgb(result.overlay_bgr), use_container_width=True)

with v2:
    st.subheader("Masque encre (debug)")
    st.image(result.debug_mask, use_container_width=True)

# Summary
st.subheader("Résumé")
met = result.diagnostics["stats"]
proto = result.protocol
st.write(
    {
        "valid_protocol": bool(proto.get("valid")),
        "n_blank": int(proto.get("n_blank", 0)),
        "ambiguous": int(met.get("ambiguous", 0)),
        "low_conf": int(met.get("low_conf", 0)),
        "imputed": int(proto.get("imputed", 0)),
        "reasons": proto.get("reasons", []),
    }
)

# Human validation UI
st.subheader("Validation humaine assistée")
st.caption("Corrige les items *blank/ambiguous/low confidence* puis recalcul des scores.")

thr_inv = binarize_inv(result.warped_bgr)
table_bbox = find_table_bbox_soft(thr_inv)

# Map item -> cell bbox
cell_map = {}
iterator = split_grid_micro_adjust(thr_inv, table_bbox, rows=cfg.rows, cols=cfg.cols) if cfg.use_micro_adjust else split_grid_uniform(table_bbox, rows=cfg.rows, cols=cfg.cols)
for r, c, cell in iterator:
    cell_map[item_id_from_rc(r, c)] = cell

flagged = []
for item_id in range(1, 241):
    m = result.meta.get(item_id, {})
    if m.get("blank") or m.get("ambiguous") or float(m.get("confidence", 1.0)) < 0.55:
        flagged.append(item_id)

st.write({"flagged_items": len(flagged)})

corrections = {}
if flagged:
    st.warning("Des items nécessitent une revue. Tu peux corriger seulement ceux que tu veux.")

    # show a paginated list
    page_size = 12
    page = st.number_input("Page", min_value=1, max_value=max(1, (len(flagged) + page_size - 1) // page_size), value=1, step=1)
    start = (page - 1) * page_size
    end = min(len(flagged), start + page_size)

    for item_id in flagged[start:end]:
        cell = cell_map.get(item_id)
        if not cell:
            continue
        x1, y1, x2, y2 = cell
        crop = result.warped_bgr[y1:y2, x1:x2]
        m = result.meta.get(item_id, {})
        current = result.responses_raw.get(item_id, -1)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(bgr_to_rgb(crop), caption=f"Item {item_id}", use_container_width=True)
        with c2:
            st.write(
                {
                    "raw": int(current),
                    "blank": bool(m.get("blank")),
                    "ambiguous": bool(m.get("ambiguous")),
                    "confidence": float(m.get("confidence", 0.0)),
                    "fills": [round(float(x), 2) for x in m.get("fills_total", [])],
                }
            )
            choice = st.selectbox(
                f"Correction Item {item_id}",
                options=[-1, 0, 1, 2, 3, 4],
                index=[-1, 0, 1, 2, 3, 4].index(current if current in [-1,0,1,2,3,4] else -1),
                help="-1 = vide ; 0..4 = option cochée",
                key=f"corr_{item_id}",
            )
            if choice != current:
                corrections[item_id] = int(choice)

# Apply corrections and recompute
final_resp = dict(result.responses_final)
if corrections:
    for k, v in corrections.items():
        final_resp[k] = v

    facette_scores, domain_scores = result.facette_scores, result.domain_scores
    # recompute with same scoring_key
    from neo_pir_omr.core.engine import compute_scores, apply_protocol_rules

    final_after_proto, proto2 = apply_protocol_rules(cfg, final_resp)
    facette_scores, domain_scores = compute_scores(final_after_proto, scoring_key)

    st.success("Corrections appliquées et scores recalculés ✅")
    st.write({"protocol": proto2, "domain_scores": domain_scores})

    # Export corrected
    # responses.csv
    out = io.StringIO()
    out.write("item,choice\n")
    for item_id in range(1, 241):
        out.write(f"{item_id},{final_after_proto.get(item_id, -1)}\n")
    bytes_download_button("Télécharger responses_corrigées.csv", out.getvalue().encode("utf-8"), "responses_corrigees.csv", "text/csv")

    bytes_download_button(
        "Télécharger scores_corrigés.json",
        json.dumps({"protocol": proto2, "facette_scores": facette_scores, "domain_scores": domain_scores}, ensure_ascii=False, indent=2).encode("utf-8"),
        "scores_corriges.json",
        "application/json",
    )

# Always offer raw outputs
st.subheader("Exports")
raw_out = io.StringIO()
raw_out.write("item,choice\n")
for item_id in range(1, 241):
    raw_out.write(f"{item_id},{result.responses_final.get(item_id, -1)}\n")
bytes_download_button("Télécharger responses.csv", raw_out.getvalue().encode("utf-8"), "responses.csv", "text/csv")

bytes_download_button(
    "Télécharger scores.json",
    json.dumps({"protocol": result.protocol, "facette_scores": result.facette_scores, "domain_scores": result.domain_scores}, ensure_ascii=False, indent=2).encode("utf-8"),
    "scores.json",
    "application/json",
)

bytes_download_button(
    "Télécharger audit.json",
    json.dumps({"diagnostics": result.diagnostics, "meta": result.meta}, ensure_ascii=False, indent=2).encode("utf-8"),
    "audit.json",
    "application/json",
)
