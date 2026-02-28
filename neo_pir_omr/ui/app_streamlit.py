from __future__ import annotations

import io
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from neo_pir_omr.core.engine import (
    OMRScanner,
    OMRConfig,
    load_scoring_key_from_bytes,
    apply_protocol_rules,
    compute_scores,
)
from neo_pir_omr.core.security import SecurityPolicy, validate_file_bytes


# =============================================================================
# App config
# =============================================================================
st.set_page_config(
    page_title="NEO PI-R — OMR & Scoring",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Design system (CSS)
# =============================================================================
st.markdown(
    """
<style>
  :root {
    --bg: rgba(255,255,255,0.03);
    --bd: rgba(255,255,255,0.08);
    --tx: rgba(255,255,255,0.92);
    --muted: rgba(255,255,255,0.70);
    --ok: rgba(46, 204, 113, 0.18);
    --warn: rgba(241, 196, 15, 0.16);
    --bad: rgba(231, 76, 60, 0.18);
  }
  .block-container { padding-top: 1.1rem; max-width: 1300px; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .muted { color: var(--muted); }
  .card {
    background: var(--bg);
    border: 1px solid var(--bd);
    border-radius: 18px;
    padding: 14px 16px;
  }
  .card-tight { padding: 10px 12px; }
  .pill {
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--bd);
    background: rgba(255,255,255,0.02);
    font-size: 0.90rem;
    color: var(--muted);
  }
  .kpi {
    background: var(--bg);
    border: 1px solid var(--bd);
    border-radius: 18px;
    padding: 14px 16px;
  }
  .kpi .label { color: var(--muted); font-size: 0.92rem; }
  .kpi .value { font-size: 1.7rem; font-weight: 700; color: var(--tx); margin-top: 4px; }
  .kpi.ok { background: var(--ok); }
  .kpi.warn { background: var(--warn); }
  .kpi.bad { background: var(--bad); }
  .section-title {
    display:flex; align-items:end; justify-content:space-between;
    margin: 6px 0 10px 0;
  }
  .section-title .right { color: var(--muted); font-size: 0.92rem; }
  div[data-testid="stDownloadButton"] button { width: 100%; border-radius: 14px; }
  div[data-testid="stButton"] button { border-radius: 14px; padding: 0.65rem 0.9rem; }
  div[data-testid="stFileUploader"] { border-radius: 18px; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Helpers
# =============================================================================
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _download(label: str, data: bytes, name: str, mime: str):
    st.download_button(label, data=data, file_name=name, mime=mime, use_container_width=True)


def _load_norms_df() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[1] / "data" / "norms.csv"
    return pd.read_csv(p)


def _pick_norms(norms: pd.DataFrame, scale_type: str, scale: str, sex: str, age: int) -> Tuple[float, float]:
    sub = norms[
        (norms["scale_type"] == scale_type)
        & (norms["scale"] == scale)
        & (norms["sex"] == sex)
        & (norms["age_min"] <= age)
        & (norms["age_max"] >= age)
    ]
    if sub.empty:
        raise ValueError("Aucune norme correspondante (sexe/âge) n’a été trouvée dans norms.csv.")
    row = sub.iloc[0]
    return float(row["mean"]), float(row["sd"])


def _z_t(raw: float, mean: float, sd: float) -> Dict[str, float]:
    if sd <= 0:
        raise ValueError("Écart-type (SD) invalide dans norms.csv.")
    z = (raw - mean) / sd
    t = 50.0 + 10.0 * z
    return {"z": float(z), "t": float(t)}


def _plot_line(domain_t: Dict[str, float]):
    labels = ["N", "E", "O", "A", "C"]
    y = [domain_t.get(k, np.nan) for k in labels]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(8.6, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xticks(x, labels)
    ax.set_ylim(20, 80)
    ax.set_ylabel("Score T")
    ax.set_title("Profil global (Scores T) — N/E/O/A/C")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def _plot_radar(domain_t: Dict[str, float]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_t.get(k, np.nan) for k in labels]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(20, 80)
    ax.set_title("Radar (Scores T)", pad=18)
    st.pyplot(fig, clear_figure=True)


def _kpi(label: str, value: int, tone: str = "base"):
    cls = "kpi"
    if tone in ("ok", "warn", "bad"):
        cls += f" {tone}"
    st.markdown(
        f"""
<div class="{cls}">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# Header
# =============================================================================
st.markdown(
    """
<div class="section-title">
  <div>
    <h1 style="margin:0">NEO PI-R — Scanner OMR & Cotation</h1>
    <div class="muted">Workflow scientifique : scan → contrôle qualité → scores bruts → scores normés → exports</div>
  </div>
  <div class="right">
    <span class="pill">🧪 Version “scientifique”</span>
    <span class="pill">🛡️ Validation fichier</span>
    <span class="pill">📦 Exports JSON/CSV</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar (settings)
# =============================================================================
policy = SecurityPolicy(max_upload_mb=15)

with st.sidebar:
    st.header("Paramètres")
    st.caption("Réglages avancés. Laissez par défaut si votre scan est propre.")

    mark_threshold = st.slider("Seuil de marque (mark_threshold)", 0.5, 6.0, 1.7, 0.1)
    ambiguity_gap = st.slider("Seuil d’ambiguïté (ambiguity_gap)", 0.1, 6.0, 0.9, 0.1)

    st.divider()
    st.subheader("Encre")
    detect_blue = st.checkbox("Détecter encre bleue", value=True)
    detect_black = st.checkbox("Détecter encre noire", value=True)
    black_dark_thresh = st.slider("Seuil noir (black_dark_thresh)", 60, 180, 110, 1)
    black_baseline_quantile = st.slider("Quantile baseline noir", 0.0, 50.0, 15.0, 1.0)

    st.divider()
    st.subheader("Normes (Z/T)")
    sex = st.selectbox("Sexe", options=["M", "F"], index=0)
    age = st.number_input("Âge", min_value=10, max_value=90, value=25, step=1)

    st.divider()
    st.subheader("Fichiers")
    key_file = st.file_uploader("Clé de cotation (scoring_key.csv)", type=["csv"])
    norms_file = st.file_uploader("Normes (norms.csv) — optionnel", type=["csv"])

    st.divider()
    st.markdown(
        """
<div class="card card-tight">
<b>Conseils scan</b><br/>
<span class="muted">Feuille complète • bonne lumière • pas de reflets • photo nette</span>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# Inputs (step 1)
# =============================================================================
st.markdown("### 1) Import")
cL, cR = st.columns([1.2, 0.8], vertical_alignment="top")
with cL:
    img_file = st.file_uploader("Image/scan de la feuille", type=["jpg", "jpeg", "png", "webp"])
with cR:
    st.markdown(
        """
<div class="card">
<b>Qualité de lecture</b><br/>
<ul class="muted" style="margin:8px 0 0 18px; line-height: 1.55;">
  <li>Photo nette (sans flou)</li>
  <li>Feuille entière dans le cadre</li>
  <li>Éclairage uniforme</li>
  <li>Éviter reflets / ombres fortes</li>
  <li>Stylo bleu/noir bien visible</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

if not img_file:
    st.stop()

validate_file_bytes(img_file.name, img_file.size, policy)
img_bytes = img_file.getvalue()
img_hash = _hash_bytes(img_bytes)
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

# scoring key
if key_file is not None:
    scoring_key = load_scoring_key_from_bytes(key_file.getvalue())
else:
    default_key_path = Path(__file__).resolve().parents[1] / "data" / "scoring_key.csv"
    scoring_key = load_scoring_key_from_bytes(default_key_path.read_bytes())

# norms
if norms_file is not None:
    norms_df = pd.read_csv(io.BytesIO(norms_file.getvalue()))
else:
    norms_df = _load_norms_df()

cfg = OMRConfig(
    mark_threshold=float(mark_threshold),
    ambiguity_gap=float(ambiguity_gap),
    detect_blue=bool(detect_blue),
    detect_black=bool(detect_black),
    black_dark_thresh=int(black_dark_thresh),
    black_baseline_quantile=float(black_baseline_quantile),
)

scanner = OMRScanner(cfg=cfg)

# =============================================================================
# Run scan (stable cache)
# =============================================================================
if "scan_cache" not in st.session_state:
    st.session_state.scan_cache = {}

cfg_sig = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False).encode("utf-8")
cache_key = f"{img_hash}:{_hash_bytes(cfg_sig)}"

st.markdown("### 2) Scan & contrôle qualité")
btn_cols = st.columns([0.55, 0.45])
with btn_cols[0]:
    run_scan = st.button("🚀 Lancer le scan & la cotation", use_container_width=True)
with btn_cols[1]:
    st.markdown(
        '<div class="pill">Astuce : les changements de sliders ne relancent pas le scan tant que vous ne cliquez pas.</div>',
        unsafe_allow_html=True,
    )

if run_scan:
    with st.spinner("Analyse en cours…"):
        try:
            result = scanner.scan_pil(pil_img, scoring_key)
        except Exception as e:
            st.error("Échec du scan. Détail ci-dessous :")
            st.exception(e)
            st.stop()
    st.session_state.scan_cache[cache_key] = result

result = st.session_state.scan_cache.get(cache_key)
if result is None:
    st.info("Cliquez sur **Lancer le scan & la cotation** pour démarrer.")
    st.stop()

# =============================================================================
# Dashboard (KPIs)
# =============================================================================
stats = result.diagnostics.get("stats", {})
proto = result.protocol or {}

k1, k2, k3, k4 = st.columns(4)
with k1:
    _kpi("Réponses vides", int(proto.get("n_blank", 0)), tone="warn" if int(proto.get("n_blank", 0)) > 0 else "ok")
with k2:
    _kpi("Ambiguës", int(stats.get("ambiguous", 0)), tone="warn" if int(stats.get("ambiguous", 0)) > 0 else "ok")
with k3:
    _kpi("Faible confiance", int(stats.get("low_conf", 0)), tone="warn" if int(stats.get("low_conf", 0)) > 0 else "ok")
with k4:
    _kpi("Imputées", int(proto.get("imputed", 0)), tone="bad" if int(proto.get("imputed", 0)) > 0 else "ok")

st.success(f"Scan terminé ✅  — scan_id : {result.scan_id}")

# =============================================================================
# Tabs (pro workflow)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Résultats", "🧠 Normes (Z/T)", "🔍 Qualité & Debug", "⬇️ Exports"])

# --- TAB 1: Résultats
with tab1:
    st.markdown("#### Scores bruts")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Domaines (N/E/O/A/C)**")
        st.json(result.domain_scores)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Facettes**")
        st.json(result.facette_scores)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Relecture humaine (optionnelle)")
    st.caption("Corriger uniquement les items vides/ambiguës/faible confiance, puis recalculer.")

    flagged = []
    for item_id in range(1, 241):
        md = result.meta.get(item_id, {})
        if md.get("blank") or md.get("ambiguous") or float(md.get("confidence", 1.0)) < 0.55:
            flagged.append(item_id)

    with st.expander(f"🧑‍🔬 Ouvrir la relecture — items à vérifier ({len(flagged)})", expanded=False):
        if not flagged:
            st.success("Aucun item à relire ✅")
        else:
            page_size = 12
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=max(1, (len(flagged) + page_size - 1) // page_size),
                value=1,
                step=1,
            )
            start = (page - 1) * page_size
            end = min(len(flagged), start + page_size)

            corrections: Dict[int, int] = {}
            for item_id in flagged[start:end]:
                current = int(result.responses_raw.get(item_id, -1))
                md = result.meta.get(item_id, {})
                r1, r2 = st.columns([0.55, 1.45])
                with r1:
                    st.markdown(f"**Item {item_id}**")
                    st.caption(f"Confiance : {float(md.get('confidence', 0.0)):.2f}")
                with r2:
                    choice = st.selectbox(
                        f"Correction item {item_id}",
                        options=[-1, 0, 1, 2, 3, 4],
                        index=[-1, 0, 1, 2, 3, 4].index(current if current in [-1, 0, 1, 2, 3, 4] else -1),
                        help="-1 = vide ; 0..4 = option sélectionnée",
                        key=f"corr_{item_id}",
                    )
                    if int(choice) != current:
                        corrections[item_id] = int(choice)

            if st.button("✅ Appliquer les corrections & recalculer", use_container_width=True):
                try:
                    final_resp = dict(result.responses_final)
                    final_resp.update(corrections)

                    final_after_proto, proto2 = apply_protocol_rules(cfg, final_resp)
                    facette_scores2, domain_scores2 = compute_scores(final_after_proto, scoring_key)

                    st.success("Recalcul terminé ✅")
                    st.json({"protocol": proto2, "domain_scores": domain_scores2, "facette_scores": facette_scores2})
                except Exception as e:
                    st.error("Échec du recalcul. Détail :")
                    st.exception(e)

# --- TAB 2: Normes (Z/T) + Graphiques
with tab2:
    st.markdown("#### Scores normés (Z & T)")
    st.caption("T = 50 + 10×Z — dépend de `norms.csv` ou du fichier importé.")

    domain_t: Dict[str, float] = {}
    domain_norm_detail: Dict[str, Any] = {}

    try:
        for d, raw in result.domain_scores.items():
            mean, sd = _pick_norms(norms_df, scale_type="domain", scale=d, sex=sex, age=int(age))
            res = _z_t(float(raw), mean, sd)
            domain_t[d] = res["t"]
            domain_norm_detail[d] = {"raw": int(raw), "mean": mean, "sd": sd, **res}
    except Exception as e:
        st.warning(f"Impossible de calculer les scores T : {e}")
        domain_norm_detail = {}

    if domain_norm_detail:
        df = pd.DataFrame(domain_norm_detail).T
        st.dataframe(df, use_container_width=True)

        g1, g2 = st.columns([1.2, 0.8])
        with g1:
            _plot_line(domain_t)
        with g2:
            _plot_radar(domain_t)
    else:
        st.info("Aucun score T calculé. Vérifiez votre fichier norms.csv (colonnes/valeurs).")

# --- TAB 3: Qualité & Debug (overlay/mask + JSON)
with tab3:
    st.markdown("#### Contrôle qualité (visuel)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Overlay (détection)**")
        st.image(result.overlay_bgr[:, :, ::-1], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Masque encre (debug)**")
        st.image(result.debug_mask, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Détails techniques")
    vals = list(result.responses_final.values())
    st.json(
        {
            "valid_protocol": bool(proto.get("valid", True)),
            "reasons": proto.get("reasons", []),
            "n_items": len(vals),
            "n_-1_blank": int(sum(1 for v in vals if v == -1)),
            "sample_1_20": {k: int(result.responses_final.get(k, -1)) for k in range(1, 21)},
            "thresholds": result.diagnostics.get("thresholds", {}),
            "table_bbox": result.diagnostics.get("table_bbox", {}),
        }
    )

# --- TAB 4: Exports (clean, stable)
with tab4:
    st.markdown("#### Exports")
    resp_csv = io.StringIO()
    resp_csv.write("item,choice\n")
    for item_id in range(1, 241):
        value = int(result.responses_final.get(item_id, -1))
        resp_csv.write(f"{item_id},{value}\n")

    c1, c2, c3 = st.columns(3)
    with c1:
        _download("Télécharger responses.csv", resp_csv.getvalue().encode("utf-8"), "responses.csv", "text/csv")
    with c2:
        _download(
            "Télécharger scores.json",
            json.dumps(
                {
                    "scan_id": result.scan_id,
                    "protocol": result.protocol,
                    "domain_scores_raw": result.domain_scores,
                    "facette_scores_raw": result.facette_scores,
                    "cfg": asdict(cfg),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            "scores.json",
            "application/json",
        )
    with c3:
        _download(
            "Télécharger audit.json",
            json.dumps(
                {"diagnostics": result.diagnostics, "meta": result.meta},
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            "audit.json",
            "application/json",
        )
