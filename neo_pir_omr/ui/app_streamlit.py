from __future__ import annotations

import io
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from neo_pir_omr.core.engine import (
    OMRScanner,
    OMRConfig,
    load_scoring_key_from_bytes,
    apply_protocol_rules,
    compute_scores,
)
from neo_pir_omr.core.security import SecurityPolicy, validate_file_bytes


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="NEO PI-R — OMR & Cotation (SaaS)",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Premium UI (CSS)
# =============================================================================
st.markdown(
    """
<style>
  :root{
    --bg: rgba(255,255,255,0.03);
    --bd: rgba(255,255,255,0.10);
    --tx: rgba(255,255,255,0.93);
    --muted: rgba(255,255,255,0.70);
    --ok: rgba(46, 204, 113, 0.18);
    --warn: rgba(241, 196, 15, 0.16);
    --bad: rgba(231, 76, 60, 0.18);
  }
  .block-container{ padding-top: 1.1rem; max-width: 1320px; }
  h1,h2,h3 { letter-spacing:-0.02em; }
  .muted{ color: var(--muted); }
  .card{
    background: var(--bg);
    border: 1px solid var(--bd);
    border-radius: 18px;
    padding: 14px 16px;
  }
  .card-tight{ padding: 10px 12px; }
  .pill{
    display:inline-flex; align-items:center; gap:8px;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--bd);
    background: rgba(255,255,255,0.02);
    font-size: 0.90rem;
    color: var(--muted);
  }
  .kpi{
    background: var(--bg);
    border: 1px solid var(--bd);
    border-radius: 18px;
    padding: 14px 16px;
  }
  .kpi .label{ color: var(--muted); font-size: .92rem; }
  .kpi .value{ font-size: 1.7rem; font-weight: 750; color: var(--tx); margin-top: 4px; }
  .kpi.ok{ background: var(--ok); }
  .kpi.warn{ background: var(--warn); }
  .kpi.bad{ background: var(--bad); }
  .section-title{
    display:flex; align-items:end; justify-content:space-between;
    margin: 6px 0 10px 0;
  }
  .section-title .right{ color: var(--muted); font-size: 0.92rem; }
  div[data-testid="stDownloadButton"] button{ width:100%; border-radius: 14px; }
  div[data-testid="stButton"] button{ border-radius: 14px; padding: .65rem .9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Constants
# =============================================================================
CHOICES = ["FD", "D", "N", "A", "FA"]  # Fortement en désaccord ... Fortement d'accord
DOMAINS = ["N", "E", "O", "A", "C"]

# =============================================================================
# Helpers
# =============================================================================
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _download(label: str, data: bytes, name: str, mime: str):
    st.download_button(label, data=data, file_name=name, mime=mime, use_container_width=True)


def _load_norms_df_from_disk() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[1] / "data" / "norms.csv"
    return pd.read_csv(p)


def interpret_t(t: float) -> str:
    """Interprétation simple et standard."""
    if not np.isfinite(t):
        return "N/A"
    if t < 45:
        return "Faible"
    if t > 55:
        return "Élevé"
    return "Moyen"


def validate_norms_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    required = {"scale_type", "scale", "sex", "age_min", "age_max", "mean", "sd"}
    missing = sorted(list(required - set(df.columns)))
    errors: list[str] = []
    if missing:
        errors.append(f"Colonnes manquantes: {missing}")
        return False, errors

    for col in ["age_min", "age_max", "mean", "sd"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Colonne '{col}' doit être numérique.")
    if (df["age_min"] > df["age_max"]).any():
        errors.append("Certaines lignes ont age_min > age_max.")
    if (df["sd"] <= 0).any():
        errors.append("Certaines lignes ont sd <= 0 (invalides).")
    if not set(df["sex"].astype(str).unique()).issubset({"M", "F"}):
        errors.append("Colonne 'sex' doit contenir uniquement 'M' ou 'F'.")
    if "domain" not in set(df["scale_type"].astype(str).unique()):
        errors.append("Colonne 'scale_type' doit contenir au moins la valeur 'domain'.")

    return (len(errors) == 0), errors


def _pick_norms(norms: pd.DataFrame, scale_type: str, scale: str, sex: str, age: int) -> Tuple[float, float]:
    sub = norms[
        (norms["scale_type"].astype(str) == scale_type)
        & (norms["scale"].astype(str) == str(scale))
        & (norms["sex"].astype(str) == str(sex))
        & (norms["age_min"] <= age)
        & (norms["age_max"] >= age)
    ]
    if sub.empty:
        raise ValueError("Aucune norme correspondante (sexe/âge) n’a été trouvée.")
    row = sub.iloc[0]
    return float(row["mean"]), float(row["sd"])


def _z_t(raw: float, mean: float, sd: float) -> Dict[str, float]:
    if sd <= 0:
        raise ValueError("Écart-type (SD) invalide dans norms.csv.")
    z = (raw - mean) / sd
    t = 50.0 + 10.0 * z
    return {"z": float(z), "t": float(t)}


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_report_pdf(
    *,
    logo_bytes: Optional[bytes],
    subject: dict,
    scan_id: str,
    domain_scores_raw: dict,
    facet_scores_raw: dict,
    domain_norm_detail: dict,  # contient raw/mean/sd/z/t
    fig_line_png: Optional[bytes],
    fig_radar_png: Optional[bytes],
) -> bytes:
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    W, H = A4

    def draw_logo():
        if not logo_bytes:
            return
        try:
            img = ImageReader(io.BytesIO(logo_bytes))
            c.drawImage(img, W - 6.0 * cm, H - 3.2 * cm, width=4.5 * cm, height=1.6 * cm, mask="auto")
        except Exception:
            pass

    def header(title: str):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, H - 2 * cm, title)
        c.setFont("Helvetica", 9)
        c.setFillGray(0.4)
        c.drawString(2 * cm, H - 2.6 * cm, f"Scan ID: {scan_id}")
        c.setFillGray(0)

    # Page 1 — Couverture
    draw_logo()
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2 * cm, H - 5 * cm, "Rapport NEO PI-R — OMR & Cotation")
    c.setFont("Helvetica", 11)
    c.setFillGray(0.25)
    c.drawString(2 * cm, H - 6 * cm, "Rapport généré automatiquement (FR)")
    c.setFillGray(0)

    y = H - 8 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Informations sujet")
    c.setFont("Helvetica", 10)
    y -= 0.8 * cm
    for k, v in subject.items():
        c.drawString(2 * cm, y, f"{k}: {v}")
        y -= 0.55 * cm
    c.showPage()

    # Page 2 — Domaines + interprétation
    header("Résumé des résultats — Domaines")
    draw_logo()

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, H - 4.0 * cm, "Domaines (brut + T-score + interprétation)")
    x0, y0 = 2 * cm, H - 5.0 * cm
    colw = [3.0 * cm, 3.0 * cm, 3.0 * cm, 7.0 * cm]
    rowh = 0.65 * cm

    headers = ["Domaine", "Brut", "T-score", "Interprétation"]
    c.setFont("Helvetica-Bold", 9)
    xx = x0
    for i, htxt in enumerate(headers):
        c.rect(xx, y0, colw[i], rowh, stroke=1, fill=0)
        c.drawString(xx + 0.2 * cm, y0 + 0.2 * cm, htxt)
        xx += colw[i]

    c.setFont("Helvetica", 9)
    y = y0 - rowh
    for dom in DOMAINS:
        raw = int(domain_scores_raw.get(dom, 0))
        t = float(domain_norm_detail.get(dom, {}).get("t", float("nan")))
        interp = interpret_t(t)
        values = [dom, str(raw), f"{t:.1f}" if np.isfinite(t) else "N/A", interp]
        xx = x0
        for i, v in enumerate(values):
            c.rect(xx, y, colw[i], rowh, stroke=1, fill=0)
            c.drawString(xx + 0.2 * cm, y + 0.2 * cm, v)
            xx += colw[i]
        y -= rowh

    c.setFillGray(0.35)
    c.setFont("Helvetica", 8)
    c.drawString(2 * cm, 2.0 * cm, "Règle: T<45 Faible | 45–55 Moyen | >55 Élevé (modifiable).")
    c.setFillGray(0)
    c.showPage()

    # Page 3 — Graphiques
    header("Visualisations (Scores T)")
    draw_logo()

    y_img = H - 4.0 * cm
    if fig_line_png:
        img = ImageReader(io.BytesIO(fig_line_png))
        c.drawImage(img, 2 * cm, y_img - 7.0 * cm, width=16 * cm, height=6.5 * cm, mask="auto")
        y_img -= 7.6 * cm
    if fig_radar_png:
        img = ImageReader(io.BytesIO(fig_radar_png))
        c.drawImage(img, 5.0 * cm, y_img - 10.0 * cm, width=10.5 * cm, height=10.5 * cm, mask="auto")
    c.showPage()

    # Page 4 — Facettes (brut)
    header("Facettes (scores bruts)")
    draw_logo()

    df_f = pd.DataFrame({"Facette": list(facet_scores_raw.keys()), "Score brut": list(facet_scores_raw.values())})
    df_f = df_f.sort_values("Facette")

    x, y = 2 * cm, H - 4.0 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Liste des facettes (brut)")
    y -= 0.8 * cm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Facette")
    c.drawString(x + 12.5 * cm, y, "Score brut")
    y -= 0.35 * cm
    c.line(x, y, W - 2 * cm, y)
    y -= 0.55 * cm
    c.setFont("Helvetica", 9)

    for _, row in df_f.iterrows():
        if y < 2.2 * cm:
            c.showPage()
            header("Facettes (suite)")
            y = H - 4.0 * cm
            c.setFont("Helvetica", 9)

        c.drawString(x, y, str(row["Facette"])[:60])
        c.drawRightString(W - 2 * cm, y, str(int(row["Score brut"])))
        y -= 0.45 * cm

    c.showPage()
    c.save()
    out.seek(0)
    return out.getvalue()


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
# Session state init
# =============================================================================
if "norms_df" not in st.session_state:
    st.session_state.norms_df = None
if "norms_version" not in st.session_state:
    st.session_state.norms_version = None
if "scan_cache" not in st.session_state:
    st.session_state.scan_cache = {}


# =============================================================================
# Header
# =============================================================================
st.markdown(
    """
<div class="section-title">
  <div>
    <h1 style="margin:0">NEO PI-R — Scanner OMR & Cotation</h1>
    <div class="muted">SaaS (FR) : import → scan → contrôle qualité → scores → interprétation → PDF → exports</div>
  </div>
  <div class="right">
    <span class="pill">🧪 Mode scientifique</span>
    <span class="pill">📄 PDF multi-pages</span>
    <span class="pill">🛠️ Admin (normes)</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar navigation + settings
# =============================================================================
policy = SecurityPolicy(max_upload_mb=15)

with st.sidebar:
    st.header("Navigation")
    nav = st.radio(" ", ["📋 Analyse", "🛠️ Administration"], index=0, label_visibility="collapsed")

    st.divider()
    st.header("Paramètres")
    st.caption("Laissez par défaut si votre scan est propre.")

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
    st.markdown(
        f"""
<div class="card card-tight">
<b>Normes actives</b><br/>
<span class="muted">{'Personnalisées (v'+str(st.session_state.norms_version)+')' if st.session_state.norms_df is not None else 'Fichier data/norms.csv'}</span>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# ADMIN PAGE
# =============================================================================
if nav == "🛠️ Administration":
    st.markdown("### 🛠️ Administration — Normes & versioning")

    st.markdown(
        """
<div class="card">
<b>Objectif</b><br/>
<span class="muted">Importer des normes (norms.csv), valider le schéma, activer la version en session.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Importer norms.csv")
    norms_upload = st.file_uploader("Fichier normes (CSV)", type=["csv"])

    if norms_upload is not None:
        raw = norms_upload.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            st.error("Impossible de lire le CSV.")
            st.exception(e)
            st.stop()

        ok, errors = validate_norms_schema(df)
        if not ok:
            st.error("Fichier invalide :")
            for e in errors:
                st.write(f"- {e}")
            st.stop()

        version = hashlib.sha256(raw).hexdigest()[:12]
        st.session_state.norms_df = df
        st.session_state.norms_version = version
        st.success(f"Normes chargées ✅  (Version: {version})")
        st.dataframe(df.head(50), use_container_width=True)

    st.divider()
    st.subheader("Réinitialiser")
    if st.button("♻️ Revenir aux normes par défaut", use_container_width=True):
        st.session_state.norms_df = None
        st.session_state.norms_version = None
        st.success("Normes réinitialisées ✅")

    st.divider()
    st.subheader("Schéma attendu (CSV)")
    st.code(
        "Colonnes requises: scale_type, scale, sex, age_min, age_max, mean, sd\n"
        "Exemple domaine: domain,N,M,18,25,mean,sd\n"
        "Rappel: sd > 0 et age_min <= age_max",
        language="text",
    )
    st.stop()


# =============================================================================
# ANALYSE PAGE
# =============================================================================
st.markdown("### 1) Import")

cL, cR = st.columns([1.2, 0.8], vertical_alignment="top")
with cL:
    img_file = st.file_uploader("Image/scan de la feuille de réponses", type=["jpg", "jpeg", "png", "webp"])
with cR:
    st.markdown(
        """
<div class="card">
<b>Conseils de capture</b><br/>
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

st.markdown("### 2) Fichiers & options")
f1, f2, f3 = st.columns([1, 1, 1], vertical_alignment="top")
with f1:
    key_file = st.file_uploader("Clé de cotation (scoring_key.csv) — optionnel", type=["csv"])
with f2:
    logo_file = st.file_uploader("Logo (PNG/JPG) pour PDF — optionnel", type=["png", "jpg", "jpeg"])
with f3:
    st.markdown(
        """
<div class="card card-tight">
<b>Info</b><br/>
<span class="muted">La clé et les normes par défaut sont intégrées au projet.</span>
</div>
""",
        unsafe_allow_html=True,
    )

# Charger clé de cotation
if key_file is not None:
    scoring_key = load_scoring_key_from_bytes(key_file.getvalue())
else:
    default_key_path = Path(__file__).resolve().parents[1] / "data" / "scoring_key.csv"
    scoring_key = load_scoring_key_from_bytes(default_key_path.read_bytes())

# Normes : session (admin) sinon disque
norms_df = st.session_state.norms_df if st.session_state.norms_df is not None else _load_norms_df_from_disk()

# Config scanner (verrouillée sur la feuille officielle)
cfg = OMRConfig(
    rows=30,
    cols=8,
    mark_threshold=float(mark_threshold),
    ambiguity_gap=float(ambiguity_gap),
    detect_blue=bool(detect_blue),
    detect_black=bool(detect_black),
    black_dark_thresh=int(black_dark_thresh),
    black_baseline_quantile=float(black_baseline_quantile),
)

scanner = OMRScanner(cfg=cfg)

# Stable cache key
cfg_sig = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False).encode("utf-8")
cache_key = f"{img_hash}:{_hash_bytes(cfg_sig)}"

st.markdown("### 3) Scan & contrôle qualité")
btn_cols = st.columns([0.55, 0.45])
with btn_cols[0]:
    run_scan = st.button("🚀 Lancer le scan & la cotation", use_container_width=True)
with btn_cols[1]:
    st.markdown('<div class="pill">Le scan ne se relance que sur clic.</div>', unsafe_allow_html=True)

if run_scan:
    with st.spinner("Analyse en cours…"):
        try:
            result = scanner.scan_pil(pil_img, scoring_key)
        except Exception as e:
            st.error("Échec du scan. Détails :")
            st.exception(e)
            st.stop()
    st.session_state.scan_cache[cache_key] = result

result = st.session_state.scan_cache.get(cache_key)
if result is None:
    st.info("Cliquez sur **Lancer le scan & la cotation** pour démarrer.")
    st.stop()

# Cohérence attendue
if len(result.responses_final) != 240:
    st.error(
        f"Grille mal détectée : {len(result.responses_final)} réponses (attendu 240). "
        "Vérifiez cadrage (feuille entière + bords visibles) et relancez."
    )
    st.stop()

stats = result.diagnostics.get("stats", {})
proto = result.protocol or {}

k1, k2, k3, k4 = st.columns(4)
with k1:
    nb = int(proto.get("n_blank", 0))
    _kpi("Réponses vides", nb, tone="warn" if nb > 0 else "ok")
with k2:
    amb = int(stats.get("ambiguous", 0))
    _kpi("Ambiguës", amb, tone="warn" if amb > 0 else "ok")
with k3:
    low = int(stats.get("low_conf", 0))
    _kpi("Faible confiance", low, tone="warn" if low > 0 else "ok")
with k4:
    imp = int(proto.get("imputed", 0))
    _kpi("Imputées", imp, tone="bad" if imp > 0 else "ok")

st.success(f"Scan terminé ✅ — scan_id : {result.scan_id}")

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Résultats", "🧠 Normes & Interprétation", "🔍 Qualité", "📄 Rapport PDF", "⬇️ Exports"]
)

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
    st.caption("Corrigez uniquement les items vides/ambiguës/faible confiance, puis recalculer.")

    flagged: list[int] = []
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
                        help="-1 = vide ; 0..4 = FD/D/N/A/FA",
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
                    st.error("Échec du recalcul. Détails :")
                    st.exception(e)

# --- TAB 2: Normes & interprétation (Z/T + graphiques)
with tab2:
    st.markdown("#### Calcul des scores normés (Z / T)")
    st.caption("T = 50 + 10×Z — dépend des normes actives.")

    domain_norm_detail: Dict[str, Any] = {}
    domain_t: Dict[str, float] = {}

    try:
        for d in DOMAINS:
            raw = float(result.domain_scores.get(d, 0))
            mean, sd = _pick_norms(norms_df, scale_type="domain", scale=d, sex=sex, age=int(age))
            res = _z_t(raw, mean, sd)
            domain_t[d] = float(res["t"])
            domain_norm_detail[d] = {"raw": int(raw), "mean": float(mean), "sd": float(sd), **res}
    except Exception as e:
        st.warning(f"Impossible de calculer les scores T : {e}")
        domain_norm_detail = {}
        domain_t = {}

    if domain_norm_detail:
        df_dom = pd.DataFrame(domain_norm_detail).T
        df_dom["interprétation"] = df_dom["t"].apply(lambda x: interpret_t(float(x)))
        st.subheader("Interprétation automatique (Domaines)")
        st.dataframe(df_dom[["raw", "mean", "sd", "z", "t", "interprétation"]], use_container_width=True)

        # Figures
        labels = DOMAINS
        y = [float(domain_norm_detail.get(k, {}).get("t", np.nan)) for k in labels]
        x = np.arange(len(labels))

        fig_line = plt.figure(figsize=(8.6, 3.2))
        ax = fig_line.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_xticks(x, labels)
        ax.set_ylim(20, 80)
        ax.set_ylabel("Score T")
        ax.set_title("Profil global (Scores T) — N/E/O/A/C")
        ax.grid(True, alpha=0.25)

        vals = y
        vals2 = vals + vals[:1]
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ang2 = ang + ang[:1]
        fig_radar = plt.figure(figsize=(5.6, 5.6))
        axr = fig_radar.add_subplot(111, polar=True)
        axr.plot(ang2, vals2)
        axr.fill(ang2, vals2, alpha=0.15)
        axr.set_thetagrids(np.degrees(ang), labels)
        axr.set_ylim(20, 80)
        axr.set_title("Radar (Scores T)", pad=18)

        g1, g2 = st.columns([1.2, 0.8])
        with g1:
            st.pyplot(fig_line, clear_figure=True)
        with g2:
            st.pyplot(fig_radar, clear_figure=True)
    else:
        st.info("Aucun score T calculé. Vérifiez les normes (Administration) ou data/norms.csv.")

# --- TAB 3: Qualité
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

# --- TAB 4: PDF Report
with tab4:
    st.markdown("#### Rapport PDF (multi-pages, FR)")
    st.caption("Inclut : couverture + domaines (brut/T/interprétation) + graphiques + facettes.")

    # Recalcule T-scores pour le PDF (si possible)
    domain_norm_detail_pdf: Dict[str, Any] = {}
    try:
        for d in DOMAINS:
            raw = float(result.domain_scores.get(d, 0))
            mean, sd = _pick_norms(norms_df, scale_type="domain", scale=d, sex=sex, age=int(age))
            res = _z_t(raw, mean, sd)
            domain_norm_detail_pdf[d] = {"raw": int(raw), "mean": float(mean), "sd": float(sd), **res}
    except Exception:
        domain_norm_detail_pdf = {}

    fig_line_png = None
    fig_radar_png = None

    if domain_norm_detail_pdf:
        labels = DOMAINS
        y = [float(domain_norm_detail_pdf.get(k, {}).get("t", np.nan)) for k in labels]
        x = np.arange(len(labels))

        fig_line = plt.figure(figsize=(8.6, 3.2))
        ax = fig_line.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_xticks(x, labels)
        ax.set_ylim(20, 80)
        ax.set_ylabel("Score T")
        ax.set_title("Profil global (Scores T) — N/E/O/A/C")
        ax.grid(True, alpha=0.25)
        fig_line_png = fig_to_png_bytes(fig_line)

        vals = y
        vals2 = vals + vals[:1]
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ang2 = ang + ang[:1]
        fig_radar = plt.figure(figsize=(5.6, 5.6))
        axr = fig_radar.add_subplot(111, polar=True)
        axr.plot(ang2, vals2)
        axr.fill(ang2, vals2, alpha=0.15)
        axr.set_thetagrids(np.degrees(ang), labels)
        axr.set_ylim(20, 80)
        axr.set_title("Radar (Scores T)", pad=18)
        fig_radar_png = fig_to_png_bytes(fig_radar)
    else:
        st.warning("Scores T non disponibles → le PDF sera généré sans graphiques et sans T-scores.")

    logo_bytes = logo_file.getvalue() if logo_file is not None else None
    subject_info = {"Sexe": sex, "Âge": int(age)}

    try:
        pdf_bytes = build_report_pdf(
            logo_bytes=logo_bytes,
            subject=subject_info,
            scan_id=str(result.scan_id),
            domain_scores_raw=result.domain_scores,
            facet_scores_raw=result.facette_scores,
            domain_norm_detail=domain_norm_detail_pdf,
            fig_line_png=fig_line_png,
            fig_radar_png=fig_radar_png,
        )

        st.download_button(
            "⬇️ Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"NEO_PI-R_Rapport_{result.scan_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error("Impossible de générer le PDF (ReportLab). Détails :")
        st.exception(e)

# --- TAB 5: Exports
with tab5:
    st.markdown("#### Exports (CSV / JSON)")

    # CSV: item,choice_index,choice_label
    resp_csv = io.StringIO()
    resp_csv.write("item,choice_index,choice_label\n")
    for item_id in range(1, 241):
        idx = int(result.responses_final.get(item_id, -1))
        lab = CHOICES[idx] if idx in range(5) else ""
        resp_csv.write(f"{item_id},{idx},{lab}\n")

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
                    "norms_version": st.session_state.norms_version,
                    "choices_order": CHOICES,
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
            json.dumps({"diagnostics": result.diagnostics, "meta": result.meta}, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            "audit.json",
            "application/json",
        )
