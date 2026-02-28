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


st.set_page_config(
    page_title="NEO PI‑R — OMR Scanner & Scoring (Scientific)",
    page_icon="🧾",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .tiny {opacity:.85; font-size: .9rem;}
      .card {padding: 1rem 1.2rem; border-radius: 18px; border: 1px solid rgba(255,255,255,.08);
             background: rgba(255,255,255,.03);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧾 نظام المسح الضوئي (OMR) لاختبار NEO PI‑R")
st.caption("هدف التطبيق: **مسح ورقة الإجابة** ➜ **استخراج الإجابات** ➜ **حساب الدرجات علمياً** ➜ **إخراج تقرير ورسوم بيانية**.")


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
        raise ValueError("لا توجد معايير (norms) مطابقة للجنس/العمر المحددين.")
    row = sub.iloc[0]
    return float(row["mean"]), float(row["sd"])


def _z_t(raw: float, mean: float, sd: float) -> Dict[str, float]:
    if sd <= 0:
        raise ValueError("قيمة الانحراف المعياري SD غير صحيحة في ملف المعايير.")
    z = (raw - mean) / sd
    t = 50.0 + 10.0 * z
    return {"z": float(z), "t": float(t)}


def _plot_curve(domain_t: Dict[str, float]):
    labels = ["N", "E", "O", "A", "C"]
    y = [domain_t.get(k, np.nan) for k in labels]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(7.5, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xticks(x, labels)
    ax.set_ylim(20, 80)
    ax.set_ylabel("T-score")
    ax.set_title("المنحنى العام لمجالات الشخصية (T-scores)")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def _plot_radar(domain_t: Dict[str, float]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_t.get(k, np.nan) for k in labels]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(20, 80)
    ax.set_title("خريطة رادارية للمجالات (T-scores)", pad=18)
    st.pyplot(fig, clear_figure=True)


policy = SecurityPolicy(max_upload_mb=15)

with st.sidebar:
    st.header("الإعدادات العلمية")
    st.caption("اضبط الحساسية عند الحاجة. الإعدادات الافتراضية غالباً كافية.")

    mark_threshold = st.slider("عتبة العلامة (mark_threshold)", 0.5, 6.0, 1.7, 0.1)
    ambiguity_gap = st.slider("فارق الغموض (ambiguity_gap)", 0.1, 6.0, 0.9, 0.1)

    st.divider()
    st.subheader("الحبر")
    detect_blue = st.checkbox("التقاط الحبر الأزرق", value=True)
    detect_black = st.checkbox("التقاط الحبر الأسود", value=True)
    black_dark_thresh = st.slider("عتبة السواد (black_dark_thresh)", 60, 180, 110, 1)
    black_baseline_quantile = st.slider("Baseline quantile للحبر الأسود", 0.0, 50.0, 15.0, 1.0)

    st.divider()
    st.subheader("المعايير والتقنين")
    sex = st.selectbox("الجنس (للـ norms)", options=["M", "F"], index=0)
    age = st.number_input("العمر", min_value=10, max_value=90, value=25, step=1)
    st.caption("المعايير الموجودة حالياً **تجريبية** داخل الملف norms.csv. يمكنك لاحقاً استبدالها بمعاييرك الرسمية.")

    st.divider()
    st.subheader("ملفات")
    key_file = st.file_uploader("ملف مفتاح التصحيح scoring_key.csv", type=["csv"])
    norms_file = st.file_uploader("ملف المعايير norms.csv (اختياري)", type=["csv"])


left, right = st.columns([1.25, 0.75], vertical_alignment="top")
with left:
    img_file = st.file_uploader("📷 ارفع صورة/سكانر ورقة الإجابة", type=["jpg", "jpeg", "png", "webp"])
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("نصائح لنتيجة دقيقة")
    st.markdown(
        "- صورة واضحة بدون اهتزاز
"
        "- الورقة كاملة داخل الإطار
"
        "- إضاءة متجانسة بدون ظل قوي
"
        "- تجنب الانعكاس اللامع
"
        "- استخدم قلم أزرق/أسود واضح",
    )
    st.markdown("</div>", unsafe_allow_html=True)

if not img_file:
    st.stop()

validate_file_bytes(img_file.name, img_file.size, policy)
img_bytes = img_file.getvalue()
img_hash = _hash_bytes(img_bytes)

pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")


if key_file is not None:
    scoring_key = load_scoring_key_from_bytes(key_file.getvalue())
else:
    default_key_path = Path(__file__).resolve().parents[1] / "data" / "scoring_key.csv"
    scoring_key = load_scoring_key_from_bytes(default_key_path.read_bytes())


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


if "scan_cache" not in st.session_state:
    st.session_state.scan_cache = {}

cfg_sig = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False)
cache_key = f"{img_hash}:{_hash_bytes(cfg_sig.encode('utf-8'))}:key"

run_scan = st.button("🚀 تنفيذ المسح والحساب", use_container_width=True)

if run_scan:
    with st.spinner("جارٍ المسح…"):
        result = scanner.scan_pil(pil_img, scoring_key)
    st.session_state.scan_cache[cache_key] = result

result = st.session_state.scan_cache.get(cache_key)
if result is None:
    st.info("اضغط زر **تنفيذ المسح والحساب** لبدء المعالجة.")
    st.stop()


st.success(f"تم المسح بنجاح ✅  (scan_id: {result.scan_id})")

m = result.diagnostics.get("stats", {})
proto = result.protocol or {}

m1, m2, m3, m4 = st.columns(4)
m1.metric("الأسئلة الفارغة", int(proto.get("n_blank", 0)))
m2.metric("الغامضة", int(m.get("ambiguous", 0)))
m3.metric("ثقة منخفضة", int(m.get("low_conf", 0)))
m4.metric("تمّ تعويضها", int(proto.get("imputed", 0)))

with st.expander("🔍 تفاصيل تقنية (للتحقق)", expanded=False):
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

v1, v2 = st.columns([1, 1])
with v1:
    st.subheader("Overlay (التحقق من التحديد)")
    st.image(result.overlay_bgr[:, :, ::-1], use_container_width=True)
with v2:
    st.subheader("Mask (حبر/تحديد — Debug)")
    st.image(result.debug_mask, use_container_width=True)


st.subheader("📌 النتائج الأولية (Raw Scores)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**مجالات الشخصية (N/E/O/A/C)**")
    st.json(result.domain_scores)
with c2:
    st.markdown("**الواجهات (Facets)**")
    st.json(result.facette_scores)


st.subheader("📈 النتائج المعيارية (Z / T)")
domain_t: Dict[str, float] = {}
domain_norm_detail: Dict[str, Any] = {}

try:
    for d, raw in result.domain_scores.items():
        mean, sd = _pick_norms(norms_df, scale_type="domain", scale=d, sex=sex, age=int(age))
        res = _z_t(float(raw), mean, sd)
        domain_t[d] = res["t"]
        domain_norm_detail[d] = {"raw": int(raw), "mean": mean, "sd": sd, **res}
except Exception as e:
    st.warning(f"تعذر حساب T-scores من ملف المعايير: {e}")
    domain_norm_detail = {}

if domain_norm_detail:
    st.dataframe(pd.DataFrame(domain_norm_detail).T, use_container_width=True)

    g1, g2 = st.columns([1.2, 0.8])
    with g1:
        _plot_curve(domain_t)
    with g2:
        _plot_radar(domain_t)


st.subheader("🧑‍🔬 مراجعة بشرية (اختيارية) + إعادة الحساب")
st.caption("استخدمها فقط إذا كان هناك فراغات/غموض. إعادة الحساب تتم عبر زر مستقل لضمان الاستقرار.")

flagged = []
for item_id in range(1, 241):
    md = result.meta.get(item_id, {})
    if md.get("blank") or md.get("ambiguous") or float(md.get("confidence", 1.0)) < 0.55:
        flagged.append(item_id)

with st.expander(f"فتح المراجعة البشرية — العناصر المحتاجة ({len(flagged)})", expanded=False):
    if not flagged:
        st.success("لا توجد عناصر تحتاج مراجعة ✅")
    else:
        st.warning("قم بتعديل ما يلزم فقط، ثم اضغط زر إعادة الحساب.")
        page_size = 12
        page = st.number_input(
            "الصفحة",
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
            col1, col2 = st.columns([0.6, 1.4])
            with col1:
                st.markdown(f"**Item {item_id}**")
                st.markdown(f"<span class='tiny'>confidence: {float(md.get('confidence', 0.0)):.2f}</span>", unsafe_allow_html=True)
            with col2:
                choice = st.selectbox(
                    f"تصحيح Item {item_id}",
                    options=[-1, 0, 1, 2, 3, 4],
                    index=[-1, 0, 1, 2, 3, 4].index(current if current in [-1,0,1,2,3,4] else -1),
                    help="-1 = فارغ ; 0..4 = خيار محدد",
                    key=f"corr_{item_id}",
                )
                if int(choice) != current:
                    corrections[item_id] = int(choice)

        if st.button("✅ تطبيق التصحيحات وإعادة الحساب", use_container_width=True):
            final_resp = dict(result.responses_final)
            final_resp.update(corrections)

            final_after_proto, proto2 = apply_protocol_rules(cfg, final_resp)
            facette_scores2, domain_scores2 = compute_scores(final_after_proto, scoring_key)

            st.success("تمت إعادة الحساب ✅")
            st.json({"protocol": proto2, "domain_scores": domain_scores2, "facette_scores": facette_scores2})


st.subheader("⬇️ تنزيل النتائج")
resp_csv = io.StringIO()
resp_csv.write("item,choice
")
for item_id in range(1, 241):
    resp_csv.write(f"{item_id},{int(result.responses_final.get(item_id, -1))}
")

colx, coly, colz = st.columns(3)
with colx:
    _download("responses.csv", resp_csv.getvalue().encode("utf-8"), "responses.csv", "text/csv")
with coly:
    _download(
        "scores.json",
        json.dumps(
            {
                "scan_id": result.scan_id,
                "protocol": result.protocol,
                "domain_scores_raw": result.domain_scores,
                "facette_scores_raw": result.facette_scores,
                "normed": domain_norm_detail,
                "cfg": asdict(cfg),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "scores.json",
        "application/json",
    )
with colz:
    _download(
        "audit.json",
        json.dumps(
            {
                "diagnostics": result.diagnostics,
                "meta": result.meta,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "audit.json",
        "application/json",
    )
