import os
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

from config import *
from models.mobilenetv2 import get_mobilenetv2

st.set_page_config(
    page_title="ROAD.QC",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# OOD (out-of-distribution) detection thresholds
# ──────────────────────────────────────────────────────────────
# An image is rejected as "not a road" if EITHER:
#   • max class probability falls below CONFIDENCE_THRESHOLD, OR
#   • prediction entropy exceeds ENTROPY_THRESHOLD
#
# Entropy ranges from 0 (fully confident in one class) to log(NUM_CLASSES)
# (uniform over all classes). For 4 classes, max entropy ≈ 1.386.
# Tune these higher to reject more aggressively, lower to accept more.
CONFIDENCE_THRESHOLD = 0.55
ENTROPY_THRESHOLD = 1.15

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Theme tokens (light mode default) ──────────────────── */
:root {
    --bg:            #fafafa;
    --surface:       #ffffff;
    --surface-alt:   #fafafa;
    --border:        #e4e4e7;
    --border-strong: #d4d4d8;
    --divider:       #f4f4f5;
    --hover:         #f4f4f5;
    --text:          #09090b;
    --text-body:     #18181b;
    --text-muted:    #52525b;
    --text-faint:    #a1a1aa;
    --bar-empty:     #f4f4f5;

    --btn-bg:        #09090b;
    --btn-fg:        #fafafa;
    --btn-border:    #09090b;
    --btn-hover:     #18181b;

    --good:          #16a34a;
    --good-bg:       #f0fdf4;
    --good-fg:       #166534;
    --satis:         #2563eb;
    --satis-bg:      #eff6ff;
    --satis-fg:      #1e40af;
    --poor:          #d97706;
    --poor-bg:       #fffbeb;
    --poor-fg:       #92400e;
    --vpoor:         #dc2626;
    --vpoor-bg:      #fef2f2;
    --vpoor-fg:      #991b1b;

    --unknown:       #71717a;
    --unknown-bg:    #f4f4f5;
    --unknown-fg:    #52525b;

    --online:        #16a34a;
    --shadow-card:   0 4px 16px rgba(0,0,0,0.03);
    --shadow-btn:    0 6px 16px rgba(0,0,0,0.12);
}

/* ── Dark mode overrides ────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:            #0a0a0a;
        --surface:       #161618;
        --surface-alt:   #1c1c1f;
        --border:        #27272a;
        --border-strong: #3f3f46;
        --divider:       #27272a;
        --hover:         #1f1f22;
        --text:          #fafafa;
        --text-body:     #e4e4e7;
        --text-muted:    #a1a1aa;
        --text-faint:    #71717a;
        --bar-empty:     #27272a;

        --btn-bg:        #fafafa;
        --btn-fg:        #09090b;
        --btn-border:    #fafafa;
        --btn-hover:     #e4e4e7;

        --good:          #22c55e;
        --good-bg:       rgba(34, 197, 94, 0.12);
        --good-fg:       #4ade80;
        --satis:         #3b82f6;
        --satis-bg:      rgba(59, 130, 246, 0.12);
        --satis-fg:      #60a5fa;
        --poor:          #f59e0b;
        --poor-bg:       rgba(245, 158, 11, 0.12);
        --poor-fg:       #fbbf24;
        --vpoor:         #ef4444;
        --vpoor-bg:      rgba(239, 68, 68, 0.12);
        --vpoor-fg:      #f87171;

        --unknown:       #a1a1aa;
        --unknown-bg:    rgba(113, 113, 122, 0.15);
        --unknown-fg:    #a1a1aa;

        --online:        #22c55e;
        --shadow-card:   0 4px 16px rgba(0,0,0,0.4);
        --shadow-btn:    0 6px 16px rgba(0,0,0,0.5);
    }
}

/* ── Base ───────────────────────────────────────────────── */
.stApp {
    background: var(--bg) !important;
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
}
.stApp, .stApp * {
    color: var(--text-body);
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.main .block-container {
    max-width: 740px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

.mono {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ── System bar ─────────────────────────────────────────── */
.sys-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1.1rem;
    border: 1px solid var(--border);
    background: var(--surface);
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    margin-bottom: 3.5rem;
    border-radius: 6px;
}
.sys-bar span { color: var(--text-body) !important; }
.sys-bar .muted { color: var(--text-faint) !important; }
.sys-bar .left { display: flex; gap: 1rem; align-items: center; }
.sys-status { display: flex; align-items: center; gap: 0.5rem; }
.sys-status-dot {
    width: 7px; height: 7px;
    background: var(--online);
    border-radius: 50%;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100% {opacity:1;} 50% {opacity:0.35;} }

/* ── Hero ───────────────────────────────────────────────── */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: var(--text-muted) !important;
    margin-bottom: 1.1rem;
    font-weight: 500;
}
.hero-title {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.035em;
    line-height: 1.08;
    margin: 0 0 1.25rem 0;
    color: var(--text) !important;
}
.hero-sub {
    font-size: 0.95rem;
    line-height: 1.65;
    color: var(--text-muted) !important;
    max-width: 520px;
    margin: 0 0 3rem 0;
}

/* ── Legend ─────────────────────────────────────────────── */
.legend {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 3rem;
}
.legend-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    color: var(--text-faint) !important;
    margin-bottom: 0.85rem;
}
.legend-row {
    display: grid;
    grid-template-columns: 130px 1fr;
    padding: 0.4rem 0;
    align-items: center;
    font-size: 0.86rem;
}
.legend-row + .legend-row { border-top: 1px solid var(--divider); }
.legend-key {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-weight: 500;
    color: var(--text-body) !important;
}
.legend-desc { color: var(--text-muted) !important; }

.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    display: inline-block;
}
.dot-good          { background: var(--good); }
.dot-satisfactory  { background: var(--satis); }
.dot-poor          { background: var(--poor); }
.dot-very-poor     { background: var(--vpoor); }
.dot-unknown       { background: var(--unknown); }

/* ── Section header ─────────────────────────────────────── */
.section-h {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin: 2.75rem 0 1rem 0;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    color: var(--text-body) !important;
}
.section-h::after {
    content: "";
    flex: 1;
    border-top: 1px solid var(--border);
}
.section-h .num { color: var(--text-faint) !important; }
.section-h .count { color: var(--text-faint) !important; font-weight: 500; }

/* ── File uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: 10px;
    padding: 0.4rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-alt) !important;
    border: none !important;
    border-radius: 8px;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: var(--text-muted) !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-faint) !important;
}

/* Browse button — ghost style */
[data-testid="stFileUploader"] button {
    background: var(--surface) !important;
    border: 1px solid var(--text-body) !important;
    border-radius: 6px !important;
    padding: 0.45rem 1rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    min-width: auto !important;
    height: auto !important;
    transition: background 0.15s ease !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button div {
    color: var(--text-body) !important;
}
[data-testid="stFileUploader"] button svg {
    fill: var(--text-body) !important;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--hover) !important;
}

/* ── Uploaded file chips ────────────────────────────────── */
[data-testid="stFileUploaderFile"] {
    background: var(--surface-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.55rem !important;
}
[data-testid="stFileUploaderFile"] svg,
[data-testid="stFileUploaderFileData"] svg {
    color: var(--text-faint) !important;
    fill: var(--text-faint) !important;
    opacity: 0.7;
}
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileName"] * {
    color: var(--text-body) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] small * {
    color: var(--text-faint) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.02em;
}
[data-testid="stFileUploaderDeleteBtn"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDeleteBtn"] svg,
[data-testid="stFileUploaderDeleteBtn"] * {
    color: var(--text-muted) !important;
    fill: var(--text-muted) !important;
    opacity: 1 !important;
}
[data-testid="stFileUploaderDeleteBtn"]:hover svg {
    color: var(--vpoor) !important;
    fill: var(--vpoor) !important;
}

/* ── Report card ────────────────────────────────────────── */
.report {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.report:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-card);
}
.report-rejected .report-name { color: var(--text-faint) !important; }

.report-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--text-faint) !important;
    margin-bottom: 1rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--divider);
}
.report-meta span { color: var(--text-faint) !important; }

.report-class {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}
.report-name {
    font-size: 1.85rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text) !important;
    line-height: 1;
}
.report-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
}
.badge-good          { background: var(--good-bg); }
.badge-good *        { color: var(--good-fg) !important; }
.badge-satisfactory  { background: var(--satis-bg); }
.badge-satisfactory *{ color: var(--satis-fg) !important; }
.badge-poor          { background: var(--poor-bg); }
.badge-poor *        { color: var(--poor-fg) !important; }
.badge-very-poor     { background: var(--vpoor-bg); }
.badge-very-poor *   { color: var(--vpoor-fg) !important; }
.badge-unknown       { background: var(--unknown-bg); }
.badge-unknown *     { color: var(--unknown-fg) !important; }

/* Confidence */
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    margin-bottom: 0.6rem;
}
.conf-label { color: var(--text-faint) !important; }
.conf-value {
    color: var(--text-body) !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    letter-spacing: 0;
}
.conf-blocks {
    display: flex;
    gap: 2px;
    margin-bottom: 1.25rem;
}
.conf-blk {
    flex: 1;
    height: 6px;
    background: var(--bar-empty);
    border-radius: 1px;
}
.blk-good          { background: var(--good) !important; }
.blk-satisfactory  { background: var(--satis) !important; }
.blk-poor          { background: var(--poor) !important; }
.blk-very-poor     { background: var(--vpoor) !important; }
.blk-unknown       { background: var(--unknown) !important; }

.report-note {
    font-size: 0.88rem;
    line-height: 1.55;
    color: var(--text-muted) !important;
    padding-top: 1rem;
    border-top: 1px solid var(--divider);
    margin: 0;
}

/* ── Primary button (Execute Analysis) ──────────────────── */
.stButton > button {
    width: 100%;
    background: var(--btn-bg) !important;
    color: var(--btn-fg) !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 8px !important;
    padding: 0.9rem 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
    margin-top: 1rem;
}
.stButton > button:hover {
    background: var(--btn-hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-btn);
}
.stButton > button:focus { box-shadow: none !important; }
.stButton > button * { color: var(--btn-fg) !important; }

/* ── Images ─────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* ── Spinner ────────────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODELS_DIR = "trained_models"
    WEIGHTS_FILENAME = "mobilenetv2_best.pth"
    REL_PATH = os.path.join(MODELS_DIR, WEIGHTS_FILENAME)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here]
    current = here
    for _ in range(4):
        current = os.path.dirname(current)
        candidates.append(current)

    MODEL_PATH = None
    for d in candidates:
        candidate_path = os.path.join(d, REL_PATH)
        if os.path.exists(candidate_path):
            MODEL_PATH = candidate_path
            break
        legacy_path = os.path.join(d, WEIGHTS_FILENAME)
        if os.path.exists(legacy_path):
            MODEL_PATH = legacy_path
            break

    if MODEL_PATH is None:
        st.error(f"Could not locate `{REL_PATH}` in any expected directory.")
        st.stop()

    model = get_mobilenetv2(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def predict_batch(images):
    """
    Returns a list of dicts:
      { 'label': str|None, 'confidence': float, 'is_road': bool, 'entropy': float }
    label is None when the image is rejected as not-a-road.
    """
    results = []
    for img in images:
        image = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        max_conf = conf.item()
        ent = entropy.item()
        is_road = (max_conf >= CONFIDENCE_THRESHOLD) and (ent <= ENTROPY_THRESHOLD)

        results.append({
            'label': CLASS_NAMES[pred.item()] if is_road else None,
            'confidence': max_conf,
            'is_road': is_road,
            'entropy': ent,
        })
    return results


LABEL_META = {
    "good":         ("Good",         "GOOD",         "good",         "Surface integrity intact. No intervention required."),
    "satisfactory": ("Satisfactory", "SATISFACTORY", "satisfactory", "Minor wear observed. Routine monitoring advised."),
    "poor":         ("Poor",         "POOR",         "poor",         "Notable deterioration. Maintenance recommended."),
    "very_poor":    ("Very Poor",    "VERY POOR",    "very-poor",    "Severe damage detected. Immediate repair required."),
}


def confidence_bar(pct: float, key: str) -> str:
    """Render a segmented 24-block confidence bar."""
    total = 24
    filled = int(round(pct / (100 / total)))
    blocks = "".join(
        f'<div class="conf-blk {"blk-" + key if i < filled else ""}"></div>'
        for i in range(total)
    )
    return f'<div class="conf-blocks">{blocks}</div>'


# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="sys-bar">
  <div class="left">
    <span>ROAD.QC</span>
    <span class="muted">MOBILENETV2</span>
  </div>
  <div class="sys-status">
    <span class="sys-status-dot"></span>
    <span>ONLINE</span>
  </div>
</div>

<div class="hero-eyebrow">PAVEMENT CONDITION ASSESSMENT</div>
<div class="hero-title">Inspect road surface integrity.</div>
<div class="hero-sub">Upload pavement imagery for automated condition classification. Each input is scored across four severity classes with confidence weighting and remediation guidance. Images not recognized as road surfaces are flagged and skipped.</div>

<div class="legend">
  <div class="legend-title">CLASSIFICATION REFERENCE</div>
  <div class="legend-row">
    <div class="legend-key"><span class="dot dot-good"></span>Good</div>
    <div class="legend-desc">Surface in good condition</div>
  </div>
  <div class="legend-row">
    <div class="legend-key"><span class="dot dot-satisfactory"></span>Satisfactory</div>
    <div class="legend-desc">Minor wear, monitoring recommended</div>
  </div>
  <div class="legend-row">
    <div class="legend-key"><span class="dot dot-poor"></span>Poor</div>
    <div class="legend-desc">Repair recommended</div>
  </div>
  <div class="legend-row">
    <div class="legend-key"><span class="dot dot-very-poor"></span>Very Poor</div>
    <div class="legend-desc">Immediate repair required</div>
  </div>
  <div class="legend-row">
    <div class="legend-key"><span class="dot dot-unknown"></span>Unrecognized</div>
    <div class="legend-desc">Image does not appear to be a road</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-h"><span class="num">01</span><span>UPLOAD</span></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload road images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    images = [Image.open(f).convert("RGB") for f in uploaded_files]

    st.markdown(
        f'<div class="section-h"><span class="num">02</span><span>PREVIEW</span>'
        f'<span class="count">{len(images):02d} FILE{"S" if len(images) > 1 else ""}</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(min(3, len(images)))
    for idx, img in enumerate(images):
        with cols[idx % len(cols)]:
            st.image(img, use_container_width=True)

    if st.button("Execute Analysis"):
        with st.spinner("Processing..."):
            results = predict_batch(images)

        st.markdown('<div class="section-h"><span class="num">03</span><span>ASSESSMENT</span></div>', unsafe_allow_html=True)

        for idx, res in enumerate(results):
            pct = res['confidence'] * 100
            filename = uploaded_files[idx].name

            if res['is_road']:
                name, name_caps, key, note = LABEL_META[res['label']]
                bar_html = confidence_bar(pct, key)
                card_class = "report"
            else:
                name = "Unrecognized"
                name_caps = "NOT A ROAD"
                key = "unknown"
                note = (
                    "Image does not appear to contain a road surface. "
                    "The model could not classify it with sufficient confidence. "
                    "Please upload a clear pavement image."
                )
                bar_html = confidence_bar(pct, key)
                card_class = "report report-rejected"

            st.markdown(f"""
            <div class="{card_class}">
              <div class="report-meta">
                <span>IMG_{idx + 1:02d}</span>
                <span>{filename}</span>
              </div>
              <div class="report-class">
                <div class="report-name">{name}</div>
                <div class="report-badge badge-{key}">
                  <span class="dot dot-{key}"></span><span>{name_caps}</span>
                </div>
              </div>
              <div class="conf-header">
                <span class="conf-label">CONFIDENCE</span>
                <span class="conf-value">{pct:.1f}%</span>
              </div>
              {bar_html}
              <p class="report-note">{note}</p>
            </div>
            """, unsafe_allow_html=True)