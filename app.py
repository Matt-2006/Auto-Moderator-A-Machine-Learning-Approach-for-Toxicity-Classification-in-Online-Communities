"""
Auto-Moderator — Multi-label Toxicity Classifier
COMP6885001 Natural Language Processing · Group 11
"""

import os
import re
import numpy as np
import streamlit as st
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-Moderator | NLP Group 11",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

LABEL_EMOJI = {
    "toxic":         "☠️",
    "severe_toxic":  "💀",
    "obscene":       "🤬",
    "threat":        "🔪",
    "insult":        "👊",
    "identity_hate": "🎯",
}

LABEL_DESC = {
    "toxic":         "General toxic or rude language",
    "severe_toxic":  "Highly aggressive, extreme toxicity",
    "obscene":       "Obscene or vulgar content",
    "threat":        "Threats of violence or harm",
    "insult":        "Insulting or demeaning language",
    "identity_hate": "Hate speech targeting identity groups",
}

# Per-label thresholds tuned for class imbalance
THRESHOLDS = {
    "toxic":         0.40,
    "severe_toxic":  0.30,
    "obscene":       0.40,
    "threat":        0.25,
    "insult":        0.40,
    "identity_hate": 0.25,
}
SUSPICIOUS_RATIO = 0.5

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint path — adjust as needed
BASE_DIR   = os.environ.get(
    "MODEL_BASE_DIR",
    os.path.join(os.path.dirname(__file__), "models")
)
CKPT_PATH  = os.path.join(BASE_DIR, "distilbert_model.pt")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Overall background */
.stApp {
    background: #0a0c14;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1120;
    border-right: 1px solid #1e2240;
}

/* Custom header banner */
.app-header {
    background: linear-gradient(135deg, #0f1120 0%, #1a1f3a 50%, #0f1120 100%);
    border: 1px solid #2a3060;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(46,117,182,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(192,0,0,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8ecf4;
    margin: 0 0 4px 0;
    letter-spacing: -1px;
}
.app-header .subtitle {
    color: #6b7a9e;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 300;
}
.app-header .badge {
    display: inline-block;
    background: rgba(46,117,182,0.15);
    border: 1px solid rgba(46,117,182,0.3);
    color: #4ba3c7;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    margin-top: 10px;
}

/* Verdict cards */
.verdict-toxic {
    background: rgba(192,0,0,0.08);
    border: 2px solid rgba(192,0,0,0.5);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.verdict-suspicious {
    background: rgba(237,125,49,0.08);
    border: 2px solid rgba(237,125,49,0.5);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.verdict-safe {
    background: rgba(46,125,50,0.08);
    border: 2px solid rgba(46,125,50,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(192,0,0,0.15); }
    50%       { box-shadow: 0 0 0 8px rgba(192,0,0,0); }
}
.verdict-icon { font-size: 2.5rem; margin-bottom: 4px; }
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 2px;
}
.verdict-sub { font-size: 0.82rem; color: #8892aa; margin-top: 6px; }

/* Label score bar */
.label-row {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1a1f35;
    gap: 12px;
}
.label-row:last-child { border-bottom: none; }
.label-name {
    width: 140px;
    font-size: 0.85rem;
    color: #c8d0e4;
    flex-shrink: 0;
}
.bar-wrap {
    flex: 1;
    background: #1a1f35;
    border-radius: 4px;
    height: 10px;
    position: relative;
    overflow: visible;
}
.bar-fill {
    height: 10px;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.bar-threshold {
    position: absolute;
    top: -3px;
    width: 2px;
    height: 16px;
    background: rgba(255,255,255,0.3);
    border-radius: 1px;
}
.label-score {
    width: 48px;
    text-align: right;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    flex-shrink: 0;
}
.label-status {
    width: 100px;
    text-align: center;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
    font-weight: 500;
}
.status-toxic    { background: rgba(192,0,0,0.2); color: #ff6b6b; border: 1px solid rgba(192,0,0,0.4); }
.status-suspicious { background: rgba(237,125,49,0.2); color: #ffaa70; border: 1px solid rgba(237,125,49,0.4); }
.status-safe     { background: rgba(46,125,50,0.15); color: #6fcf97; border: 1px solid rgba(46,125,50,0.3); }

/* Input styling */
.stTextArea textarea {
    background: #0f1120 !important;
    border: 1px solid #2a3060 !important;
    border-radius: 10px !important;
    color: #e8ecf4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #2e75b6 !important;
    box-shadow: 0 0 0 2px rgba(46,117,182,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f4e79, #2e75b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(46,117,182,0.4) !important;
}

/* Metric cards */
.metric-card {
    background: #0f1120;
    border: 1px solid #1e2240;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #4ba3c7;
}
.metric-label { font-size: 0.75rem; color: #6b7a9e; margin-top: 2px; }

/* Section card */
.section-card {
    background: #0f1120;
    border: 1px solid #1e2240;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Batch table */
.batch-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.batch-table th {
    background: #1a1f35;
    color: #8892aa;
    padding: 10px 12px;
    text-align: left;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.batch-table td { padding: 10px 12px; border-bottom: 1px solid #1a1f35; color: #c8d0e4; }
.batch-table tr:last-child td { border-bottom: none; }
.batch-table tr:hover td { background: rgba(255,255,255,0.02); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1120;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2240;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7a9e;
    border-radius: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2240 !important;
    color: #4ba3c7 !important;
}

/* Info box */
.info-box {
    background: rgba(46,117,182,0.07);
    border: 1px solid rgba(46,117,182,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.83rem;
    color: #8892aa;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Model Architecture ────────────────────────────────────────────────────────
class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden          = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out     = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_out)

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CKPT_PATH):
        return None, None, f"Model checkpoint not found at: `{CKPT_PATH}`"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        ckpt      = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model     = DistilBERTClassifier(MODEL_NAME, len(LABEL_COLS)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(texts: list, model, tokenizer) -> np.ndarray:
    cleaned = [clean_text(t) for t in texts]
    enc = tokenizer(
        cleaned, max_length=MAX_LEN,
        padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def get_status(label: str, score: float) -> str:
    thr = THRESHOLDS[label]
    if score >= thr:               return "toxic"
    if score >= thr * SUSPICIOUS_RATIO: return "suspicious"
    return "safe"

# ── Label Bars HTML ───────────────────────────────────────────────────────────
def render_label_bars(prob_dict: dict) -> str:
    COLOR = {"toxic": "#c00000", "suspicious": "#ed7d31", "safe": "#2e7d32"}
    STATUS_LABEL = {"toxic": "🔴 Toxic", "suspicious": "🟠 Suspicious", "safe": "🟢 Safe"}

    rows = ""
    for lbl in LABEL_COLS:
        score  = prob_dict[lbl]
        status = get_status(lbl, score)
        thr    = THRESHOLDS[lbl]
        color  = COLOR[status]
        bar_w  = min(score * 100, 100)
        thr_pct = thr * 100

        rows += f"""
        <div class="label-row">
            <div class="label-name">{LABEL_EMOJI[lbl]} {lbl}</div>
            <div class="bar-wrap">
                <div class="bar-fill" style="width:{bar_w:.1f}%;background:{color};"></div>
                <div class="bar-threshold" style="left:{thr_pct:.1f}%;"></div>
            </div>
            <div class="label-score" style="color:{color};">{score:.1%}</div>
            <div class="label-status status-{status}">{STATUS_LABEL[status]}</div>
        </div>"""
    return rows

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#4ba3c7;font-weight:700;">🛡️ AUTO-MODERATOR</div>
        <div style="font-size:0.75rem;color:#6b7a9e;margin-top:4px;">NLP Group 11 · COMP6885001</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Info**")
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:8px;">
        <div class="metric-val">0.9851</div>
        <div class="metric-label">ROC-AUC (macro)</div>
    </div>
    <div class="metric-card" style="margin-bottom:8px;">
        <div class="metric-val">0.6162</div>
        <div class="metric-label">F1 (macro)</div>
    </div>
    <div class="metric-card" style="margin-bottom:8px;">
        <div class="metric-val">0.0300</div>
        <div class="metric-label">Hamming Loss</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(f"""
    <div class="info-box">
        <b style="color:#c8d0e4;">DistilBERT</b><br>
        <code style="color:#4ba3c7;">distilbert-base-uncased</code><br><br>
        → Dropout(0.3)<br>
        → Linear(768 → 6)<br>
        → Sigmoid per label<br><br>
        <b>Training:</b> BCEWithLogitsLoss · AdamW lr=2e-5 · 3 epochs · batch=32<br><br>
        <b>Device:</b> <code style="color:#4ba3c7;">{DEVICE}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Labels & Thresholds**")
    for lbl in LABEL_COLS:
        thr = THRESHOLDS[lbl]
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.8rem;border-bottom:1px solid #1a1f35;">
            <span style="color:#c8d0e4;">{LABEL_EMOJI[lbl]} {lbl}</span>
            <span style="font-family:'Space Mono',monospace;color:#4ba3c7;">{thr:.0%}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#4a5168;padding-top:4px;">
        Matthew Fitch Aurick<br>
        Jayson Prasada Siswoyo<br>
        Evan Chastya Pahan<br><br>
        Bina Nusantara University · 2025/2026
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🛡️ Auto-Moderator</h1>
    <p class="subtitle">Multi-label Toxicity Classifier — Jigsaw Dataset · 6 Categories</p>
    <span class="badge">⚡ Powered by DistilBERT · ROC-AUC 0.9851 ✅</span>
</div>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("Loading DistilBERT model..."):
    model, tokenizer, load_error = load_model()

if load_error:
    st.error(f"⚠️ **Model not loaded:** {load_error}")
    st.markdown("""
    <div class="info-box">
        <b>How to fix:</b><br>
        1. Train the model using the notebook (<code>nlp_group11_fixed.ipynb</code>, Section 3.3)<br>
        2. Place <code>distilbert_model.pt</code> in the <code>models/</code> folder next to <code>app.py</code><br>
        3. Or set the environment variable: <code>MODEL_BASE_DIR=/path/to/your/output</code>
    </div>
    """, unsafe_allow_html=True)
    model_loaded = False
else:
    model_loaded = True
    st.success(f"✅ Model loaded on `{DEVICE}`")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Comment", "📋 Batch Classification", "📖 Reference"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Comment
# ════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Enter Comment")

        # Callback functions for buttons
        def clear_text():
            st.session_state["single_input"] = ""

        def set_example(example_text):
            def callback():
                st.session_state["single_input"] = example_text
            return callback

        user_input = st.text_area(
            label="comment_input",
            label_visibility="collapsed",
            placeholder="Type or paste a comment to classify…",
            height=140,
            key="single_input"
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            classify_btn = st.button("🔍 Classify", use_container_width=True, key="classify_btn")
        with col_btn2:
            st.button("🗑️ Clear", use_container_width=True, key="clear_btn", on_click=clear_text)

        st.markdown("---")
        st.markdown("**📋 Examples**")
        examples = [
            "Thank you for your great contribution! This article is much better now.",
            "You are an absolute idiot and should be banned from this platform.",
            "I will find you and make you regret everything you have done.",
            "This is a neutral comment about the weather today.",
            "Disgusting garbage, you should be ashamed of yourself.",
            "Great work on this section, very well researched!",
        ]
        for i, ex in enumerate(examples):
            st.button(f"↳ {ex[:60]}{'…' if len(ex) > 60 else ''}", key=f"ex_{i}", use_container_width=True, on_click=set_example(ex))

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown("#### 📊 Result")

        run_classify = classify_btn and user_input.strip()

        if run_classify:
            if not model_loaded:
                st.error("Model not loaded. Please check the model path.")
            else:
                with st.spinner("Analyzing..."):
                    probs      = predict([user_input.strip()], model, tokenizer)[0]
                    prob_dict  = dict(zip(LABEL_COLS, probs.tolist()))
                    statuses   = {lbl: get_status(lbl, prob_dict[lbl]) for lbl in LABEL_COLS}
                    is_toxic   = any(s == "toxic"      for s in statuses.values())
                    is_sus     = any(s == "suspicious" for s in statuses.values())

                    # Verdict
                    if is_toxic:
                        flagged = [lbl for lbl in LABEL_COLS if statuses[lbl] == "toxic"]
                        st.markdown(f"""
                        <div class="verdict-toxic">
                            <div class="verdict-icon">🔴</div>
                            <div class="verdict-label" style="color:#ff6b6b;">TOXIC</div>
                            <div class="verdict-sub">Flagged: {", ".join(flagged)}</div>
                        </div>""", unsafe_allow_html=True)
                    elif is_sus:
                        flagged = [lbl for lbl in LABEL_COLS if statuses[lbl] == "suspicious"]
                        st.markdown(f"""
                        <div class="verdict-suspicious">
                            <div class="verdict-icon">🟠</div>
                            <div class="verdict-label" style="color:#ffaa70;">SUSPICIOUS</div>
                            <div class="verdict-sub">Review: {", ".join(flagged)}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="verdict-safe">
                            <div class="verdict-icon">🟢</div>
                            <div class="verdict-label" style="color:#6fcf97;">SAFE</div>
                            <div class="verdict-sub">No toxicity detected across all 6 labels</div>
                        </div>""", unsafe_allow_html=True)

                    # Label bars
                    st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
                    bars_html = render_label_bars(prob_dict)
                    st.markdown(f'<div class="section-card" style="padding:12px 16px;">{bars_html}</div>',
                                unsafe_allow_html=True)

                    # Threshold note
                    st.markdown("""
                    <div class="info-box" style="margin-top:8px;">
                        ℹ️ Vertical bar (│) in each row marks the decision threshold.
                        Rare labels use lower thresholds to reduce false negatives.
                    </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        elif not user_input.strip():
            st.markdown("""
            <div style="padding:60px 20px; text-align:center; color:#4a5168;">
                <div style="font-size:2.5rem;margin-bottom:12px;">🛡️</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.9rem;">
                    Enter a comment and click Classify
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Classification
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📋 Batch Classification")
    st.markdown("""
    <div class="info-box">
        Enter one comment per line. Each comment will be classified independently across all 6 toxicity labels.
    </div>
    """, unsafe_allow_html=True)

    default_batch = "\n".join([
        "Thank you for your great contribution! This article is much better now.",
        "You are an absolute idiot and should be banned from this platform.",
        "I will find you and make you regret everything you have done.",
        "This is a neutral comment about the weather today.",
        "Disgusting garbage, you should be ashamed of yourself.",
        "Great work on this section, very well researched!",
    ])

    batch_input = st.text_area(
        "Comments (one per line)",
        value=default_batch,
        height=200,
        key="batch_input"
    )

    batch_btn = st.button("🔍 Classify All", use_container_width=False, key="batch_btn")

    if batch_btn:
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one comment.")
        elif not model_loaded:
            st.error("Model not loaded.")
        else:
            with st.spinner(f"Classifying {len(lines)} comments..."):
                probs = predict(lines, model, tokenizer)

            # Summary stats
            verdicts = []
            for prob_row in probs:
                sts = [get_status(lbl, float(prob_row[j])) for j, lbl in enumerate(LABEL_COLS)]
                if "toxic" in sts:       verdicts.append("toxic")
                elif "suspicious" in sts: verdicts.append("suspicious")
                else:                     verdicts.append("safe")

            n_toxic = verdicts.count("toxic")
            n_sus   = verdicts.count("suspicious")
            n_safe  = verdicts.count("safe")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(lines)}</div><div class="metric-label">Total Comments</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ff6b6b;">{n_toxic}</div><div class="metric-label">🔴 Toxic</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ffaa70;">{n_sus}</div><div class="metric-label">🟠 Suspicious</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#6fcf97;">{n_safe}</div><div class="metric-label">🟢 Safe</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Build table
            COLOR = {"toxic": "#ff6b6b", "suspicious": "#ffaa70", "safe": "#6fcf97"}
            ICON  = {"toxic": "🔴", "suspicious": "🟠", "safe": "🟢"}

            header_labels = "".join(
                f'<th style="text-align:center;">{LABEL_EMOJI[lbl]}<br><span style="font-size:0.7em;color:#6b7a9e;">{lbl[:6]}</span></th>'
                for lbl in LABEL_COLS
            )
            header = f"""
            <thead><tr>
                <th>#</th><th>Verdict</th><th style="min-width:240px;">Comment</th>
                {header_labels}
            </tr></thead>"""

            rows_html = ""
            for i, (line, prob_row, verdict) in enumerate(zip(lines, probs, verdicts)):
                icon = ICON[verdict]
                bg   = "#0f1120" if i % 2 == 0 else "#0a0c14"
                score_cells = ""
                for j, lbl in enumerate(LABEL_COLS):
                    v   = float(prob_row[j])
                    st_ = get_status(lbl, v)
                    c   = COLOR[st_]
                    bold = "font-weight:600;" if st_ != "safe" else ""
                    score_cells += f'<td style="text-align:center;font-family:Space Mono,monospace;font-size:0.8rem;color:{c};{bold}">{v:.0%}</td>'

                short = line[:80] + ("…" if len(line) > 80 else "")
                rows_html += f"""
                <tr style="background:{bg};">
                    <td style="color:#6b7a9e;font-size:0.8rem;">{i+1}</td>
                    <td style="font-size:1.1rem;">{icon}</td>
                    <td style="color:#c8d0e4;font-size:0.85rem;" title="{line}">{short}</td>
                    {score_cells}
                </tr>"""

            table_html = f"""
            <div style="overflow-x:auto;margin-top:8px;">
                <table class="batch-table">
                    {header}
                    <tbody>{rows_html}</tbody>
                </table>
            </div>"""
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:12px;">
                Scores shown are per-label probabilities. Each label uses its own threshold
                (see sidebar). 🔴 = ≥ threshold · 🟠 = ≥ ½ threshold · 🟢 = safe
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Reference
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📖 Label Definitions & Thresholds")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Toxicity Labels**")
        for lbl in LABEL_COLS:
            thr = THRESHOLDS[lbl]
            sus = thr * SUSPICIOUS_RATIO
            st.markdown(f"""
            <div class="section-card" style="margin-bottom:10px;padding:14px 18px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1rem;">{LABEL_EMOJI[lbl]} <b style="color:#e8ecf4;">{lbl}</b></span>
                    <span style="font-family:'Space Mono',monospace;color:#4ba3c7;font-size:0.85rem;">thr={thr:.0%}</span>
                </div>
                <div style="font-size:0.8rem;color:#6b7a9e;margin-top:6px;">{LABEL_DESC[lbl]}</div>
                <div style="font-size:0.75rem;margin-top:6px;">
                    <span style="color:#ff6b6b;">🔴 Toxic if ≥ {thr:.0%}</span> ·
                    <span style="color:#ffaa70;">🟠 Suspicious if ≥ {sus:.0%}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("**Verdict Logic**")
        st.markdown("""
        <div class="section-card">
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #1e2240;">
                    <td style="padding:10px 8px;color:#ff6b6b;font-size:1.1rem;">🔴</td>
                    <td style="padding:10px 8px;color:#e8ecf4;font-weight:600;">Toxic</td>
                    <td style="padding:10px 8px;color:#8892aa;">At least 1 label ≥ its threshold</td>
                </tr>
                <tr style="border-bottom:1px solid #1e2240;">
                    <td style="padding:10px 8px;color:#ffaa70;font-size:1.1rem;">🟠</td>
                    <td style="padding:10px 8px;color:#e8ecf4;font-weight:600;">Suspicious</td>
                    <td style="padding:10px 8px;color:#8892aa;">At least 1 label ≥ ½ threshold, none ≥ threshold</td>
                </tr>
                <tr>
                    <td style="padding:10px 8px;color:#6fcf97;font-size:1.1rem;">🟢</td>
                    <td style="padding:10px 8px;color:#e8ecf4;font-weight:600;">Safe</td>
                    <td style="padding:10px 8px;color:#8892aa;">All labels below ½ threshold</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Why Per-Label Thresholds?**")
        st.markdown("""
        <div class="section-card">
            <p style="font-size:0.83rem;color:#8892aa;line-height:1.6;margin:0;">
                The dataset is severely imbalanced — rare labels like <b style="color:#c8d0e4;">threat</b> (0.30%)
                and <b style="color:#c8d0e4;">identity_hate</b> (0.88%) need lower thresholds
                to avoid missing real instances (reducing false negatives).<br><br>
                A single 0.5 threshold across all labels would under-detect rare categories.
                Per-label tuning balances precision and recall for each class individually.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Model Performance Summary**")
        perf_data = {
            "DistilBERT":        {"ROC-AUC": 0.9851, "F1": 0.6162, "Hamming": 0.0300},
            "RoBERTa":           {"ROC-AUC": 0.9839, "F1": 0.6169, "Hamming": 0.0309},
            "LinearSVC":         {"ROC-AUC": 0.9747, "F1": 0.5401, "Hamming": 0.0267},
            "Logistic Regression": {"ROC-AUC": 0.9746, "F1": 0.4725, "Hamming": 0.0502},
        }
        table_rows = ""
        for name, m in perf_data.items():
            bold = "font-weight:600;" if name == "DistilBERT" else ""
            color = "#4ba3c7" if name == "DistilBERT" else "#c8d0e4"
            table_rows += f"""
            <tr style="border-bottom:1px solid #1a1f35;">
                <td style="padding:8px;font-size:0.82rem;color:{color};{bold}">{name}</td>
                <td style="padding:8px;text-align:center;font-family:'Space Mono',monospace;font-size:0.8rem;color:#4ba3c7;">{m['ROC-AUC']}</td>
                <td style="padding:8px;text-align:center;font-family:'Space Mono',monospace;font-size:0.8rem;color:#c8d0e4;">{m['F1']}</td>
                <td style="padding:8px;text-align:center;font-family:'Space Mono',monospace;font-size:0.8rem;color:#8892aa;">{m['Hamming']}</td>
            </tr>"""

        st.markdown(f"""
        <div class="section-card" style="padding:12px;">
            <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                <thead>
                    <tr style="border-bottom:2px solid #1e2240;">
                        <th style="padding:8px;text-align:left;color:#6b7a9e;font-family:'Space Mono',monospace;font-size:0.75rem;">Model</th>
                        <th style="padding:8px;text-align:center;color:#6b7a9e;font-family:'Space Mono',monospace;font-size:0.75rem;">ROC-AUC</th>
                        <th style="padding:8px;text-align:center;color:#6b7a9e;font-family:'Space Mono',monospace;font-size:0.75rem;">F1</th>
                        <th style="padding:8px;text-align:center;color:#6b7a9e;font-family:'Space Mono',monospace;font-size:0.75rem;">Hamming</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
            <div style="font-size:0.72rem;color:#4a5168;margin-top:8px;">★ App uses DistilBERT (best ROC-AUC). All models meet target > 0.90 ✅</div>
        </div>
        """, unsafe_allow_html=True)
