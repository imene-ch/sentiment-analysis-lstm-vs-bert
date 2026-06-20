# ============================================================
# SENTIMENT ANALYSIS APP — LSTM vs BERT
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── PALETTE
   bg-deep:    #0E0A07   (near-black warm brown)
   bg-card:    #1A1208   (dark espresso)
   accent:     #C8873A   (amber / burnt gold)
   accent2:    #8B5E2A   (deep brown)
   text-hi:    #EDE0CC   (warm cream)
   text-mid:   #A08060   (warm muted)
   text-lo:    #5C4530   (dim brown)
   success:    #7BAE6B   (muted olive-green)
   error:      #B85C4A   (terracotta-red)
── */

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── BACKGROUND ── */
.stApp {
    background-color: #0E0A07;
    background-image:
        linear-gradient(rgba(200, 135, 58, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(200, 135, 58, 0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── MAIN BLOCK ── */
.block-container {
    padding-top: 2.5rem !important;
    max-width: 860px !important;
}

/* ── TITLE h1 ── */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.1rem !important;
    background: linear-gradient(135deg, #C8873A 0%, #EDE0CC 60%, #C8873A 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.02em !important;
}

h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #EDE0CC !important;
    letter-spacing: -0.01em !important;
}

/* ── BODY TEXT ── */
p, .stMarkdown p, label {
    color: #A08060 !important;
    font-size: 0.95rem !important;
}

strong {
    color: #EDE0CC !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(200, 135, 58, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: #1A1208 !important;
    border: 1px solid rgba(200, 135, 58, 0.22) !important;
    border-radius: 10px !important;
    color: #EDE0CC !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    resize: vertical !important;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.stTextArea textarea:focus {
    border-color: #C8873A !important;
    box-shadow: 0 0 0 3px rgba(200, 135, 58, 0.15) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #5C4530 !important;
}

/* ── ALL BUTTONS ── */
.stButton > button,
.stButton > button[kind="primary"],
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #C8873A 0%, #8B5E2A 100%) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 10px !important;
    height: 3rem !important;
    box-shadow: 0 4px 20px rgba(200, 135, 58, 0.30) !important;
    transition: box-shadow 0.25s, transform 0.15s !important;
    opacity: 1 !important;
}
/* Force button text white via child elements */
.stButton > button p,
.stButton > button span,
.stButton > button div,
.stButton > button *,
[data-testid="stBaseButton-primary"] *,
[data-testid="baseButton-primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:hover,
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 6px 32px rgba(200, 135, 58, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #C8873A !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #5C4530 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #7BAE6B !important;
}

/* ── INFO BOX ── */
div[data-baseweb="notification"] {
    border-radius: 10px !important;
}
.stAlert,
[data-baseweb="notification"][kind="info"],
div[role="alert"] {
    background: rgba(200, 135, 58, 0.07) !important;
    border: 1px solid rgba(200, 135, 58, 0.22) !important;
    border-radius: 10px !important;
}

/* ── ERROR BOX (LSTM) ── */
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(184, 92, 74, 0.10) !important;
    border: 1px solid rgba(184, 92, 74, 0.38) !important;
    border-radius: 12px !important;
}
div[data-baseweb="notification"][kind="negative"] p {
    color: #E8A090 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

/* ── SUCCESS BOX (BERT) ── */
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(123, 174, 107, 0.09) !important;
    border: 1px solid rgba(123, 174, 107, 0.35) !important;
    border-radius: 12px !important;
}
div[data-baseweb="notification"][kind="positive"] p {
    color: #A8D898 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

/* ── WARNING ── */
div[data-baseweb="notification"][kind="warning"] {
    background: rgba(200, 135, 58, 0.09) !important;
    border: 1px solid rgba(200, 135, 58, 0.32) !important;
    border-radius: 10px !important;
}

/* ── CAPTION ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #5C4530 !important;
    font-size: 0.75rem !important;
}

/* ── TABLE ── */
table {
    background: #1A1208 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(200, 135, 58, 0.12) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    width: 100% !important;
    font-size: 0.88rem !important;
}
thead tr {
    background: rgba(200, 135, 58, 0.08) !important;
}
thead th {
    color: #C8873A !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.07em !important;
    border-bottom: 1px solid rgba(200, 135, 58, 0.14) !important;
}
tbody td {
    color: #A08060 !important;
    padding: 0.65rem 1rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
tbody tr:last-child td { border-bottom: none !important; }
tbody tr:hover td {
    background: rgba(200, 135, 58, 0.05) !important;
    color: #EDE0CC !important;
}

/* ── LINE CHART ── */
[data-testid="stArrowVegaLiteChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #1A1208 !important;
    border: 1px solid rgba(200, 135, 58, 0.10) !important;
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── LSTM MODEL DEFINITION ────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=256,
                 hidden_dim=128, num_layers=1, num_classes=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)

# ── LOAD MODELS ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device('cpu')

    # load LSTM
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load('lstm_model.pth',
                         map_location=device))
    lstm.eval()

    # load BERT
    bert = BertForSequenceClassification.from_pretrained('bert_model')
    tokenizer = BertTokenizer.from_pretrained('bert_model')
    bert.eval()

    return lstm, bert, tokenizer, device

lstm_model, bert_model, tokenizer, device = load_models()

# ── SIMPLE TOKENIZER FOR LSTM ────────────────────────────────
def simple_tokenize(text, max_len=200):
    words = text.lower().split()
    vocab = {w: i+1 for i, w in enumerate(words)}
    ids = [vocab.get(w, 0) for w in words[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids])

# ── PREDICT FUNCTIONS ─────────────────────────────────────────
def predict_lstm(text):
    tokens = simple_tokenize(text)
    with torch.no_grad():
        out = lstm_model(tokens)
        probs = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()
    return pred, probs[1].item()

def predict_bert(text):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, max_length=512,
                       padding=True)
    with torch.no_grad():
        out = bert_model(**inputs)
        probs = torch.softmax(out.logits, dim=1)[0]
        pred = out.logits.argmax(dim=1).item()
    return pred, probs[1].item()

# ── UI ────────────────────────────────────────────────────────
st.title("Movie Review Sentiment Analyzer")
st.markdown("### LSTM vs BERT — Which model gets it right?")
st.markdown("---")

review = st.text_area("✍️ Enter a movie review:",
                       placeholder="e.g. This movie was absolutely amazing!",
                       height=150)

if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        col1, col2 = st.columns(2)

        # LSTM result
        with col1:
            st.markdown("### 🔴 LSTM")
            lstm_pred, lstm_conf = predict_lstm(review)
            label = "POSITIVE 😊" if lstm_pred == 1 else "NEGATIVE 😞"
            color = "green" if lstm_pred == 1 else "red"
            st.markdown(f"**Prediction:** :{color}[{label}]")
            st.metric("Confidence", f"{lstm_conf*100:.1f}%")

        # BERT result
        with col2:
            st.markdown("### 🟢 BERT")
            bert_pred, bert_conf = predict_bert(review)
            label = "POSITIVE 😊" if bert_pred == 1 else "NEGATIVE 😞"
            color = "green" if bert_pred == 1 else "red"
            st.markdown(f"**Prediction:** :{color}[{label}]")
            st.metric("Confidence", f"{bert_conf*100:.1f}%")

        st.markdown("---")
        if lstm_pred == bert_pred:
            st.success("✅ Both models agree!")
        else:
            st.error("⚠️ Models disagree — BERT is more reliable (92% accuracy)")