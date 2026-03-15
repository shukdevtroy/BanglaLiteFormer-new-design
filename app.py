import os
import time  # ← for latency / inference-time measurement
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import gradio as gr


# ============================================================
# CUSTOM LAYERS
# ============================================================

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape): super().build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                        "ff_dim": self.ff_dim, "rate": self.rate})
        return config


@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def build(self, input_shape): super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size,
                        "embed_dim": self.embed_dim})
        return config


# ============================================================
# LOAD TOKENIZER & MODEL
# ============================================================

with open("improved_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 80

model = tf.keras.models.load_model(
    "improved_bangla_sentiment.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    },
    compile=False
)


# ============================================================
# PREDICTION
# ============================================================

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    return padded


EMPTY_RESULT = """
<div style="
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:14px;min-height:160px;padding:40px;
  border:1.5px dashed rgba(255,255,255,0.08);border-radius:16px;
  text-align:center;
">
  <div style="position:relative;width:60px;height:60px;display:flex;align-items:center;justify-content:center;">
    <div style="position:absolute;inset:0;border:1.5px solid rgba(139,92,246,0.4);border-radius:50%;animation:bsa-pulse 2.5s ease-out infinite;"></div>
    <div style="position:absolute;inset:0;border:1.5px solid rgba(139,92,246,0.2);border-radius:50%;animation:bsa-pulse 2.5s 0.9s ease-out infinite;"></div>
    <span style="font-size:28px;position:relative;z-index:1;opacity:0.5;">💬</span>
  </div>
  <p style="font-family:'Hind Siliguri',sans-serif;font-size:16px;font-weight:600;color:#a78bfa;margin:0;">কিছু একটা লিখুন বিশ্লেষণ করতে...</p>
  <p style="font-family:monospace;font-size:11px;letter-spacing:1.5px;color:white;margin:0;">ENTER BANGLA TEXT ABOVE AND CLICK ANALYZE</p>
</div>
"""

EXAMPLES = [
    "এই পণ্যটি অসাধারণ! আমি খুব সন্তুষ্ট।",
    "ডেলিভারি অনেক দেরিতে এসেছে, খুব খারাপ অভিজ্ঞতা।",
    "কোয়ালিটি মোটামুটি ভালো, দাম একটু বেশি।",
    "আমি একদমই খুশি নই, পণ্যটি প্রত্যাশা অনুযায়ী নয়।",
    "চমৎকার সার্ভিস এবং দ্রুত ডেলিভারি।",
]


def predict_sentiment(text):
    if not text or not text.strip():
        return EMPTY_RESULT

    # ── Timing: total wall-clock time (latency) ──────────────
    t_start = time.perf_counter()

    # ── Preprocessing ────────────────────────────────────────
    t_pre0 = time.perf_counter()
    processed = preprocess_text(text)
    t_pre1 = time.perf_counter()
    preprocess_ms = (t_pre1 - t_pre0) * 1000

    # ── Inference (model.predict) ─────────────────────────────
    t_inf0 = time.perf_counter()
    prediction = model.predict(processed, verbose=0)[0]
    t_inf1 = time.perf_counter()
    inference_ms = (t_inf1 - t_inf0) * 1000

    # ── Total latency ─────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Decode prediction ─────────────────────────────────────
    if np.ndim(prediction) == 0 or len(np.atleast_1d(prediction)) == 1:
        score = float(prediction)
        is_positive = score >= 0.5
        confidence = score if is_positive else 1 - score
    else:
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        is_positive = class_id == 1

    pct = int(confidence * 100)
    word_count = len(text.split())
    char_count = len(text.strip())

    # ── Uncertainty flag (confidence near 0.5) ───────────────
    is_uncertain = confidence < 0.65

    radius = 52
    circ = 2 * 3.14159 * radius
    offset = circ * (1 - confidence)

    if is_positive:
        emoji     = "😊"
        label_en  = "Positive"
        label_bn  = "ইতিবাচক"
        color     = "#00e5a0"
        glow      = "rgba(0,229,160,0.28)"
        card_bg   = "rgba(0,229,160,0.06)"
        card_bdr  = "rgba(0,229,160,0.22)"
        bar_grad  = "linear-gradient(90deg,#00b87844,#00e5a0)"
        intensity = "উচ্চ" if pct >= 80 else ("মাঝারি" if pct >= 60 else "হালকা")
    else:
        emoji     = "😠"
        label_en  = "Negative"
        label_bn  = "নেতিবাচক"
        color     = "#f87171"
        glow      = "rgba(248,113,113,0.28)"
        card_bg   = "rgba(248,113,113,0.06)"
        card_bdr  = "rgba(248,113,113,0.22)"
        bar_grad  = "linear-gradient(90deg,#f8717144,#f87171)"
        intensity = "উচ্চ" if pct >= 80 else ("মাঝারি" if pct >= 60 else "হালকা")

    # ── Uncertainty banner (shown only when confidence < 65%) ─
    uncertain_banner = ""
    if is_uncertain:
        uncertain_banner = f"""
        <div style="
          margin-bottom:16px;padding:10px 16px;border-radius:10px;
          background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.28);
          display:flex;align-items:center;gap:10px;
        ">
          <span style="font-size:18px;">⚠️</span>
          <span style="font-family:monospace;font-size:11px;letter-spacing:1.5px;color:#fbbf24;">
            LOW CONFIDENCE ({pct}%) — PREDICTION MAY BE UNRELIABLE
          </span>
        </div>
        """

    # ── Performance stats row ──────────────────────────────────
    stats_row = f"""
    <div style="margin-top:18px;">
      <div style="font-family:monospace;font-size:11px;letter-spacing:2.5px;text-transform:uppercase;
        color:#22d3ee;margin-bottom:8px;">⏱ PERFORMANCE METRICS</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <div style="display:flex;flex-direction:column;gap:3px;padding:10px 16px;border-radius:12px;
          border:1px solid rgba(34,211,238,0.18);background:rgba(34,211,238,0.05);flex:1;min-width:90px;">
          <span style="font-family:monospace;font-size:9px;letter-spacing:2px;color:#22d3ee;">TOTAL LATENCY</span>
          <span style="font-family:monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{latency_ms:.1f}<span style="font-size:11px;color:#94a3b8;"> ms</span></span>
          <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.45);">wall-clock end-to-end</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:3px;padding:10px 16px;border-radius:12px;
          border:1px solid rgba(167,139,250,0.18);background:rgba(167,139,250,0.05);flex:1;min-width:90px;">
          <span style="font-family:monospace;font-size:9px;letter-spacing:2px;color:#a78bfa;">INFERENCE TIME</span>
          <span style="font-family:monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{inference_ms:.1f}<span style="font-size:11px;color:#94a3b8;"> ms</span></span>
          <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.45);">model.predict only</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:3px;padding:10px 16px;border-radius:12px;
          border:1px solid rgba(0,229,160,0.18);background:rgba(0,229,160,0.05);flex:1;min-width:90px;">
          <span style="font-family:monospace;font-size:9px;letter-spacing:2px;color:#00e5a0;">PREPROCESS</span>
          <span style="font-family:monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{preprocess_ms:.1f}<span style="font-size:11px;color:#94a3b8;"> ms</span></span>
          <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.45);">tokenize + pad</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:3px;padding:10px 16px;border-radius:12px;
          border:1px solid rgba(251,191,36,0.18);background:rgba(251,191,36,0.05);flex:1;min-width:90px;">
          <span style="font-family:monospace;font-size:9px;letter-spacing:2px;color:#fbbf24;">CHARS / TOKENS</span>
          <span style="font-family:monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{char_count}<span style="font-size:11px;color:#94a3b8;"> / {MAX_LEN}</span></span>
          <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.45);">input / max seq len</span>
        </div>
      </div>
    </div>
    """

    return f"""
<div style="
  background:{card_bg};border:1.5px solid {card_bdr};border-radius:20px;
  padding:26px;position:relative;overflow:hidden;
  animation:bsa-fadein 0.45s cubic-bezier(0.23,1,0.32,1) both;
  box-shadow:0 16px 50px rgba(0,0,0,0.35), 0 0 60px {glow};
">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,{color}99,transparent);"></div>
  {uncertain_banner}
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px;flex-wrap:wrap;">
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="
        width:70px;height:70px;flex-shrink:0;display:flex;
        align-items:center;justify-content:center;border-radius:50%;
        border:2px solid {color};
        box-shadow:0 0 20px {glow},inset 0 0 16px {glow};
        background:rgba(0,0,0,0.25);
      ">
        <span style="font-size:36px;line-height:1;">{emoji}</span>
      </div>
      <div>
        <div style="
          font-family:monospace;font-size:9px;letter-spacing:2.5px;text-transform:uppercase;
          color:{color};border:1px solid {card_bdr};padding:3px 10px;border-radius:100px;
          display:inline-block;margin-bottom:6px;background:rgba(0,0,0,0.2);
        ">{intensity} · {pct}%</div>
        <div style="
          font-family:'Outfit','Hind Siliguri',sans-serif;
          font-size:36px;font-weight:900;letter-spacing:-1.5px;
          color:{color};line-height:1;margin-bottom:4px;
          text-shadow:0 0 28px {glow};
        ">{label_en}</div>
        <div style="font-family:'Hind Siliguri',sans-serif;font-size:15px;color:rgba(148,163,184,0.75);font-weight:500;">{label_bn}</div>
      </div>
    </div>
    <svg viewBox="0 0 120 120" width="110" height="110" style="display:block;flex-shrink:0;">
      <circle cx="60" cy="60" r="{radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="9"/>
      <circle cx="60" cy="60" r="{radius}" fill="none"
        stroke="{color}" stroke-width="9" stroke-linecap="round"
        stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"
        transform="rotate(-90 60 60)"
        style="filter:drop-shadow(0 0 5px {color});"/>
      <text x="60" y="56" text-anchor="middle"
        style="font-family:monospace;font-size:21px;font-weight:700;fill:{color};">{pct}</text>
      <text x="60" y="73" text-anchor="middle"
        style="font-family:monospace;font-size:10px;fill:rgba(148,163,184,0.45);">%</text>
    </svg>
  </div>
  <div style="font-family:monospace;font-size:11px;letter-spacing:2.5px;text-transform:uppercase;
    color:#c4b5fd;margin-bottom:8px;">CONFIDENCE SCORE</div>
  <div style="background:rgba(255,255,255,0.05);border-radius:100px;height:8px;
    position:relative;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
    <div style="width:{pct}%;height:100%;border-radius:100px;background:{bar_grad};
      box-shadow:0 0 10px {glow};position:relative;">
      <div style="position:absolute;right:-1px;top:50%;transform:translateY(-50%);
        width:13px;height:13px;border-radius:50%;background:{color};box-shadow:0 0 9px {color};"></div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:5px;padding:0 2px;">
    <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.22);">0</span>
    <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.22);">25</span>
    <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.22);">50</span>
    <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.22);">75</span>
    <span style="font-family:monospace;font-size:9px;color:rgba(148,163,184,0.22);">100</span>
  </div>
  <div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap;">
    <div style="display:flex;align-items:center;gap:6px;padding:7px 14px;border-radius:100px;
      border:1px solid {card_bdr};background:rgba(0,0,0,0.22);flex:1;min-width:70px;">
      <span style="font-size:13px;">🔤</span>
      <span style="font-family:monospace;font-size:12px;font-weight:700;color:#e2e8f0;">{word_count}</span>
      <span style="font-family:monospace;font-size:12px;color:#22D3EE;letter-spacing:1px;">WORDS</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;padding:7px 14px;border-radius:100px;
      border:1px solid {card_bdr};background:rgba(0,0,0,0.22);flex:1;min-width:70px;">
      <span style="font-size:13px;">🎯</span>
      <span style="font-family:monospace;font-size:12px;font-weight:700;color:#e2e8f0;">{pct}%</span>
      <span style="font-family:monospace;font-size:12px;color:#22D3EE;letter-spacing:1px;">CONF</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;padding:7px 14px;border-radius:100px;
      border:1px solid {card_bdr};background:rgba(0,0,0,0.22);flex:1;min-width:90px;">
      <span style="font-size:13px;">⚡</span>
      <span style="font-family:monospace;font-size:12px;font-weight:700;color:#e2e8f0;">Transformer</span>
    </div>
  </div>
  {stats_row}
</div>
"""


# ============================================================
# CSS
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
@keyframes bsa-fadein { from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);} }
@keyframes bsa-pulse  { 0%{transform:scale(1);opacity:0.8;}100%{transform:scale(2.4);opacity:0;} }
@keyframes bsa-shim   { 0%{background-position:200% 0;}100%{background-position:-200% 0;} }
@keyframes bsa-blink  { 0%,100%{opacity:1;}50%{opacity:0.15;} }
@keyframes bsa-aurora { 0%{opacity:0.6;}50%{opacity:1;}100%{opacity:0.7;} }
body { background: #070a14 !important; }
.gradio-container {
  background: #070a14 !important;
  font-family: 'Outfit', 'Hind Siliguri', sans-serif !important;
  min-height: 100vh !important;
}
.gradio-container::before {
  content:'';
  position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 70% 50% at 15% 25%, rgba(124,58,237,0.17) 0%, transparent 55%),
    radial-gradient(ellipse 55% 55% at 85% 75%, rgba(34,211,238,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 45% 35% at 50% 50%, rgba(0,229,160,0.06) 0%, transparent 55%);
  animation: bsa-aurora 10s ease-in-out infinite alternate;
}
.gradio-container > .main,
.gradio-container > .main > .wrap,
.contain {
  background: transparent !important;
  max-width: 100% !important;
}
#bsa-wrap {
  max-width: 800px !important;
  margin: 0 auto !important;
  padding: 0 24px 80px !important;
  position: relative !important;
  z-index: 1 !important;
  background: transparent !important;
}
#bsa-wrap .block,
#bsa-wrap .form,
#bsa-wrap .gap,
#bsa-card .block,
#bsa-card .form,
#bsa-card .gap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  gap: 0 !important;
}
#bsa-card {
  background: rgba(13,16,35,0.88) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 24px !important;
  padding: 36px 36px 28px !important;
  position: relative !important;
  overflow: hidden !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  box-shadow: 0 30px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04) !important;
}
#bsa-card::before {
  content:'';
  position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent 0%,rgba(124,58,237,0.8) 30%,rgba(34,211,238,1) 50%,rgba(124,58,237,0.8) 70%,transparent 100%);
  background-size:200% 100%;
  animation:bsa-shim 4s linear infinite;
}
#bsa-info {
  background: rgba(13,16,35,0.88) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 24px !important;
  padding: 32px 36px !important;
  margin-top: 20px !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.4) !important;
}
/* ── TEXTAREA ── */
.gradio-container textarea,
#bsa-card textarea,
#bsa-wrap textarea,
textarea {
  background-color: rgba(15,18,40,0.8) !important;
  background: rgba(15,18,40,0.8) !important;
  border: 1.5px solid rgba(255,255,255,0.1) !important;
  border-radius: 14px !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 16px !important;
  line-height: 1.8 !important;
  padding: 16px 18px !important;
  resize: none !important;
  caret-color: #22d3ee !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
  box-shadow: none !important;
}
.gradio-container textarea:focus,
textarea:focus {
  background-color: rgba(20,24,55,0.9) !important;
  background: rgba(20,24,55,0.9) !important;
  border-color: rgba(124,58,237,0.6) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.14) !important;
  outline: none !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
}
textarea::placeholder { color: navajowhite !important; -webkit-text-fill-color: navajowhite !important; }
label > span, .gradio-container label > span { display: none !important; }
/* ── PRIMARY BUTTON ── */
.gradio-container button.primary,
button.primary,
button[variant="primary"] {
  background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
  border: 1px solid rgba(139,92,246,0.45) !important;
  border-radius: 13px !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  padding: 14px 24px !important;
  cursor: pointer !important;
  transition: all 0.25s !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.38) !important;
  text-shadow: none !important;
}
button.primary:hover, button[variant="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 35px rgba(124,58,237,0.58) !important;
  background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
}
/* ── SECONDARY BUTTON ── */
.gradio-container button.secondary,
button.secondary,
button[variant="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1.5px solid rgba(255,255,255,0.1) !important;
  border-radius: 13px !important;
  color: rgba(148,163,184,0.85) !important;
  -webkit-text-fill-color: rgba(148,163,184,0.85) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 14px !important;
  padding: 14px 20px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  box-shadow: none !important;
  text-shadow: none !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.18) !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
  transform: translateY(-1px) !important;
}
/* ── EXAMPLE BUTTONS ── */
.ex-btn button,
.gradio-container .ex-btn button {
  background: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  color: navajowhite !important;
  -webkit-text-fill-color: navajowhite !important;
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 10px 16px !important;
  text-align: left !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  box-shadow: none !important;
  text-shadow: none !important;
  line-height: 1.6 !important;
  white-space: normal !important;
  height: auto !important;
  min-height: unset !important;
}
.ex-btn button:hover,
.gradio-container .ex-btn button:hover {
  background: rgba(124,58,237,0.12) !important;
  border-color: rgba(124,58,237,0.35) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  transform: translateX(3px) !important;
  box-shadow: none !important;
}
/* ── EXAMPLE BUTTON ROW ── */
.ex-btn-row {
  gap: 8px !important;
  flex-wrap: wrap !important;
  margin: 0 !important;
}
.ex-btn-row .block,
.ex-btn-row .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-width: 0 !important;
}
/* ── BUTTON ROW ── */
.bsa-btnrow { gap: 12px !important; margin: 18px 0 28px !important; }
.bsa-btnrow .block, .bsa-btnrow .form {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
/* ── RESULT ── */
.bsa-result .block, .bsa-result .prose, .bsa-result .wrap, .bsa-result > div {
  background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important;
}
footer, .svelte-footer, .gr-footer { display: none !important; }

/* ── SCROLL TOGGLE BUTTON ── */
#bsa-scroll-btn {
  position: fixed !important;
  bottom: 32px !important;
  right: 32px !important;
  z-index: 9999 !important;
  width: 52px !important;
  height: 52px !important;
  border-radius: 50% !important;
  background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important;
  border: 1.5px solid rgba(139,92,246,0.5) !important;
  box-shadow: 0 8px 28px rgba(124,58,237,0.45), 0 0 0 1px rgba(255,255,255,0.06) !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.3s ease !important;
  opacity: 0.92 !important;
  padding: 0 !important;
  min-width: unset !important;
  font-size: 22px !important;
  line-height: 1 !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  text-shadow: none !important;
}
#bsa-scroll-btn:hover {
  transform: translateY(-3px) scale(1.08) !important;
  box-shadow: 0 14px 38px rgba(124,58,237,0.65), 0 0 0 1px rgba(255,255,255,0.1) !important;
  opacity: 1 !important;
}
#bsa-scroll-btn:active {
  transform: scale(0.95) !important;
}
"""


# ============================================================
# HTML BLOCKS
# ============================================================

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
@keyframes bsa-blink2 { 0%,100%{opacity:1;}50%{opacity:0.15;} }
@keyframes bsa-float2 { 0%,100%{transform:translateX(-50%) translateY(0);}50%{transform:translateX(-50%) translateY(-18px);} }
</style>
<div style="text-align:center;padding:64px 20px 50px;position:relative;overflow:visible;">
  <div style="position:absolute;width:560px;height:320px;
    background:radial-gradient(circle,rgba(124,58,237,0.2) 0%,transparent 65%);
    filter:blur(100px);border-radius:50%;top:-60px;left:50%;transform:translateX(-50%);
    pointer-events:none;z-index:0;animation:bsa-float2 9s ease-in-out infinite;"></div>
  <div style="position:absolute;width:260px;height:260px;
    background:radial-gradient(circle,rgba(34,211,238,0.13) 0%,transparent 65%);
    filter:blur(80px);border-radius:50%;top:0;right:0;pointer-events:none;z-index:0;"></div>
  <div style="display:inline-flex;align-items:center;gap:8px;
    font-family:monospace;font-size:11px;letter-spacing:3.5px;text-transform:uppercase;
    color:#22d3ee;border:2px solid rgba(34,211,238,0.22);
    padding:7px 22px;border-radius:100px;background:rgba(34,211,238,0.05);
    margin-bottom:26px;position:relative;z-index:1;">
    <span style="width:6px;height:6px;border-radius:50%;background:#22d3ee;
      box-shadow:0 0 7px #22d3ee;animation:bsa-blink2 2s ease-in-out infinite;display:inline-block;"></span>
    transformer · nlp · bengali
  </div>
  <div style="font-family:'Outfit',sans-serif;font-size:clamp(34px,6vw,58px);
    font-weight:900;line-height:1.05;letter-spacing:-2px;margin-bottom:16px;position:relative;z-index:1;">
    <span style="display:block;color:#c4b5fd;">Bangla Sentiment</span>
    <span style="display:block;color:#22d3ee;">Analyzer</span>
  </div>
  <p style="font-family:'Hind Siliguri',sans-serif;font-size:17px;
    color:white;margin-bottom:26px;position:relative;z-index:1;">
    বাংলা রিভিউ বা মন্তব্য লিখুন — তাৎক্ষণিক বিশ্লেষণ পান
  </p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;position:relative;z-index:1;">
    <span style="font-family:monospace;font-size:11px;padding:6px 16px;border-radius:100px;
      border:1px solid rgba(255,255,255,0.09);color:#00ffdfa6;background:rgba(255,255,255,0.03);">⚡ TensorFlow</span>
    <span style="font-family:monospace;font-size:11px;padding:6px 16px;border-radius:100px;
      border:1px solid rgba(255,255,255,0.09);color:#C4B5FD;background:rgba(255,255,255,0.03);">🔤 Transformer</span>
    <span style="font-family:monospace;font-size:11px;padding:6px 16px;border-radius:100px;
      border:1px solid rgba(255,255,255,0.09);color:#00ffdfa6;background:rgba(255,255,255,0.03);
      display:inline-flex;align-items:center;gap:6px;">
      <img src="https://flagcdn.com/16x12/bd.png" width="16" height="12"
        style="border-radius:2px;vertical-align:middle;" alt="BD"> Bengali NLP</span>
    <span style="font-family:monospace;font-size:11px;padding:6px 16px;border-radius:100px;
      border:1px solid rgba(255,255,255,0.09);color:#C4B5FD;background:rgba(255,255,255,0.03);">🎯 Classification</span>
  </div>
</div>
"""

SEC_INPUT  = """<div style="display:flex;align-items:center;gap:10px;font-family:monospace;font-size:13px;letter-spacing:3.5px;text-transform:uppercase;color:white;margin-bottom:12px;"><span style="width:22px;height:1.5px;border-radius:2px;display:inline-block;flex-shrink:0;background:linear-gradient(90deg,#7c3aed,#22d3ee);"></span>Input</div>"""
SEC_RESULT = """<div style="display:flex;align-items:center;gap:10px;font-family:monospace;font-size:13px;letter-spacing:3.5px;text-transform:uppercase;color:white;margin-top:8px;margin-bottom:12px;"><span style="width:22px;height:1.5px;border-radius:2px;display:inline-block;flex-shrink:0;background:linear-gradient(90deg,#7c3aed,#22d3ee);"></span>Result</div>"""
DIVIDER    = """<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);margin:26px 0 18px;"></div>"""
EX_LABEL   = """<div style="font-family:monospace;font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#A78BFA;margin-bottom:10px;">✦ Example inputs — click to try</div>"""
FOOTER     = """<div style="text-align:center;padding:32px 0 10px;"><p style="font-family:monospace;font-size:11px;letter-spacing:2px;color:#A78BFA;">Powered by Custom Transformer <span style="display:inline-block;width:3px;height:3px;background:#7c3aed;border-radius:50%;margin:0 10px;vertical-align:middle;opacity:0.5;"></span> Built with TensorFlow &amp; Gradio <span style="display:inline-block;width:3px;height:3px;background:#7c3aed;border-radius:50%;margin:0 10px;vertical-align:middle;opacity:0.5;"></span> Bengali NLP</p></div>"""

# ── Scroll Toggle Button (injected once, persists on page) ────────────────────
SCROLL_TOGGLE_BTN = """
<button id="bsa-scroll-btn" title="Scroll to bottom" aria-label="Scroll to bottom">&#8593;</button>
<script>
(function () {
  // Wait until the button is in the DOM
  var btn = document.getElementById('bsa-scroll-btn');
  if (!btn) return;

  // true  → currently showing ↑ (arrow up)  → click scrolls to TOP
  // false → currently showing ↓ (arrow down) → click scrolls to BOTTOM
  var atBottom = false;

  function updateBtn() {
    if (atBottom) {
      btn.innerHTML = '&#8595;'; // ↓
      btn.title = 'Scroll to bottom';
      btn.setAttribute('aria-label', 'Scroll to bottom');
    } else {
      btn.innerHTML = '&#8593;'; // ↑
      btn.title = 'Scroll to top';
      btn.setAttribute('aria-label', 'Scroll to top');
    }
  }

  btn.addEventListener('click', function () {
    if (!atBottom) {
      // Arrow UP was showing → scroll to top
      window.scrollTo({ top: 0, behavior: 'smooth' });
      atBottom = true;
    } else {
      // Arrow DOWN was showing → scroll to bottom
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      atBottom = false;
    }
    updateBtn();
  });

  // Keep button in sync when user scrolls manually
  window.addEventListener('scroll', function () {
    var scrolledToTop    = window.scrollY < 60;
    var scrolledToBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 60);

    if (scrolledToTop && atBottom) {
      atBottom = false;
      updateBtn();
    } else if (scrolledToBottom && !atBottom) {
      atBottom = true;
      updateBtn();
    }
  }, { passive: true });

  updateBtn();
})();
</script>
"""

# ── Static information panel (always visible below the main card) ─────────────
INFO_PANEL = """
<div id="bsa-info" style="
  background:rgba(13,16,35,0.88);
  border:1px solid rgba(255,255,255,0.09);
  border-radius:24px;
  padding:32px 36px;
  position:relative;
  overflow:hidden;
  backdrop-filter:blur(20px);
  box-shadow:0 20px 60px rgba(0,0,0,0.4);
">
  <!-- top shimmer line -->
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(124,58,237,0.7) 40%,rgba(34,211,238,0.9) 60%,transparent);"></div>
  <!-- Section header -->
  <div style="font-family:monospace;font-size:12px;letter-spacing:3.5px;text-transform:uppercase;
    color:#22d3ee;margin-bottom:24px;display:flex;align-items:center;gap:10px;">
    <span style="width:22px;height:1.5px;border-radius:2px;display:inline-block;
      background:linear-gradient(90deg,#7c3aed,#22d3ee);"></span>
    Model &amp; Performance Reference
  </div>
  <!-- ── 1. Computational Cost ── -->
  <div style="margin-bottom:24px;padding:20px;border-radius:14px;
    background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.18);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-size:20px;">💻</span>
      <span style="font-family:monospace;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;
        color:#a78bfa;font-weight:700;">1 · Computational Cost</span>
    </div>
    <div style="font-family:'Hind Siliguri',sans-serif;font-size:14px;
      color:rgba(226,232,240,0.82);line-height:1.85;">
      This model is a <b style="color:#c4b5fd;">single-head Transformer</b> trained for binary text classification.
      Its computational footprint is intentionally lightweight:
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
      gap:10px;margin-top:14px;">
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#a78bfa;margin-bottom:4px;">PARAMETERS</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">~1–5 M</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">typical small Transformer</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#a78bfa;margin-bottom:4px;">MEMORY (RAM)</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">~50–200 MB</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">model weights + TF runtime</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#a78bfa;margin-bottom:4px;">HARDWARE</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">CPU only</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">GPU not required at inference</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#a78bfa;margin-bottom:4px;">MAX SEQ LEN</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">80 tokens</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">O(n²) attention applies</div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px 14px;border-radius:8px;
      background:rgba(124,58,237,0.08);border-left:3px solid rgba(124,58,237,0.5);
      font-family:monospace;font-size:11px;color:rgba(196,181,253,0.75);line-height:1.7;">
      ℹ️ &nbsp;Self-attention complexity scales as <b style="color: white !important;">O(n²·d)</b> where n = sequence length and d = embedding dimension.
      At n=80 this is negligible — the dominant cost is the first inference call due to TF graph compilation
      (TensorFlow's XLA warm-up). Subsequent calls are significantly faster.
    </div>
  </div>
  <!-- ── 2. Latency ── -->
  <div style="margin-bottom:24px;padding:20px;border-radius:14px;
    background:rgba(34,211,238,0.06);border:1px solid rgba(34,211,238,0.16);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-size:20px;">🌐</span>
      <span style="font-family:monospace;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;
        color:#22d3ee;font-weight:700;">2 · Latency</span>
    </div>
    <div style="font-family:'Hind Siliguri',sans-serif;font-size:14px;
      color:rgba(226,232,240,0.82);line-height:1.85;margin-bottom:14px;">
      <b style="color:#22d3ee;">Latency</b> is the total wall-clock time from receiving user input to returning the
      result — it includes tokenization, padding, model forward-pass, and HTML rendering.
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;">
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#22d3ee;margin-bottom:4px;">1st REQUEST (cold)</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">500 ms – 3 s</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">TF graph compilation</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#22d3ee;margin-bottom:4px;">WARM REQUESTS</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">20 – 80 ms</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">CPU inference, typical</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#22d3ee;margin-bottom:4px;">WITH GPU</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">&lt; 10 ms</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">if CUDA is available</div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px 14px;border-radius:8px;
      background:rgba(34,211,238,0.07);border-left:3px solid rgba(34,211,238,0.4);
      font-family:monospace;font-size:11px;color:rgba(167,243,254,0.7);line-height:1.7;">
      ℹ️ &nbsp;Network latency (Gradio UI ↔ Python backend) adds ~5–15 ms on localhost.
      On Hugging Face Spaces or remote servers, add 50–150 ms round-trip depending on geography.
    </div>
  </div>
  <!-- ── 3. Inference Time ── -->
  <div style="margin-bottom:24px;padding:20px;border-radius:14px;
    background:rgba(0,229,160,0.05);border:1px solid rgba(0,229,160,0.16);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-size:20px;">⚡</span>
      <span style="font-family:monospace;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;
        color:#00e5a0;font-weight:700;">3 · Inference Time</span>
    </div>
    <div style="font-family:'Hind Siliguri',sans-serif;font-size:14px;
      color:rgba(226,232,240,0.82);line-height:1.85;margin-bottom:14px;">
      <b style="color:#00e5a0;">Inference time</b> measures only the <code style="color:#00e5a0;
      background:rgba(0,0,0,0.3);padding:1px 6px;border-radius:4px;">model.predict()</code> call —
      the neural network forward pass itself. This is the pure compute cost and excludes
      tokenization, Gradio overhead, and network round-trips. Each run shows the <em>live</em>
      measured value above in the result card.
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;">
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#00e5a0;margin-bottom:4px;">BATCH SIZE</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">1 sample</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">single-text inference</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#00e5a0;margin-bottom:4px;">TYPICAL RANGE</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">5 – 50 ms</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">warm, CPU, seq=80</div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.25);
        border:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:monospace;font-size:10px;letter-spacing:1.5px;color:#00e5a0;margin-bottom:4px;">VARIABILITY</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;color:#e2e8f0;">±10–30%</div>
        <div style="font-family:monospace;font-size:10px;color:rgba(148,163,184,0.45);">OS scheduling, cache state</div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px 14px;border-radius:8px;
      background:rgba(0,229,160,0.06);border-left:3px solid rgba(0,229,160,0.4);
      font-family:monospace;font-size:11px;color:rgba(110,231,183,0.75);line-height:1.7;">
      ℹ️ &nbsp;First-call inference is slower because TensorFlow traces and compiles the computation graph
      (XLA/JIT). After the first call the compiled graph is cached in memory, making all
      subsequent inferences significantly faster and more consistent.
    </div>
  </div>
  <!-- ── 4. When Prediction May Vary ── -->
  <div style="padding:20px;border-radius:14px;
    background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.18);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-size:20px;">⚠️</span>
      <span style="font-family:monospace;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;
        color:#fbbf24;font-weight:700;">4 · When Predictions May Vary or Be Unreliable</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">🔀</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">MIXED SENTIMENT</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            Text containing both positive and negative signals (e.g., "পণ্যটি সুন্দর কিন্তু দাম বেশি") confuses the
            binary classifier. Confidence will hover near 50 % and may flip with small edits.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">🔤</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">OOV / RARE VOCABULARY</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            Words not seen during training are mapped to an unknown token <code style="color:#fbbf24;
            background:rgba(0,0,0,0.3);padding:1px 5px;border-radius:3px;">&lt;UNK&gt;</code>.
            Heavy use of slang, dialect, transliteration, or technical jargon will degrade accuracy.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">🌐</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">CODE-SWITCHING (BANGLISH)</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            Mixing Bangla script with English words or Roman-script Bangla causes many tokens to be
            OOV, making predictions unreliable. Use pure Unicode Bangla for best results.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">📏</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">VERY SHORT OR VERY LONG TEXT</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            Single-word inputs lack context. Texts exceeding 80 tokens are silently truncated —
            sentiment expressed only in the trailing portion will be ignored entirely.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">😏</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">SARCASM &amp; IRONY</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            The model has no pragmatic understanding. Ironic phrases like "হ্যাঁ, অসাধারণ সার্ভিস!" (sarcastic)
            are likely classified as Positive because the surface tokens are positive.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">🎭</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">DOMAIN SHIFT</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            If the training data consisted mainly of product reviews, the model may perform poorly
            on political commentary, news text, or social media posts with different linguistic patterns.
          </div>
        </div>
      </div>
      <div style="padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);
        border-left:3px solid rgba(251,191,36,0.5);display:flex;gap:12px;align-items:flex-start;">
        <span style="font-size:16px;flex-shrink:0;">🎲</span>
        <div>
          <div style="font-family:monospace;font-size:11px;letter-spacing:1px;color:#fbbf24;margin-bottom:4px;">DROPOUT AT INFERENCE (if training=True)</div>
          <div style="font-family:'Hind Siliguri',sans-serif;font-size:13px;color:rgba(226,232,240,0.75);line-height:1.7;">
            Dropout layers are disabled during inference (<code style="color:#fbbf24;
            background:rgba(0,0,0,0.3);padding:1px 5px;border-radius:3px;">training=False</code>),
            so results are deterministic. If you call the model with <code style="color:#fbbf24;
            background:rgba(0,0,0,0.3);padding:1px 5px;border-radius:3px;">training=True</code>,
            outputs will differ randomly on each run.
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(css=css, title="Bangla Sentiment Analyzer") as demo:

    with gr.Column(elem_id="bsa-wrap"):

        gr.HTML(HERO)

        with gr.Column(elem_id="bsa-card"):

            gr.HTML(SEC_INPUT)

            text_input = gr.Textbox(
                lines=4,
                placeholder="এখানে আপনার বাংলা রিভিউ বা মন্তব্য লিখুন...",
                label="", show_label=False,
            )

            with gr.Row(elem_classes="bsa-btnrow"):
                submit_btn = gr.Button("🔍  Analyze Sentiment", variant="primary", scale=3)
                clear_btn  = gr.Button("✕  Clear", variant="secondary", scale=1)

            gr.HTML(SEC_RESULT)

            with gr.Column(elem_classes="bsa-result"):
                output_html = gr.HTML(value=EMPTY_RESULT)

            gr.HTML(DIVIDER)
            gr.HTML(EX_LABEL)

            # ── EXAMPLE BUTTONS ──
            with gr.Row(elem_classes="ex-btn-row"):
                ex1 = gr.Button(EXAMPLES[0], elem_classes="ex-btn")
                ex2 = gr.Button(EXAMPLES[1], elem_classes="ex-btn")
            with gr.Row(elem_classes="ex-btn-row"):
                ex3 = gr.Button(EXAMPLES[2], elem_classes="ex-btn")
                ex4 = gr.Button(EXAMPLES[3], elem_classes="ex-btn")
            with gr.Row(elem_classes="ex-btn-row"):
                ex5 = gr.Button(EXAMPLES[4], elem_classes="ex-btn")

        # ── Static info panel (always visible below the main card) ──
        gr.HTML(INFO_PANEL)

        gr.HTML(FOOTER)

    # ── Floating scroll toggle button (fixed, always on screen) ──
    gr.HTML(SCROLL_TOGGLE_BTN)

    # ── Main events ──
    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    text_input.submit(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    clear_btn.click(fn=lambda: ("", EMPTY_RESULT), inputs=None, outputs=[text_input, output_html])

    # ── Example button events ──
    ex1.click(fn=lambda: EXAMPLES[0], inputs=None, outputs=text_input)
    ex2.click(fn=lambda: EXAMPLES[1], inputs=None, outputs=text_input)
    ex3.click(fn=lambda: EXAMPLES[2], inputs=None, outputs=text_input)
    ex4.click(fn=lambda: EXAMPLES[3], inputs=None, outputs=text_input)
    ex5.click(fn=lambda: EXAMPLES[4], inputs=None, outputs=text_input)


if __name__ == "__main__":
    demo.launch(share=True)