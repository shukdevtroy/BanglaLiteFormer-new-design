import os
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
<div class="result-empty">
  <div class="empty-visual">
    <div class="pulse-ring"></div>
    <div class="pulse-ring delay1"></div>
    <div class="pulse-ring delay2"></div>
    <div class="empty-icon-inner">💬</div>
  </div>
  <p class="empty-text">কিছু একটা লিখুন বিশ্লেষণ করতে...</p>
  <p class="empty-sub">— Enter Bangla text above and click Analyze —</p>
</div>
"""


def predict_sentiment(text):
    if not text or text.strip() == "":
        return EMPTY_RESULT

    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)[0]

    if np.ndim(prediction) == 0 or len(np.atleast_1d(prediction)) == 1:
        score = float(prediction)
        is_positive = score >= 0.5
        confidence = score if is_positive else 1 - score
    else:
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        is_positive = class_id == 1

    pct = int(confidence * 100)

    # SVG radial gauge
    radius = 54
    circumference = 2 * 3.14159 * radius
    dash_offset = circumference * (1 - confidence)

    if is_positive:
        emoji       = "😊"
        label_en    = "Positive"
        label_bn    = "ইতিবাচক"
        bar_color   = "#00e5a0"
        glow        = "rgba(0,229,160,0.35)"
        bg          = "rgba(0,229,160,0.05)"
        border      = "rgba(0,229,160,0.25)"
        text_shadow = "0 0 40px rgba(0,229,160,0.6)"
        ring_color  = "#00e5a0"
        track_color = "rgba(0,229,160,0.12)"
        intensity   = "Highly" if pct >= 80 else ("Moderately" if pct >= 60 else "Slightly")
    else:
        emoji       = "😠"
        label_en    = "Negative"
        label_bn    = "নেতিবাচক"
        bar_color   = "#ff4d6d"
        glow        = "rgba(255,77,109,0.35)"
        bg          = "rgba(255,77,109,0.05)"
        border      = "rgba(255,77,109,0.25)"
        text_shadow = "0 0 40px rgba(255,77,109,0.6)"
        ring_color  = "#ff4d6d"
        track_color = "rgba(255,77,109,0.12)"
        intensity   = "Highly" if pct >= 80 else ("Moderately" if pct >= 60 else "Slightly")

    word_count = len(text.split())

    return f"""
<div class="result-card" style="--card-bg:{bg}; --card-border:{border}; --glow:{glow}; --bar:{bar_color}; --track:{track_color};">

  <!-- Top row: emoji label + radial gauge -->
  <div class="result-top">
    <div class="result-left">
      <div class="emoji-wrap" style="--glow:{glow}; --bar:{bar_color};">
        <span class="result-emoji">{emoji}</span>
        <div class="emoji-ring" style="border-color:{bar_color}; box-shadow:0 0 20px {glow};"></div>
      </div>
      <div class="result-text-block">
        <span class="intensity-tag" style="color:{bar_color}; border-color:{border}; background:{bg};">{intensity}</span>
        <span class="result-label-en" style="color:{bar_color}; text-shadow:{text_shadow};">{label_en}</span>
        <span class="result-label-bn">{label_bn}</span>
      </div>
    </div>

    <!-- Radial SVG gauge -->
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 130 130" width="130" height="130">
        <defs>
          <filter id="glow-filter">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <!-- Track circle -->
        <circle cx="65" cy="65" r="{radius}" fill="none"
          stroke="{track_color}" stroke-width="10" stroke-linecap="round"/>
        <!-- Progress arc -->
        <circle cx="65" cy="65" r="{radius}" fill="none"
          stroke="{ring_color}" stroke-width="10" stroke-linecap="round"
          stroke-dasharray="{circumference:.1f}"
          stroke-dashoffset="{dash_offset:.1f}"
          transform="rotate(-90 65 65)"
          filter="url(#glow-filter)"
          class="gauge-arc"/>
        <!-- Center text -->
        <text x="65" y="58" text-anchor="middle" class="gauge-pct" fill="{bar_color}">{pct}</text>
        <text x="65" y="76" text-anchor="middle" class="gauge-sub" fill="rgba(136,146,176,0.7)">%</text>
      </svg>
    </div>
  </div>

  <!-- Horizontal bar -->
  <p class="bar-label">CONFIDENCE SCORE</p>
  <div class="confidence-track">
    <div class="confidence-fill" style="width:{pct}%; background:linear-gradient(90deg,{bar_color}55,{bar_color});">
      <div class="bar-glow" style="background:{bar_color}; box-shadow: 0 0 18px {bar_color}, 0 0 40px {glow};"></div>
    </div>
    <div class="bar-markers">
      <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
    </div>
  </div>

  <!-- Meta chips -->
  <div class="meta-row">
    <div class="meta-chip" style="border-color:{border};">
      <span class="meta-icon">🔤</span>
      <span class="meta-val">{word_count}</span>
      <span class="meta-key">words</span>
    </div>
    <div class="meta-chip" style="border-color:{border};">
      <span class="meta-icon">🎯</span>
      <span class="meta-val">{pct}%</span>
      <span class="meta-key">confidence</span>
    </div>
    <div class="meta-chip" style="border-color:{border};">
      <span class="meta-icon">⚡</span>
      <span class="meta-val">Transformer</span>
      <span class="meta-key">model</span>
    </div>
  </div>
</div>
"""


# ============================================================
# CSS — UPGRADED
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

/* ── ROOT ───────────────────────────────────────────── */
:root {
  --bg:       #060810;
  --surface:  #0c0e1a;
  --card:     #0f1122;
  --border:   rgba(255,255,255,0.07);
  --accent:   #7c3aed;
  --cyan:     #22d3ee;
  --green:    #00e5a0;
  --red:      #ff4d6d;
  --text:     #eef0ff;
  --muted:    #7a84a8;
  --r:        20px;
}

/* ── BASE RESET ─────────────────────────────────────── */
html, body {
  width: 100% !important;
  background: var(--bg) !important;
  overflow-x: hidden !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  font-family: 'Outfit', sans-serif !important;
  color: var(--text) !important;
}

/* ── AURORA BACKGROUND ──────────────────────────────── */
body {
  background: var(--bg) !important;
  position: relative;
}

body::after {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 10% 20%, rgba(124,58,237,0.13) 0%, transparent 60%),
    radial-gradient(ellipse 60% 60% at 90% 80%, rgba(34,211,238,0.10) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 50% 50%, rgba(0,229,160,0.05) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
  animation: auroraPulse 12s ease-in-out infinite alternate;
}

@keyframes auroraPulse {
  0%   { opacity: 0.7; transform: scale(1) translateY(0); }
  50%  { opacity: 1;   transform: scale(1.04) translateY(-20px); }
  100% { opacity: 0.8; transform: scale(1.01) translateY(10px); }
}

/* Noise texture overlay */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
  opacity: 0.4;
}

/* ── PAGE WRAPPER ───────────────────────────────────── */
#page-wrap {
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
  padding: 0 24px 100px;
  position: relative;
  z-index: 1;
}

/* ── HERO ───────────────────────────────────────────── */
#hero {
  text-align: center;
  padding: 72px 20px 52px;
  position: relative;
}

/* Floating orbs */
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  pointer-events: none;
  z-index: -1;
}
.orb-a {
  width: 600px; height: 350px;
  background: radial-gradient(circle, rgba(124,58,237,0.22) 0%, transparent 70%);
  top: -60px; left: 50%; transform: translateX(-50%);
  animation: floatA 10s ease-in-out infinite alternate;
}
.orb-b {
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(34,211,238,0.15) 0%, transparent 70%);
  top: 20px; right: -60px;
  animation: floatB 7s 1.5s ease-in-out infinite alternate;
}
.orb-c {
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,229,160,0.12) 0%, transparent 70%);
  bottom: -20px; left: -40px;
  animation: floatB 9s 3s ease-in-out infinite alternate;
}
@keyframes floatA { to { transform: translateX(-50%) translateY(-40px) scale(1.05); } }
@keyframes floatB { to { transform: translateY(-25px) scale(1.08); } }

/* Status chip */
.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'Space Mono', monospace;
  font-size: 10px;
  letter-spacing: 3.5px;
  text-transform: uppercase;
  color: var(--cyan);
  border: 1px solid rgba(34,211,238,0.22);
  padding: 7px 22px;
  border-radius: 100px;
  background: rgba(34,211,238,0.04);
  margin-bottom: 28px;
  backdrop-filter: blur(10px);
}
.chip-dot {
  width: 7px; height: 7px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--cyan);
  animation: blink 2.2s ease-in-out infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.15; transform: scale(0.7); }
}

/* Hero title */
.hero-title {
  font-size: clamp(36px, 6vw, 62px);
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -2.5px;
  margin-bottom: 16px;
  position: relative;
  display: inline-block;
}
.title-line1 {
  display: block;
  background: linear-gradient(135deg, #ffffff 0%, #e0d4ff 40%, #a78bfa 70%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-size: 200% 200%;
  animation: gradShift 6s ease-in-out infinite alternate;
}
.title-line2 {
  display: block;
  background: linear-gradient(135deg, var(--cyan) 0%, #a78bfa 50%, #ffffff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-size: 200% 200%;
  animation: gradShift 6s 1s ease-in-out infinite alternate;
}
@keyframes gradShift {
  0%   { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

.hero-sub {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 18px;
  font-weight: 400;
  color: var(--muted);
  margin-bottom: 28px;
  letter-spacing: 0.2px;
}

/* Badges */
.hero-badges {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
}
.badge {
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  padding: 6px 16px;
  border-radius: 100px;
  border: 1px solid rgba(255,255,255,0.09);
  color: var(--muted) !important;
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(8px);
  transition: all 0.3s;
  cursor: default;
}
.badge:hover {
  color: var(--text) !important;
  border-color: rgba(124,58,237,0.4);
  background: rgba(124,58,237,0.08);
  transform: translateY(-2px);
}

/* ── MAIN ANALYSIS CARD ─────────────────────────────── */
#analysis-card {
  background: rgba(15,17,34,0.7);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 28px;
  padding: 40px 40px 32px;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.04),
    0 40px 100px rgba(0,0,0,0.6),
    0 0 80px rgba(124,58,237,0.06);
}

/* Top shimmer line */
#analysis-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(124,58,237,0.6) 20%,
    rgba(34,211,238,0.9) 50%,
    rgba(124,58,237,0.6) 80%,
    transparent 100%);
  background-size: 200% 100%;
  animation: shimLine 4s linear infinite;
}
@keyframes shimLine {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Bottom glow */
#analysis-card::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 20%; right: 20%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(34,211,238,0.3), transparent);
}

/* ── SECTION LABEL ──────────────────────────────────── */
.sec-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Space Mono', monospace;
  font-size: 10px;
  letter-spacing: 3.5px;
  text-transform: uppercase;
  color: rgba(255,255,255,0.4) !important;
  margin-bottom: 14px;
}
.sec-label::before {
  content: '';
  width: 20px; height: 1.5px;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  border-radius: 2px;
}

/* ── TEXTAREA ───────────────────────────────────────── */
label > span { display: none !important; }

textarea {
  width: 100% !important;
  background: rgba(255,255,255,0.025) !important;
  border: 1.5px solid rgba(255,255,255,0.08) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 17px !important;
  font-weight: 400 !important;
  line-height: 1.8 !important;
  padding: 18px 22px !important;
  resize: none !important;
  caret-color: var(--cyan) !important;
  transition: border-color 0.3s, box-shadow 0.3s, background 0.3s !important;
}
textarea:focus {
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(124,58,237,0.55) !important;
  box-shadow:
    0 0 0 4px rgba(124,58,237,0.1),
    0 0 50px rgba(124,58,237,0.08) !important;
  outline: none !important;
}
textarea::placeholder {
  color: rgba(122,132,168,0.35) !important;
  font-family: 'Hind Siliguri', sans-serif !important;
}

/* ── BUTTONS ────────────────────────────────────────── */
.btn-row { margin: 20px 0 32px !important; gap: 12px !important; }

button[variant="primary"], .gr-button-primary {
  background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 50%, #4c1d95 100%) !important;
  border: 1px solid rgba(124,58,237,0.4) !important;
  border-radius: 14px !important;
  color: #fff !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  letter-spacing: 0.3px !important;
  padding: 16px 28px !important;
  cursor: pointer !important;
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
}
button[variant="primary"]::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}
button[variant="primary"]:hover {
  transform: translateY(-3px) scale(1.02) !important;
  box-shadow: 0 12px 40px rgba(124,58,237,0.55), 0 0 0 1px rgba(124,58,237,0.5) !important;
}
button[variant="primary"]:active {
  transform: translateY(0) scale(0.99) !important;
}

button[variant="secondary"], .gr-button-secondary {
  background: rgba(255,255,255,0.03) !important;
  border: 1.5px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  color: var(--muted) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 14px !important;
  padding: 16px 22px !important;
  cursor: pointer !important;
  transition: all 0.25s !important;
}
button[variant="secondary"]:hover {
  background: rgba(255,255,255,0.07) !important;
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.16) !important;
  transform: translateY(-1px) !important;
}

/* ── RESULT WRAPPER ─────────────────────────────────── */
.result-wrap > .block, .result-wrap .prose {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Empty state */
.result-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 14px; min-height: 160px;
  border: 1.5px dashed rgba(255,255,255,0.06);
  border-radius: 18px; text-align: center; padding: 40px;
  position: relative;
}

.empty-visual {
  position: relative;
  width: 64px; height: 64px;
  display: flex; align-items: center; justify-content: center;
}

.pulse-ring {
  position: absolute; inset: 0;
  border: 1.5px solid rgba(124,58,237,0.3);
  border-radius: 50%;
  animation: pulseRing 3s ease-out infinite;
}
.pulse-ring.delay1 { animation-delay: 1s; }
.pulse-ring.delay2 { animation-delay: 2s; }

@keyframes pulseRing {
  0%   { transform: scale(0.8); opacity: 0.8; }
  100% { transform: scale(2.2); opacity: 0; }
}

.empty-icon-inner { font-size: 32px; position: relative; z-index: 1; opacity: 0.5; }

.empty-text {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 16px; font-weight: 500;
  background: linear-gradient(135deg, var(--accent), var(--cyan));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.empty-sub {
  font-family: 'Space Mono', monospace;
  font-size: 10px; letter-spacing: 1.5px;
  color: rgba(122,132,168,0.3);
}

/* ── RESULT CARD ────────────────────────────────────── */
.result-card {
  background: var(--card-bg, rgba(0,229,160,0.05));
  border: 1.5px solid var(--card-border, rgba(0,229,160,0.25));
  border-radius: 20px;
  padding: 28px 28px 22px;
  animation: fadeUp 0.5s cubic-bezier(0.23,1,0.32,1) both;
  box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03);
  position: relative;
  overflow: hidden;
}
.result-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--bar, #00e5a0), transparent);
  opacity: 0.7;
}

.result-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  gap: 16px;
}
.result-left {
  display: flex;
  align-items: center;
  gap: 18px;
}

/* Emoji */
.emoji-wrap {
  position: relative;
  width: 70px; height: 70px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.result-emoji {
  font-size: 42px;
  line-height: 1;
  position: relative;
  z-index: 1;
  animation: emojiPop 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes emojiPop {
  0%   { transform: scale(0) rotate(-20deg); opacity: 0; }
  100% { transform: scale(1) rotate(0deg);  opacity: 1; }
}
.emoji-ring {
  position: absolute; inset: 0;
  border: 1.5px solid;
  border-radius: 50%;
  animation: spinSlow 8s linear infinite;
}
@keyframes spinSlow {
  to { transform: rotate(360deg); }
}

/* Text block */
.result-text-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.intensity-tag {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  border: 1px solid;
  padding: 3px 10px;
  border-radius: 100px;
  width: fit-content;
  margin-bottom: 2px;
}
.result-label-en {
  font-family: 'Outfit', sans-serif;
  font-weight: 900;
  font-size: 34px;
  letter-spacing: -1.5px;
  line-height: 1;
  animation: labelSlide 0.5s 0.1s cubic-bezier(0.23,1,0.32,1) both;
}
@keyframes labelSlide {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}
.result-label-bn {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 15px;
  color: var(--muted);
  font-weight: 500;
}

/* Radial gauge */
.gauge-wrap {
  flex-shrink: 0;
  animation: gaugeFadeIn 0.6s 0.2s ease both;
}
@keyframes gaugeFadeIn {
  from { opacity: 0; transform: scale(0.8) rotate(-10deg); }
  to   { opacity: 1; transform: scale(1) rotate(0deg); }
}
.gauge-svg { display: block; }
.gauge-arc {
  transition: stroke-dashoffset 1.2s cubic-bezier(0.34,1,0.64,1);
}
.gauge-pct {
  font-family: 'Outfit', sans-serif;
  font-size: 26px;
  font-weight: 900;
  letter-spacing: -1px;
}
.gauge-sub {
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  font-weight: 700;
}

/* Bar */
.bar-label {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: rgba(122,132,168,0.4);
  margin-bottom: 10px;
}
.confidence-track {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 100px;
  height: 10px;
  position: relative;
  overflow: visible;
  margin-bottom: 8px;
}
.confidence-fill {
  height: 100%;
  border-radius: 100px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  animation: barGrow 1s cubic-bezier(0.34,1,0.64,1) both;
  animation-delay: 0.3s;
}
@keyframes barGrow {
  from { width: 0% !important; }
}
.bar-glow {
  width: 16px; height: 16px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-right: -8px;
  animation: tipPulse 2s ease-in-out infinite;
}
@keyframes tipPulse {
  50% { transform: scale(1.7); opacity: 0.3; }
}
.bar-markers {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  padding: 0 2px;
}
.bar-markers span {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  color: rgba(122,132,168,0.25);
}

/* Meta chips row */
.meta-row {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  flex-wrap: wrap;
}
.meta-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 14px;
  border-radius: 100px;
  border: 1px solid;
  background: rgba(0,0,0,0.2);
  flex: 1;
  min-width: 0;
}
.meta-icon { font-size: 14px; flex-shrink: 0; }
.meta-val {
  font-family: 'Outfit', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
  white-space: nowrap;
}
.meta-key {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 1px;
  white-space: nowrap;
}

/* ── DIVIDER ────────────────────────────────────────── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
  margin: 28px 0 20px;
}

/* ── EXAMPLES ───────────────────────────────────────── */
.ex-label {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: rgba(255,255,255,0.25) !important;
  margin-bottom: 14px;
}

table {
  width: 100% !important;
  border-collapse: separate !important;
  border-spacing: 0 6px !important;
}
table thead { display: none !important; }
table td {
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  color: rgba(200,208,232,0.8) !important;
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.05) !important;
  padding: 13px 18px !important;
  border-radius: 10px !important;
  cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
}
table tr:hover td {
  background: rgba(124,58,237,0.08) !important;
  color: #fff !important;
  border-color: rgba(124,58,237,0.22) !important;
  transform: translateX(4px) !important;
}

/* ── FOOTER ─────────────────────────────────────────── */
#footer {
  text-align: center;
  padding: 36px 0 10px;
  position: relative;
  z-index: 1;
}
.footer-text {
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  letter-spacing: 2px;
  color: rgba(122,132,168,0.25);
}
.footer-dot {
  display: inline-block;
  width: 3px; height: 3px;
  background: var(--accent);
  border-radius: 50%;
  margin: 0 12px;
  vertical-align: middle;
  opacity: 0.4;
}

/* ── ANIMATIONS ─────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── SCROLLBAR ──────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 2px; }

/* ── GRADIO CLEANUP ─────────────────────────────────── */
footer, .svelte-footer { display: none !important; }
.gr-form, .gr-box, .gr-panel { background: transparent !important; border: none !important; box-shadow: none !important; }
.gap, .gap-2 { gap: 0 !important; }
"""


# ============================================================
# GRADIO UI
# ============================================================

examples = [
    ["এই পণ্যটি অসাধারণ! আমি খুব সন্তুষ্ট।"],
    ["ডেলিভারি অনেক দেরিতে এসেছে, খুব খারাপ অভিজ্ঞতা।"],
    ["কোয়ালিটি মোটামুটি ভালো, দাম একটু বেশি।"],
    ["আমি একদমই খুশি নই, পণ্যটি প্রত্যাশা অনুযায়ী নয়।"],
    ["চমৎকার সার্ভিস এবং দ্রুত ডেলিভারি।"]
]

with gr.Blocks(css=css, title="Bangla Sentiment Analyzer") as demo:

    with gr.Column(elem_id="page-wrap"):

        # ── HERO ──────────────────────────────────────
        gr.HTML("""
        <div id="hero">
          <div class="orb orb-a"></div>
          <div class="orb orb-b"></div>
          <div class="orb orb-c"></div>

          <div class="hero-chip">
            <span class="chip-dot"></span>transformer · nlp · bengali
          </div>

          <h1 class="hero-title">
            <span class="title-line1">Bangla Sentiment</span>
            <span class="title-line2">Analyzer</span>
          </h1>

          <p class="hero-sub">বাংলা রিভিউ বা মন্তব্য লিখুন — তাৎক্ষণিক বিশ্লেষণ পান</p>

          <div class="hero-badges">
            <span class="badge">⚡ TensorFlow</span>
            <span class="badge">🔤 Transformer</span>
            <span class="badge">🇧🇩 Bengali NLP</span>
            <span class="badge">🎯 Binary Classification</span>
          </div>
        </div>
        """)

        # ── CARD ──────────────────────────────────────
        with gr.Column(elem_id="analysis-card"):

            gr.HTML('<div class="sec-label">Input</div>')

            text_input = gr.Textbox(
                lines=4,
                placeholder="এখানে আপনার বাংলা রিভিউ বা মন্তব্য লিখুন...",
                label="",
                show_label=False,
            )

            with gr.Row(elem_classes="btn-row"):
                submit_btn = gr.Button("🔍  Analyze Sentiment", variant="primary", scale=3)
                clear_btn  = gr.Button("✕  Clear", variant="secondary", scale=1)

            gr.HTML('<div class="sec-label">Result</div>')

            with gr.Column(elem_classes="result-wrap"):
                output_html = gr.HTML(value=EMPTY_RESULT)

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="ex-label">✦ Example inputs — click to try</div>')

            gr.Examples(examples=examples, inputs=text_input, label="")

        # ── FOOTER ────────────────────────────────────
        gr.HTML("""
        <div id="footer">
          <p class="footer-text">
            Powered by Custom Transformer
            <span class="footer-dot"></span>
            Built with TensorFlow &amp; Gradio
            <span class="footer-dot"></span>
            Bengali NLP
          </p>
        </div>
        """)

    # ── EVENTS ────────────────────────────────────────
    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    text_input.submit(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    clear_btn.click(fn=lambda: ("", EMPTY_RESULT), inputs=None, outputs=[text_input, output_html])


if __name__ == "__main__":
    demo.launch()