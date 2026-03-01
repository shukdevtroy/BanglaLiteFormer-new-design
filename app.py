import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import gradio as gr


# ============================================================
# CUSTOM LAYERS (HF + TF SAFE)
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
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        super().build(input_shape)

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

    def build(self, input_shape):
        super().build(input_shape)

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
# PREDICTION LOGIC
# ============================================================

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    return padded


def predict_sentiment(text):
    if not text or text.strip() == "":
        return (
            '<div class="result-card empty">'
            '<span class="result-icon">💬</span>'
            '<p class="result-hint">কিছু একটা লিখুন বিশ্লেষণ করতে...</p>'
            '</div>'
        )

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
    bar_color = "#00e5a0" if is_positive else "#ff4d6d"
    label_text = "Positive" if is_positive else "Negative"
    emoji = "😊" if is_positive else "😠"
    label_class = "positive" if is_positive else "negative"
    bangla_label = "ইতিবাচক" if is_positive else "নেতিবাচক"

    return f"""
<div class="result-card {label_class} animate-in">
  <div class="result-header">
    <span class="result-emoji">{emoji}</span>
    <div class="result-labels">
      <span class="result-label-en">{label_text}</span>
      <span class="result-label-bn">{bangla_label}</span>
    </div>
    <span class="result-pct" style="color:{bar_color}">{pct}%</span>
  </div>
  <div class="confidence-track">
    <div class="confidence-fill" style="width:{pct}%; background:{bar_color};">
      <span class="confidence-glow" style="background:{bar_color};"></span>
    </div>
  </div>
  <p class="confidence-label">Confidence Score</p>
</div>
"""


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """
/* ── IMPORTS ───────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Hind+Siliguri:wght@400;600;700&display=swap');

/* ── RESET & ROOT ──────────────────────────────────── */
:root {
  --bg:        #05060f;
  --surface:   #0c0e1e;
  --card:      #111326;
  --border:    rgba(255,255,255,0.07);
  --accent1:   #7c3aed;
  --accent2:   #06b6d4;
  --accent3:   #f59e0b;
  --pos:       #00e5a0;
  --neg:       #ff4d6d;
  --text:      #e8eaf6;
  --muted:     #6b7280;
  --radius:    14px;
  --glow1:     0 0 40px rgba(124,58,237,0.25);
  --glow2:     0 0 40px rgba(6,182,212,0.2);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── BODY & GRADIO SHELL ───────────────────────────── */
body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
  min-height: 100vh;
}

.gradio-container {
  max-width: 860px !important;
  padding: 0 16px 60px !important;
}

/* ── ANIMATED BACKGROUND GRID ──────────────────────── */
.gradio-container::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
  animation: gridDrift 20s linear infinite;
}

@keyframes gridDrift {
  0%   { background-position: 0 0; }
  100% { background-position: 48px 48px; }
}

/* ── HERO HEADER ───────────────────────────────────── */
#hero-wrap {
  position: relative;
  text-align: center;
  padding: 52px 20px 36px;
  z-index: 1;
}

.hero-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  pointer-events: none;
  animation: orbFloat 6s ease-in-out infinite alternate;
}

.orb-purple {
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(124,58,237,0.35) 0%, transparent 70%);
  top: -60px; left: 50%;
  transform: translateX(-60%);
}

.orb-cyan {
  width: 240px; height: 240px;
  background: radial-gradient(circle, rgba(6,182,212,0.25) 0%, transparent 70%);
  top: 20px; right: 0;
  animation-delay: -3s;
}

@keyframes orbFloat {
  from { transform: translateX(-60%) translateY(0); }
  to   { transform: translateX(-60%) translateY(-24px); }
}

.hero-tag {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent2);
  border: 1px solid rgba(6,182,212,0.35);
  padding: 5px 14px;
  border-radius: 100px;
  margin-bottom: 20px;
  background: rgba(6,182,212,0.05);
  animation: fadeSlideDown 0.6s ease both;
}

.hero-title {
  font-size: clamp(28px, 5vw, 48px);
  font-weight: 800;
  line-height: 1.1;
  background: linear-gradient(135deg, #fff 0%, var(--accent1) 50%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
  animation: fadeSlideDown 0.6s 0.1s ease both;
}

.hero-subtitle {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 16px;
  color: var(--muted);
  animation: fadeSlideDown 0.6s 0.2s ease both;
}

.hero-badges {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 18px;
  flex-wrap: wrap;
  animation: fadeSlideDown 0.6s 0.3s ease both;
}

.badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 4px 12px;
  border-radius: 100px;
  border: 1px solid var(--border);
  color: var(--muted);
  background: var(--surface);
}

/* ── MAIN CARD ─────────────────────────────────────── */
#main-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px;
  position: relative;
  z-index: 1;
  box-shadow: var(--glow1), 0 20px 60px rgba(0,0,0,0.5);
  animation: fadeSlideUp 0.7s 0.2s ease both;
  overflow: hidden;
}

#main-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent1), var(--accent2), transparent);
  animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
  0%   { opacity: 0.4; }
  50%  { opacity: 1; }
  100% { opacity: 0.4; }
}

/* ── SECTION LABELS ────────────────────────────────── */
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent1);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-label::before {
  content: '';
  width: 20px; height: 1px;
  background: var(--accent1);
}

/* ── TEXTAREA ──────────────────────────────────────── */
.gr-textbox, .gr-textbox textarea,
label.block textarea,
textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 16px !important;
  line-height: 1.7 !important;
  padding: 16px !important;
  resize: none !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
  caret-color: var(--accent2) !important;
}

textarea:focus {
  border-color: var(--accent1) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.15), var(--glow1) !important;
  outline: none !important;
}

textarea::placeholder {
  color: rgba(107,114,128,0.6) !important;
  font-family: 'Hind Siliguri', sans-serif !important;
}

label[for], .label-wrap span, .svelte-1f354aw {
  font-family: 'Syne', sans-serif !important;
  color: var(--muted) !important;
  font-size: 13px !important;
}

/* ── BUTTONS ───────────────────────────────────────── */
button.primary, .gr-button-primary, button[variant="primary"] {
  background: linear-gradient(135deg, var(--accent1), #5b21b6) !important;
  border: none !important;
  border-radius: var(--radius) !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  padding: 14px 28px !important;
  cursor: pointer !important;
  transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s !important;
  position: relative;
  overflow: hidden;
  letter-spacing: 0.5px;
}

button.primary::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
  opacity: 0;
  transition: opacity 0.2s;
}

button.primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(124,58,237,0.45) !important;
}

button.primary:hover::after { opacity: 1; }
button.primary:active { transform: translateY(0) !important; }

button.secondary, .gr-button-secondary {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--muted) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  padding: 13px 22px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
}

button.secondary:hover {
  border-color: rgba(255,255,255,0.15) !important;
  color: var(--text) !important;
}

/* ── RESULT CARD (HTML output) ─────────────────────── */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  min-height: 110px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.result-card.positive { border-color: rgba(0,229,160,0.25); }
.result-card.negative { border-color: rgba(255,77,109,0.25); }

.result-card.empty {
  align-items: center;
  gap: 10px;
  text-align: center;
}

.result-icon { font-size: 28px; opacity: 0.4; }

.result-hint {
  font-family: 'Hind Siliguri', sans-serif;
  color: var(--muted);
  font-size: 14px;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 18px;
}

.result-emoji { font-size: 36px; line-height: 1; }

.result-labels {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.result-label-en {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 22px;
  color: var(--text);
  letter-spacing: -0.5px;
}

.result-label-bn {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 14px;
  color: var(--muted);
}

.result-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 32px;
  font-weight: 600;
  letter-spacing: -1px;
}

/* ── CONFIDENCE BAR ────────────────────────────────── */
.confidence-track {
  background: rgba(255,255,255,0.05);
  border-radius: 100px;
  height: 6px;
  overflow: hidden;
  margin-bottom: 8px;
}

.confidence-fill {
  height: 100%;
  border-radius: 100px;
  position: relative;
  transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
}

.confidence-glow {
  position: absolute;
  right: 0; top: 50%;
  transform: translateY(-50%);
  width: 10px; height: 10px;
  border-radius: 50%;
  filter: blur(4px);
  opacity: 0.8;
  animation: pulseDot 1.5s ease-in-out infinite;
}

@keyframes pulseDot {
  0%, 100% { transform: translateY(-50%) scale(1); opacity: 0.8; }
  50%       { transform: translateY(-50%) scale(1.6); opacity: 0.4; }
}

.confidence-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
}

/* ── ANIMATE-IN ────────────────────────────────────── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-16px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}

.animate-in {
  animation: fadeSlideUp 0.5s cubic-bezier(0.23, 1, 0.32, 1) both;
}

/* ── EXAMPLES SECTION ──────────────────────────────── */
.examples-header {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  margin: 24px 0 12px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  z-index: 1;
  position: relative;
}

.gr-samples table, .gr-samples-table,
table.gr-samples-table {
  background: transparent !important;
}

.gr-samples td, .gr-samples th,
table.gr-samples-table td {
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 14px !important;
  color: var(--muted) !important;
  border-color: var(--border) !important;
  background: var(--surface) !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: background 0.2s, color 0.2s !important;
}

.gr-samples tr:hover td {
  background: rgba(124,58,237,0.08) !important;
  color: var(--text) !important;
}

/* ── FOOTER ────────────────────────────────────────── */
#footer-wrap {
  text-align: center;
  padding: 28px 0 0;
  z-index: 1;
  position: relative;
}

.footer-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 1.5px;
  color: var(--muted);
  opacity: 0.5;
}

.footer-dot {
  display: inline-block;
  width: 4px; height: 4px;
  background: var(--accent1);
  border-radius: 50%;
  margin: 0 8px;
  vertical-align: middle;
  animation: pulseDot 2s ease-in-out infinite;
}

/* ── SCROLLBAR ─────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent1); }

/* ── MISC GRADIO CLEANUP ───────────────────────────── */
.gr-form, .gr-box, .gr-panel { background: transparent !important; border: none !important; }
footer { display: none !important; }
.gap, .gap-2 { gap: 16px !important; }
"""


# ============================================================
# GRADIO BLOCKS UI
# ============================================================

examples = [
    ["এই পণ্যটি অসাধারণ! আমি খুব সন্তুষ্ট।"],
    ["ডেলিভারি অনেক দেরিতে এসেছে, খুব খারাপ অভিজ্ঞতা।"],
    ["কোয়ালিটি মোটামুটি ভালো, দাম একটু বেশি।"],
    ["আমি একদমই খুশি নই, পণ্যটি প্রত্যাশা অনুযায়ী নয়।"],
    ["চমৎকার সার্ভিস এবং দ্রুত ডেলিভারি।"]
]

with gr.Blocks(css=custom_css, title="Bangla Sentiment Analyzer", theme=gr.themes.Ocean()) as demo:

    # ── HERO ────────────────────────────────────────────
    gr.HTML("""
    <div id="hero-wrap">
      <div class="hero-orb orb-purple"></div>
      <div class="hero-orb orb-cyan"></div>
      <div class="hero-tag">transformer · nlp · bengali</div>
      <h1 class="hero-title">Bangla Sentiment<br>Analyzer</h1>
      <p class="hero-subtitle">বাংলা রিভিউ বিশ্লেষণ করুন — তাৎক্ষণিকভাবে</p>
      <div class="hero-badges">
        <span class="badge">⚡ TensorFlow</span>
        <span class="badge">🔤 Transformer</span>
        <span class="badge">🇧🇩 Bengali NLP</span>
        <span class="badge">🎯 Binary Classification</span>
      </div>
    </div>
    """)

    # ── MAIN CARD ────────────────────────────────────────
    with gr.Group(elem_id="main-card"):

        gr.HTML('<p class="section-label">Input</p>')

        text_input = gr.Textbox(
            lines=4,
            placeholder="এখানে আপনার বাংলা রিভিউ বা মন্তব্য লিখুন...",
            label="",
            show_label=False,
        )

        with gr.Row():
            submit_btn = gr.Button("🔍  Analyze Sentiment", variant="primary", scale=3)
            clear_btn  = gr.Button("✕  Clear", variant="secondary", scale=1)

        gr.HTML('<p class="section-label" style="margin-top:24px;">Result</p>')

        output_html = gr.HTML(
            value='<div class="result-card empty"><span class="result-icon">💬</span>'
                  '<p class="result-hint">কিছু একটা লিখুন বিশ্লেষণ করতে...</p></div>'
        )

        gr.HTML('<p class="examples-header">✦ Example Inputs — click to try</p>')

        gr.Examples(
            examples=examples,
            inputs=text_input,
            label="",
        )

    # ── FOOTER ──────────────────────────────────────────
    gr.HTML("""
    <div id="footer-wrap">
      <p class="footer-text">
        Powered by Custom Transformer
        <span class="footer-dot"></span>
        Built with TensorFlow &amp; Gradio
      </p>
    </div>
    """)

    # ── EVENTS ──────────────────────────────────────────
    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    clear_btn.click(fn=lambda: ("", '<div class="result-card empty"><span class="result-icon">💬</span>'
                                    '<p class="result-hint">কিছু একটা লিখুন বিশ্লেষণ করতে...</p></div>'),
                    inputs=None, outputs=[text_input, output_html])

    # Also trigger on Enter
    text_input.submit(fn=predict_sentiment, inputs=text_input, outputs=output_html)


if __name__ == "__main__":
    demo.launch()