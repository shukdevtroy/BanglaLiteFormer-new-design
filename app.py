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
  <div class="empty-icon">💬</div>
  <p class="empty-text">কিছু একটা লিখুন বিশ্লেষণ করতে...</p>
  <p class="empty-sub">Enter Bangla text above and click Analyze</p>
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

    if is_positive:
        emoji, label_en, label_bn = "😊", "Positive", "ইতিবাচক"
        bar_color = "#00e5a0"
        glow       = "rgba(0,229,160,0.30)"
        bg         = "rgba(0,229,160,0.06)"
        border     = "rgba(0,229,160,0.28)"
    else:
        emoji, label_en, label_bn = "😠", "Negative", "নেতিবাচক"
        bar_color = "#ff4d6d"
        glow       = "rgba(255,77,109,0.30)"
        bg         = "rgba(255,77,109,0.06)"
        border     = "rgba(255,77,109,0.28)"

    return f"""
<div class="result-card" style="background:{bg}; border-color:{border};">
  <div class="result-top">
    <div class="result-left">
      <span class="result-emoji">{emoji}</span>
      <div class="result-text-block">
        <span class="result-label-en" style="color:{bar_color};">{label_en}</span>
        <span class="result-label-bn">{label_bn}</span>
      </div>
    </div>
    <div class="result-pct" style="color:{bar_color}; border-color:{border}; box-shadow:0 0 20px {glow};">
      {pct}%
    </div>
  </div>
  <p class="bar-label">Confidence Score</p>
  <div class="confidence-track">
    <div class="confidence-fill" style="width:{pct}%; background:linear-gradient(90deg,{bar_color}88,{bar_color});">
      <div class="bar-tip" style="background:{bar_color}; box-shadow:0 0 12px {bar_color};"></div>
    </div>
  </div>
</div>
"""


# ============================================================
# CSS
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

/* ── ROOT ─────────────────────────────────────────── */
:root {
  --bg:      #07080f;
  --surface: #0f1120;
  --card:    #131526;
  --border:  rgba(255,255,255,0.08);
  --accent:  #7c3aed;
  --cyan:    #22d3ee;
  --text:    #f1f3ff;
  --muted:   #8892b0;
  --r:       16px;
}

/* ── FULL WIDTH RESET ─────────────────────────────── */
html, body {
  width: 100% !important;
  background: var(--bg) !important;
}
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #858282ab !important;
  font-family: 'Syne', sans-serif !important;
  color: white !important;
}

/* ── ANIMATED GRID ────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(124,58,237,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,58,237,0.035) 1px, transparent 1px);
  background-size: 56px 56px;
  pointer-events: none;
  z-index: 0;
  animation: gridScroll 28s linear infinite;
}
@keyframes gridScroll {
  to { background-position: 56px 56px; }
}

/* ── CENTERING WRAPPER ────────────────────────────── */
#page-wrap {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
  padding: 0 28px 80px;
  position: relative;
  z-index: 1;
}

/* ── HERO ─────────────────────────────────────────── */
#hero {
  text-align: center;
  padding: 64px 20px 48px;
  position: relative;
}
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(100px);
  pointer-events: none;
  z-index: -1;
}
.orb-a {
  width: 550px; height: 380px;
  background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 65%);
  top: -80px; left: 50%; transform: translateX(-50%);
  animation: orbA 8s ease-in-out infinite alternate;
}
.orb-b {
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(34,211,238,0.18) 0%, transparent 65%);
  top: 30px; right: -40px;
  animation: orbB 6s 2s ease-in-out infinite alternate;
}
@keyframes orbA { to { transform: translateX(-50%) translateY(-30px); } }
@keyframes orbB { to { transform: translateY(-20px) scale(1.1); } }

.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--cyan);
  border: 1px solid rgba(34,211,238,0.28);
  padding: 6px 20px;
  border-radius: 100px;
  background: rgba(34,211,238,0.05);
  margin-bottom: 22px;
}
.chip-dot {
  width: 6px; height: 6px;
  background: var(--cyan);
  border-radius: 50%;
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 50% { opacity: 0.15; } }

.hero-title {
  font-size: clamp(34px, 5.5vw, 56px);
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: -1.5px;
  background: linear-gradient(135deg, #fff 0%, #c4b5fd 45%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 14px;
}
.hero-sub {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 17px;
  color: var(--muted);
  margin-bottom: 22px;
}
.hero-badges {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
}
.badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  padding: 5px 14px;
  border-radius: 100px;
  border: 1px solid var(--border);
  color: white !important;
  background: var(--surface);
}

/* ── MAIN CARD ────────────────────────────────────── */
#analysis-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 36px 36px 28px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 80px rgba(124,58,237,0.08), 0 30px 80px rgba(0,0,0,0.55);
}
#analysis-card::before {
  content: '';
  position: absolute;
  top: 0; left: -100%; right: -100%;
  height: 1px;
  background: linear-gradient(90deg,
    transparent, rgba(124,58,237,0.7), rgba(34,211,238,0.9), transparent);
  animation: shimLine 5s linear infinite;
}
@keyframes shimLine {
  from { left: -100%; right: 100%; }
  to   { left: 100%; right: -100%; }
}

/* ── SECTION LABEL ────────────────────────────────── */
.sec-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: white !important;
  margin-bottom: 14px;
}
.sec-label::before {
  content: '';
  width: 22px; height: 1.5px;
  background: var(--accent);
  border-radius: 2px;
}

/* ── TEXTAREA ─────────────────────────────────────── */
label > span { display: none !important; }

textarea {
  width: 100% !important;
  background: rgba(255,255,255,0.035) !important;
  border: 1.5px solid rgba(255,255,255,0.09) !important;
  border-radius: 14px !important;
  color: black !important;
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 17px !important;
  font-weight: 500 !important;
  line-height: 1.75 !important;
  padding: 18px 20px !important;
  resize: none !important;
  caret-color: var(--cyan) !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus {
  border-color: rgba(124,58,237,0.6) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.15), 0 0 40px rgba(124,58,237,0.1) !important;
  outline: none !important;
}
textarea::placeholder {
  color: rgba(136,146,176,0.45) !important;
  font-family: 'Hind Siliguri', sans-serif !important;
}

/* ── BUTTONS ──────────────────────────────────────── */
.btn-row { margin: 20px 0 30px !important; gap: 12px !important; }

button[variant="primary"], .gr-button-primary {
  background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
  border: none !important;
  border-radius: 14px !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  padding: 16px 28px !important;
  cursor: pointer !important;
  transition: all 0.25s !important;
}
button[variant="primary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(124,58,237,0.5) !important; }
button[variant="primary"]:active { transform: translateY(0) !important; }

button[variant="secondary"], .gr-button-secondary {
  background: rgba(255,255,255,0.04) !important;
  border: 1.5px solid rgba(255,255,255,0.09) !important;
  border-radius: 14px !important;
  color: var(--muted) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 15px !important;
  padding: 16px 22px !important;
  cursor: pointer !important;
  transition: all 0.25s !important;
}
button[variant="secondary"]:hover {
  background: rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.18) !important;
}

/* ── RESULT ───────────────────────────────────────── */
.result-wrap > .block, .result-wrap .prose {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.result-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; min-height: 130px;
  border: 1.5px dashed rgba(255,255,255,0.07);
  border-radius: 14px; text-align: center; padding: 32px;
}
.empty-icon { font-size: 38px; opacity: 0.3; }
.empty-text {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 16px; color: var(--muted); font-weight: 500;
  color: #cd00ff !important;
}
.empty-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: rgba(136,146,176,0.38); letter-spacing: 0.4px;
  color: #cd00ff !important;
}

.result-card {
  border: 1.5px solid;
  border-radius: 16px;
  padding: 26px 26px 20px;
  animation: fadeUp 0.4s cubic-bezier(0.23,1,0.32,1) both;
}
.result-top {
  display: flex; align-items: center;
  justify-content: space-between; margin-bottom: 20px;
}
.result-left { display: flex; align-items: center; gap: 16px; }
.result-emoji { font-size: 46px; line-height: 1; }
.result-text-block { display: flex; flex-direction: column; gap: 5px; }
.result-label-en {
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 30px;
  letter-spacing: -0.8px; line-height: 1;
}
.result-label-bn {
  font-family: 'Hind Siliguri', sans-serif;
  font-size: 15px; color: var(--muted); font-weight: 500;
}
.result-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 40px; font-weight: 600;
  letter-spacing: -2px; line-height: 1;
  border: 1.5px solid;
  border-radius: 12px;
  padding: 10px 20px;
  background: rgba(0,0,0,0.2);
}
.bar-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 2.5px;
  text-transform: uppercase;
  color: rgba(136,146,176,0.45);
  margin-bottom: 10px;
}
.confidence-track {
  background: rgba(255,255,255,0.06);
  border-radius: 100px;
  height: 8px; position: relative; overflow: visible;
}
.confidence-fill {
  height: 100%; border-radius: 100px;
  position: relative; display: flex;
  align-items: center; justify-content: flex-end;
}
.bar-tip {
  width: 14px; height: 14px; border-radius: 50%;
  flex-shrink: 0; margin-right: -7px;
  animation: tipPulse 1.8s ease-in-out infinite;
}
@keyframes tipPulse {
  50% { transform: scale(1.6); opacity: 0.4; }
}

/* ── DIVIDER ──────────────────────────────────────── */
.divider { height: 1px; background: var(--border); margin: 26px 0 18px; }

/* ── EXAMPLES ─────────────────────────────────────── */
.ex-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 2.5px;
  text-transform: uppercase;
  color: white !important; margin-bottom: 12px;
}
table {
  width: 100% !important;
  border-collapse: separate !important;
  border-spacing: 0 6px !important;
}
table thead { display: none !important; }
table td {
  font-family: 'Hind Siliguri', sans-serif !important;
  font-size: 14px !important; font-weight: 500 !important;
  color: #c8d0e8 !important;
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  padding: 13px 16px !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
}
table tr:hover td {
  background: rgba(124,58,237,0.1) !important;
  color: #fff !important;
  border-color: rgba(124,58,237,0.25) !important;
}

/* ── FOOTER ───────────────────────────────────────── */
#footer {
  text-align: center; padding: 30px 0 10px;
  position: relative; z-index: 1;
}
.footer-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; letter-spacing: 1.5px;
  color: rgba(136,146,176,0.3);
}
.footer-dot {
  display: inline-block;
  width: 4px; height: 4px;
  background: var(--accent); border-radius: 50%;
  margin: 0 10px; vertical-align: middle; opacity: 0.5;
}

/* ── ANIMATIONS ───────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── SCROLLBAR ────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.35); border-radius: 3px; }

/* ── GRADIO CLEANUP ───────────────────────────────── */
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
          <div class="hero-chip"><span class="chip-dot"></span>transformer · nlp · bengali</div>
          <h1 class="hero-title">Bangla Sentiment<br>Analyzer</h1>
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
          </p>
        </div>
        """)

    # ── EVENTS ────────────────────────────────────────
    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    text_input.submit(fn=predict_sentiment, inputs=text_input, outputs=output_html)
    clear_btn.click(fn=lambda: ("", EMPTY_RESULT), inputs=None, outputs=[text_input, output_html])


if __name__ == "__main__":
    demo.launch()