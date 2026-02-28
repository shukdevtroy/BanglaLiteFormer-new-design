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

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

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
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

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
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open("improved_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 80


# ============================================================
# LOAD MODEL (EXPLICIT CUSTOM OBJECTS)
# ============================================================

model = tf.keras.models.load_model(
    "improved_bangla_sentiment.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    },
    compile=False
)


# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LEN
    )
    return padded


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):
    if not text or text.strip() == "":
        return "Please enter Bangla text."

    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)[0]

    # Binary output (sigmoid)
    if np.ndim(prediction) == 0 or len(np.atleast_1d(prediction)) == 1:
        score = float(prediction)
        label = "Positive 😊" if score >= 0.5 else "Negative 😠"
        return f"{label}\nConfidence: {score:.4f}"

    # Softmax output
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    labels = {
        0: "Negative 😠",
        1: "Positive 😊"
    }

    return f"{labels[class_id]}\nConfidence: {confidence:.4f}"


# ============================================================
# GRADIO UI (GRADIO 4 SAFE)
# ============================================================

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="এখানে বাংলা বাক্য লিখুন...",
        label="Bangla Input Text"
    ),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Bangla Transformer Sentiment Analyzer",
    description="Enter a Bangla sentence to classify sentiment."
)
# ============================================================
# ENHANCED GRADIO UI
# ============================================================

examples = [
    ["এই পণ্যটি অসাধারণ! আমি খুব সন্তুষ্ট।"],
    ["ডেলিভারি অনেক দেরিতে এসেছে, খুব খারাপ অভিজ্ঞতা।"],
    ["কোয়ালিটি মোটামুটি ভালো, দাম একটু বেশি।"],
    ["আমি একদমই খুশি নই, পণ্যটি প্রত্যাশা অনুযায়ী নয়।"],
    ["চমৎকার সার্ভিস এবং দ্রুত ডেলিভারি।"]
]

with gr.Blocks(theme=gr.themes.Soft(), title="Bangla Sentiment Analyzer") as demo:

    gr.Markdown(
        """
        # 🧠 Bangla Transformer Sentiment Analyzer
        
        বাংলা রিভিউ বা মন্তব্য লিখুন এবং তা Positive না Negative তা তাৎক্ষণিকভাবে জেনে নিন।
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=4,
                placeholder="এখানে আপনার বাংলা রিভিউ লিখুন...",
                label="📝 আপনার রিভিউ"
            )

            submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
            clear_btn = gr.Button("🧹 Clear")

        with gr.Column():
            output_text = gr.Markdown(label="📊 Result")

    gr.Markdown("### ✨ Example Inputs")
    gr.Examples(
        examples=examples,
        inputs=text_input
    )

    # Button Actions
    def formatted_prediction(text):
        result = predict_sentiment(text)
        return f"## 🎯 Prediction Result\n\n{result}"

    submit_btn.click(
        formatted_prediction,
        inputs=text_input,
        outputs=output_text
    )

    clear_btn.click(
        lambda: "",
        inputs=None,
        outputs=text_input
    )

    gr.Markdown(
        """
        ---
        ⚡ Powered by Custom Transformer Model  
        Built with TensorFlow & Gradio  
        """
    )

if __name__ == "__main__":
    demo.launch()