import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import gradio as gr

# ============================================================
# ✅ REGISTER CUSTOM LAYERS (KERAS 3 SAFE)
# ============================================================

@keras.saving.register_keras_serializable()
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
            layers.Dense(embed_dim),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

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
            "rate": self.rate,
        })
        return config


@keras.saving.register_keras_serializable()
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
            "embed_dim": self.embed_dim,
        })
        return config


# ============================================================
# ✅ LOAD TOKENIZER
# ============================================================

with open("improved_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 80


# ============================================================
# ✅ LOAD MODEL (CRITICAL FIX HERE)
# ============================================================

model = tf.keras.models.load_model(
    "improved_bangla_sentiment.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    },
    compile=False,
)


# ============================================================
# ✅ PREPROCESS FUNCTION
# ============================================================

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LEN
    )
    return padded


# ============================================================
# ✅ PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)[0]

    # Binary sigmoid case
    if np.ndim(prediction) == 0 or len(np.atleast_1d(prediction)) == 1:
        score = float(prediction)
        label = "Positive 😊" if score >= 0.5 else "Negative 😠"
        return f"{label}\nConfidence: {score:.4f}"

    # Softmax case
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    label_map = {
        0: "Negative 😠",
        1: "Positive 😊",
    }

    return f"{label_map[class_id]}\nConfidence: {confidence:.4f}"


# ============================================================
# ✅ GRADIO UI
# ============================================================

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="এখানে বাংলা বাক্য লিখুন...",
        label="Bangla Input Text",
    ),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Bangla Transformer Sentiment Analyzer",
    description="Enter a Bangla sentence to classify sentiment.",
    allow_flagging="never",
)

demo.launch()