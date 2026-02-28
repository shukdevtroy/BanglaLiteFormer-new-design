import gradio as gr
import tensorflow as tf
import pickle
import numpy as np

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
model = tf.keras.models.load_model("model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):
    # Preprocess exactly like training time
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)[0][0]

    if pred >= 0.5:
        return "🟢 Positive Sentiment"
    else:
        return "🔴 Negative Sentiment"

# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="বাংলা বাক্য লিখুন"),
    outputs=gr.Textbox(label="Prediction"),
    title="Bangla Sentiment Analyzer",
    description="Enter a Bangla sentence to detect sentiment."
)

demo.launch()