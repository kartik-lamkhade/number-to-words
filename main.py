import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# -------------------------
# Load Model
# -------------------------
model = load_model("model.h5")  # make sure your model is in .keras format

# -------------------------
# Load Tokenizer
# -------------------------
with open("tokenizer_out.json", "r") as f:
    tokenizer_data = f.read()

tokenizer_out = tokenizer_from_json(tokenizer_data)

# Index to word dictionary
tokenizer_word_index = tokenizer_out.word_index
idx2word = {v: k for k, v in tokenizer_word_index.items()}

start_token = tokenizer_word_index['start']
end_token = tokenizer_word_index['end']

# -------------------------
# Streamlit App UI (Dark Theme)
# -------------------------
st.set_page_config(page_title="Number to Words", page_icon="🔢", layout="centered")

st.markdown("""
<style>
/* Dark background */
.stApp {
    background-color: #0f111a;
    color: #ffffff;
    font-family: 'Helvetica', sans-serif;
}

/* Title style */
.title {
    color: #00d8ff;
    font-size: 38px;
    font-weight: bold;
}

/* Note style */
.note {
    color: #ff6b6b;
    font-size: 16px;
}

/* Button style */
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    height: 50px;
    width: 180px;
    border-radius: 10px;
}

/* Output card */
.output-card {
    background-color: #1a1c2b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🔢 Number to Words Converter</h1>', unsafe_allow_html=True)

# Note
st.markdown('<p class="note">This is an <b>initial version</b> of the model. Currently predicts numbers from 1 to 10,000 accurately.</p>', unsafe_allow_html=True)

# Input and button
number_input = st.number_input("Enter a number", min_value=0.0, step=1.0, format="%.0f")
predict_button = st.button("🔹 Predict")

# -------------------------
# Prediction Logic
# -------------------------
if predict_button:
    with st.spinner("Converting number to words..."):
        encoder_input = np.array([[number_input]])  # shape (1,1)

        # IMPORTANT: Use decoder length same as trained model
        decoder_length = 8  # match your trained model
        decoder_input = np.zeros((1, decoder_length), dtype=int)
        decoder_input[0, 0] = start_token

        decoded_sequence = []

        for t in range(1, decoder_length):
            preds = model.predict([encoder_input, decoder_input], verbose=0)
            next_token = np.argmax(preds[0, t-1, :])

            if next_token == end_token:
                break

            decoded_sequence.append(next_token)
            decoder_input[0, t] = next_token

        decoded_words = [idx2word[i] for i in decoded_sequence]
        decoded_text = " ".join(decoded_words)

    # Output in dark card
    st.markdown(f"""
    <div class="output-card">
        <h3>✨ In Words:</h3>
        <p style="font-size:24px; color:#00d8ff;">{decoded_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer note
st.markdown("---")
st.markdown('<p class="note">⚠️ Model is in initial state, trained only for numbers 1 to 10,000.</p>', unsafe_allow_html=True)
