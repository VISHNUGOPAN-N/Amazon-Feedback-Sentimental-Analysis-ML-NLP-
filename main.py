import streamlit as st
import pickle
import time

with open("model.pkl", "rb") as obj:
    model = pickle.load(obj)

with open("vectorizer.pkl", "rb") as obj2:
    vectorizer = pickle.load(obj2)

def predict(text):
    """Provide marketing recommendations based on customer feedback."""
    try:
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]
        if prediction == 1:
            return "🎯 The customer seems interested. You should market your product to them."
        else:
            return "🚫 The customer does not seem interested. No need to market your product to them."
    except Exception as e:
        return f"Error: {e}"


st.set_page_config(page_title="Marketing AI", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-image: url('https://source.unsplash.com/1600x900/?marketing,business');
            background-size: cover;
        }
        .stTextInput, .stTextArea {
            border-radius: 10px;
            border: 2px solid #FF4B4B;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #E84343;
            transform: scale(1.05);
        }
        .result-box {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Marketing Predictor 🚀</h1>", unsafe_allow_html=True)
st.write("Enter a customer review below to analyze its sentiment and get marketing recommendations.")

review_text = st.text_area("📝 Enter your customer review:", height=150)

if st.button("📊 Analyze Sentiment"):
    if review_text:
        with st.spinner("🔍 Analyzing... Please wait."):
            time.sleep(2)
            result = predict(review_text)
        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review to analyze.")
