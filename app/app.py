import sys
import streamlit as st

# Helpful import wrapper so users see a clear message instead of a redacted traceback
missing = []
try:
    import joblib
except Exception as e:
    missing.append("joblib")

try:
    import numpy as np
except Exception:
    missing.append("numpy")

try:
    import pandas as pd
except Exception:
    missing.append("pandas")

# If any required library is missing, show a friendly error and stop
if missing:
    st.set_page_config(page_title="Dependency error")
    st.title("⛔ Missing dependencies")
    st.error(
        "Your app is missing the following Python package(s):\n\n" +
        ", ".join(missing) +
        "\n\nInstall them locally with `pip install <package>` or add them to `requirements.txt` and redeploy."
    )
    st.markdown("**Quick fixes:**")
    st.code("pip install " + " ".join(missing))
    st.stop()

# If we reach here, imports are present
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and feature names
data = joblib.load("xgboost_optuna_model.pkl")
model = data["model"]
feature_names = data["columns"]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎵 Music Popularity Predictor", page_icon="🎧", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background: #0f1116;
            color: #e5e5e5;
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            color: #1DB954;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #ccc;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #1DB954;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 3rem;
            width: 100%;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #169e47;
            transform: scale(1.02);
        }
        .prediction-box {
            background: #181a1f;
            border-radius: 15px;
            padding: 25px;
            margin-top: 25px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='main-title'>🎶 Music Popularity Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter your song’s attributes below and see if it will be a hit!</div>", unsafe_allow_html=True)

# --- Input Layout ---
col1, col2, col3 = st.columns(3)
with col1:
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    duration_ms = st.number_input("Duration (ms)", min_value=10000, max_value=600000, value=200000, step=1000)
    energy = st.slider("Energy", 0.0, 1.0, 0.6)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
with col2:
    key = st.number_input("Key", min_value=0, max_value=11, value=5)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0)
    mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
with col3:
    tempo = st.number_input("Tempo (BPM)", min_value=50.0, max_value=250.0, value=120.0)
    time_signature = st.selectbox("Time Signature", [3, 4, 5])
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    artist_encoded = st.number_input("Artist Encoded", min_value=0, value=1)
    song_title_encoded = st.number_input("Song Title Encoded", min_value=0, value=1)

# --- Predict Button ---
if st.button("🔮 Predict Popularity"):
    input_data = np.array([[acousticness, danceability, duration_ms, energy, instrumentalness,
                            key, liveness, loudness, mode, speechiness, tempo, time_signature,
                            valence, artist_encoded, song_title_encoded]])
    
    # Ensure feature order matches
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    result = "🔥 Popular Song" if prediction == 1 else "💤 Not So Popular"

    # Display result
    st.markdown(f"""
        <div class='prediction-box'>
            <h2 style='text-align:center;color:#1DB954;'>Prediction Result</h2>
            <h3 style='text-align:center;font-size:1.5rem;'>{result}</h3>
        </div>
    """, unsafe_allow_html=True)

    st.write("🔢 Probability of being popular:", round(proba[1], 3))
    st.write("🔢 Probability of not being popular:", round(proba[0], 3))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit & XGBoost</p>",
    unsafe_allow_html=True
)
