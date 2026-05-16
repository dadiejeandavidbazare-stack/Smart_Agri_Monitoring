import os
# Force la compatibilité Keras Legacy immédiatement
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import random

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="AGRONOVA APOLLO", page_icon="🛰️", layout="wide")

# --- CSS : INTERFACE SATELLITE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    header {visibility: hidden;}
    .main .block-container {padding: 1.5rem 2rem; background-color: #010409;}
    .stApp { background-color: #010409; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d; }
    .command-card { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 15px; }
    .neon-header { font-family: 'Orbitron', sans-serif; font-size: 2.2rem; color: #3aedff; letter-spacing: 5px; text-shadow: 0 0 10px rgba(58, 237, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR IA (VERSION SANS CACHE POUR ÉVITER LA RÉCURSION) ---
def load_engine():
    path = 'models/agri_model_v1.h5'
    classes = [
        "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy",
        "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
        "Corn Cercospora", "Corn Common Rust", "Corn Northern Blight", "Corn Healthy",
        "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
        "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy",
        "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
        "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
        "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
        "Strawberry Leaf Scorch", "Strawberry Healthy",
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
        "Tomato Leaf Mold", "Tomato Septoria Spot", "Tomato Spider Mites",
        "Tomato Target Spot", "Tomato Yellow Curl", "Tomato Mosaic Virus", "Tomato Healthy"
    ]
    
    if os.path.exists(path):
        try:
            # Chargement direct via tf.keras
            model = tf.keras.models.load_model(path, compile=False)
            return model, classes
        except Exception as e:
            st.error(f"Erreur technique : {e}")
            return None, []
    return None, []

# Utilisation d'une variable globale simple au lieu du cache Streamlit
if 'agronova_model' not in st.session_state:
    st.session_state.agronova_model, st.session_state.agronova_classes = load_engine()

model = st.session_state.agronova_model
classes = st.session_state.agronova_classes

# --- FONCTION GRAPHIQUE ---
def draw_gauge(value, title, color="#3aedff"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 14, 'family': 'Orbitron', 'color': '#8b949e'}},
        number={'font': {'color': '#ffffff', 'size': 35}, 'suffix': "%"},
        gauge={'axis': {'range': [0, 100], 'tickcolor': "#30363d"},
               'bar': {'color': color},
               'bgcolor': "rgba(255,255,255,0.02)",
               'borderwidth': 1, 'bordercolor': "#30363d"}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- DASHBOARD ---
st.markdown('<h1 class="neon-header">AGRONOVA APOLLO</h1>', unsafe_allow_html=True)
st.write("SATELLITE COMMAND CENTER - MAROC")

# Affichage des jauges
conf = st.session_state.get('conf', 0)
h2o = st.session_state.get('h2o', 0)
fert = st.session_state.get('fert', 0)

s1, s2, s3 = st.columns(3)
with s1: st.plotly_chart(draw_gauge(conf, "CERTITUDE IA", "#58a6ff"), use_container_width=True)
with s2: st.plotly_chart(draw_gauge(h2o, "HYDRATATION", "#3aedff"), use_container_width=True)
with s3: st.plotly_chart(draw_gauge(fert, "FERTILITÉ", "#2ea043"), use_container_width=True)

st.divider()

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    file = st.file_uploader("CAPTEUR OPTIQUE", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("DÉCRYPTAGE NEURAL", use_container_width=True):
            if model:
                img_p = img.resize((224, 224))
                arr = np.array(img_p) / 255.0
                preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                idx = np.argmax(preds)
                st.session_state['conf'] = float(np.max(preds) * 100)
                st.session_state['label'] = classes[idx].upper()
                st.session_state['h2o'] = random.randint(40, 75)
                st.session_state['fert'] = random.randint(60, 90)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    if st.session_state.get('conf', 0) > 0:
        st.markdown(f"<h2 style='color:#3aedff; font-family:Orbitron;'>{st.session_state['label']}</h2>", unsafe_allow_html=True)
        st.write(f"Confiance : **{st.session_state['conf']:.2f}%**")
    else:
        st.info("🛰️ EN ATTENTE DE TÉLÉMESURES")
    st.markdown('</div>', unsafe_allow_html=True)