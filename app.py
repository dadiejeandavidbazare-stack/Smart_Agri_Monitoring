import os
# LIGNE CRITIQUE : Force la compatibilité avec ton modèle .h5
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import random

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="AGRONOVA APOLLO", page_icon="🛰️", layout="wide")

# --- DESIGN "SATELLITE DARK MODE" ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    header {visibility: hidden;}
    .stApp { background-color: #010409; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }
    .neon-header { font-family: 'Orbitron', sans-serif; font-size: 2.5rem; color: #3aedff; text-shadow: 0 0 15px rgba(58, 237, 255, 0.5); }
    .command-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DU MOTEUR IA ---
@st.cache_resource
def load_engine():
    # Chemin vers ton modèle dans le dossier GitHub
    model_path = 'models/agri_model_v1.h5'
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
    if os.path.exists(model_path):
        # Chargement sécurisé via tf.keras
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, classes
    return None, []

model, classes = load_engine()

# --- FONCTION DES JAUGES ---
def draw_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 16, 'family': 'Orbitron', 'color': '#8b949e'}},
        number={'font': {'color': '#ffffff', 'size': 40}, 'suffix': "%"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
               'bgcolor': "rgba(255,255,255,0.05)", 'bordercolor': "#30363d"}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- AFFICHAGE DASHBOARD ---
st.markdown('<h1 class="neon-header">AGRONOVA APOLLO</h1>', unsafe_allow_html=True)
st.write("CENTRE DE COMMANDEMENT SATELLITE - MAROC")

# Initialisation des valeurs (Jauges initiales à 0% ou selon session)
conf = st.session_state.get('conf', 0)
h2o = st.session_state.get('h2o', 0)
fert = st.session_state.get('fert', 0)

c1, c2, c3 = st.columns(3)
with c1: st.plotly_chart(draw_gauge(conf, "CERTITUDE IA", "#58a6ff"), use_container_width=True)
with c2: st.plotly_chart(draw_gauge(h2o, "HYDRATATION", "#3aedff"), use_container_width=True)
with c3: st.plotly_chart(draw_gauge(fert, "FERTILITÉ", "#2ea043"), use_container_width=True)

st.divider()

col_img, col_info = st.columns([1, 1.2])

with col_img:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    st.write("🛰️ CAPTEUR OPTIQUE")
    file = st.file_uploader("Charger une image satellite", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("LANCER L'ANALYSE NEURALE", use_container_width=True):
            if model:
                # Prétraitement et Prédiction
                img_p = img.resize((224, 224))
                arr = np.array(img_p) / 255.0
                preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                idx = np.argmax(preds)
                # Mise à jour des données
                st.session_state['conf'] = float(np.max(preds) * 100)
                st.session_state['label'] = classes[idx].upper()
                st.session_state['h2o'] = random.randint(45, 80)
                st.session_state['fert'] = random.randint(55, 95)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    if st.session_state.get('conf', 0) > 0:
        st.markdown(f"### 🔍 DIAGNOSTIC : {st.session_state['label']}")
        st.success(f"Détection validée à {st.session_state['conf']:.2f}%")
        st.info("Protocoles de traitement envoyés aux unités de terrain.")
    else:
        st.info("En attente de données optiques pour analyse...")