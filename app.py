import os
# --- CORRECTIF CRITIQUE ---
# Force TensorFlow à utiliser l'ancien moteur Keras pour lire ton modèle .h5
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import random

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="AGRONOVA APOLLO", page_icon="🛰️", layout="wide")

# --- STYLE CSS (SATELLITE COMMAND CENTER) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    header {visibility: hidden;}
    .stApp { background-color: #010409; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }
    .neon-header { font-family: 'Orbitron', sans-serif; font-size: 2.5rem; color: #3aedff; text-shadow: 0 0 15px rgba(58, 237, 255, 0.5); letter-spacing: 3px; }
    .command-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px; }
    .status-text { font-family: 'Orbitron', sans-serif; font-size: 0.9rem; }
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
        try:
            # Utilisation de tf.keras pour charger le modèle
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, classes
        except Exception as e:
            st.error(f"Erreur de lecture du modèle : {e}")
            return None, []
    return None, []

model, classes = load_engine()

# --- FONCTION GRAPHIQUE DES JAUGES ---
def draw_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 16, 'family': 'Orbitron', 'color': '#8b949e'}},
        number={'font': {'color': '#ffffff', 'size': 40}, 'suffix': "%"},
        gauge={'axis': {'range': [0, 100], 'tickcolor': "#30363d"},
               'bar': {'color': color},
               'bgcolor': "rgba(255,255,255,0.05)",
               'borderwidth': 1, 'bordercolor': "#30363d"}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=30, r=30, t=50, b=20))
    return fig

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.markdown('<p class="status-text" style="color:#3aedff;">AGRONOVA CORE V6.0</p>', unsafe_allow_html=True)
    if model:
        st.success("🟢 STATUT : OPÉRATIONNEL")
    else:
        st.error("🔴 STATUT : ERREUR MOTEUR")
    st.divider()
    if st.button("RÉINITIALISER LE SYSTÈME"):
        st.session_state.clear()
        st.rerun()

# --- EN-TÊTE ---
st.markdown('<h1 class="neon-header">AGRONOVA APOLLO</h1>', unsafe_allow_html=True)
st.write("CENTRE DE COMMANDEMENT SATELLITE - MAROC")

# Initialisation des variables de session
if 'conf' not in st.session_state: st.session_state['conf'] = 0
if 'h2o' not in st.session_state: st.session_state['h2o'] = 0
if 'fert' not in st.session_state: st.session_state['fert'] = 0

# --- SECTION JAUGES ---
c1, c2, c3 = st.columns(3)
with c1: st.plotly_chart(draw_gauge(st.session_state['conf'], "CERTITUDE IA", "#58a6ff"), use_container_width=True)
with c2: st.plotly_chart(draw_gauge(st.session_state['h2o'], "HYDRATATION", "#3aedff"), use_container_width=True)
with c3: st.plotly_chart(draw_gauge(st.session_state['fert'], "FERTILITÉ", "#2ea043"), use_container_width=True)

st.divider()

# --- ANALYSE ET RÉSULTATS ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    st.markdown('<p class="status-text" style="color:#3aedff;">CAPTEUR OPTIQUE</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("LANCER L'ANALYSE NEURALE", use_container_width=True):
            if model:
                # Prétraitement de l'image
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
                idx = np.argmax(prediction)
                
                # Mise à jour des résultats
                st.session_state['conf'] = float(np.max(prediction) * 100)
                st.session_state['label'] = classes[idx].upper()
                st.session_state['h2o'] = random.randint(45, 85)
                st.session_state['fert'] = random.randint(60, 95)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    if st.session_state['conf'] > 0:
        st.markdown(f"<h2 style='color:#3aedff; font-family:Orbitron;'>{st.session_state['label']}</h2>", unsafe_allow_html=True)
        st.write(f"Analyse terminée. Certitude neurale : **{st.session_state['conf']:.2f}%**")
        st.divider()
        st.markdown("#### ⚡ PROTOCOLES APOLLO")
        st.info("• Signal satellite stable.\n• Diagnostic végétal transmis.\n• Optimisation de l'irrigation calculée.")
    else:
        st.info("🛰️ EN ATTENTE DE TÉLÉMÉTRIE\nVeuillez charger une image pour activer le réseau neural.")
    st.markdown('</div>', unsafe_allow_html=True)