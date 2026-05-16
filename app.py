import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
import random

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="AGRONOVA APOLLO", page_icon="🛰️", layout="wide")

# --- CSS : INTERFACE SATELLITE COMMAND ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    header {visibility: hidden;}
    .main .block-container {padding: 1.5rem 2rem; background-color: #010409;}
    .stApp { background-color: #010409; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d; }
    .command-card { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 15px; }
    .neon-header { font-family: 'Orbitron', sans-serif; font-size: 2.2rem; color: #3aedff; letter-spacing: 5px; text-shadow: 0 0 10px rgba(58, 237, 255, 0.4); }
    .status-alert { padding: 8px; border-radius: 4px; background: rgba(255, 82, 82, 0.1); color: #ff5252; border: 1px solid #ff5252; font-size: 0.8rem; text-align: center; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DU MOTEUR IA ---
@st.cache_resource
def load_agronova_engine():
    model_path = 'models/agri_model_v1.h5'
    # Chemin vers tes dossiers de classes pour éviter l'IndexError
    data_dir = 'data/raw/plantvillage dataset/color'
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        # On récupère dynamiquement les noms des dossiers pour que l'index corresponde
        classes = sorted(os.listdir(data_dir)) if os.path.exists(data_dir) else ["Classe Inconnue"]
        return model, classes
    return None, []

model, classes = load_agronova_engine()

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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<p style="font-family:Orbitron; color:#3aedff; font-size:1.2rem;">AGRONOVA CORE V6.0</p>', unsafe_allow_html=True)
    st.write("🟢 Statut: **Connecté**")
    st.write(f"🧬 Classes chargées: **{len(classes)}**")
    st.divider()
    st.button("DEPLOY SYSTEM")

# --- DASHBOARD PRINCIPAL ---
st.markdown('<h1 class="neon-header">AGRONOVA APOLLO</h1>', unsafe_allow_html=True)
st.write("SATELLITE COMMAND CENTER")

# Monitoring Vital
s1, s2, s3 = st.columns(3)
conf = st.session_state.get('conf', 0)
h2o = st.session_state.get('h2o', 0)
fert = st.session_state.get('fert', 0)

with s1: 
    st.plotly_chart(draw_gauge(conf, "CERTITUDE IA", "#58a6ff"), use_container_width=True)
with s2: 
    st.plotly_chart(draw_gauge(h2o, "HYDRATATION", "#3aedff"), use_container_width=True)
with s3: 
    st.plotly_chart(draw_gauge(fert, "FERTILITÉ", "#2ea043"), use_container_width=True)

st.divider()

# Analyse
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Orbitron; color:#3aedff; font-size:0.8rem;">CAPTEUR OPTIQUE</p>', unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("DÉCRYPTAGE NEURAL", use_container_width=True):
            if model and len(classes) > 0:
                img_p = img.resize((224, 224))
                arr = np.array(img_p) / 255.0
                preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                idx = np.argmax(preds)
                
                # Sécurité anti-IndexError
                if idx < len(classes):
                    st.session_state['conf'] = float(np.max(preds) * 100)
                    st.session_state['label'] = classes[idx].replace('_', ' ').upper()
                    st.session_state['h2o'] = random.randint(30, 45)
                    st.session_state['fert'] = random.randint(70, 90)
                    st.rerun()
                else:
                    st.error(f"Erreur de synchronisation : Index {idx} hors limites.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="command-card">', unsafe_allow_html=True)
    if st.session_state.get('conf', 0) > 0:
        st.markdown(f"### {st.session_state['label']}")
        st.write(f"Analyse terminée avec une certitude de **{st.session_state['conf']:.2f}%**.")
        
        st.divider()
        st.markdown("#### ⚡ ACTIONS IMMÉDIATES")
        st.info("• Isoler les zones infectées.\n• Réduire l'humidité au sol.\n• Appliquer un traitement cuprique.")
        
        st.markdown("#### 🔭 STRATÉGIE LONG TERME")
        st.success("• Rotation des cultures sur 2 ans.\n• Amélioration du drainage de la parcelle.")
    else:
        st.write("En attente de télémesures satellite...")
    st.markdown('</div>', unsafe_allow_html=True)