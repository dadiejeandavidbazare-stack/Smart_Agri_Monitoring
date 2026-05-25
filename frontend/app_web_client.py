# frontend/app_web_client.py - Dashboard Sophistiqué - Agronova Apollo
import streamlit as st
import requests
from PIL import Image
import io

# 1. Configuration de la page avec un thème large et moderne
st.set_page_config(
    page_title="Agronova Apollo | Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Barre latérale (Sidebar) - Configuration & Statut Cloud
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/leaf.png", width=80)
    st.title("AGRONOVA APOLLO")
    st.subheader("Configuration Système")
    st.markdown("---")
    
    # Sélecteur d'environnement pour préparer ton intégration OpenCloud
    env = st.selectbox("Environnement réseau", ["Localhost (Développement)", "OpenCloud Compute Instance"])
    
    if env == "Localhost (Développement)":
        API_URL = "http://127.0.0.1:8000/predict"
        st.info("🔌 Connecté au serveur d'inférence local.")
    else:
        # Tu n'auras qu'à modifier cette IP une fois ta VM créée
        API_URL = "http://157.245.xx.xx:8000/predict"
        st.warning("☁️ Mode OpenCloud sélectionné (Vérifie l'IP du serveur).")
        
    st.markdown("---")
    st.markdown("**Version du modèle :** MobileNetV2 v3.14")
    st.markdown("**Capacité d'analyse :** 38 Catégories")
    st.caption("Développé pour l'analyse phytosanitaire distribuée.")

# 3. Corps Principal - Grille d'indicateurs (KPIs)
st.title("🌱 Tableau de Bord de Surveillance Agricole")
st.markdown("Système de vision artificielle appliqué à la détection précoce des maladies.")

# Ligne d'indicateurs globaux
kp1, kp2, kp3 = st.columns(3)
kp1.metric(label="Statut du Moteur IA", value="Opérationnel", delta="En ligne")
kp2.metric(label="Architecture", value="Distribuée", delta="API REST")
kp3.metric(label="Modèle Embarqué", value="MobileNetV2", delta="Keras 3")

st.markdown("---")

# 4. Zone de traitement (Découpage en 2 Colonnes : Import / Résultats)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📸 Captures de Terrain")
    st.write("Importez des images haute résolution issues de drones, de satellites ou de smartphones.")
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez l'image d'une feuille ici...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Utilisation de la syntaxe compatible avec ta version locale de Streamlit
        st.image(image, caption="Image phytosanitaire chargée", use_column_width=True)

with col_right:
    st.subheader("🔬 Diagnostic de l'Inférence IA")
    st.write("Les résultats de l'analyse matricielle s'afficheront ci-dessous après traitement.")
    
    if uploaded_file is not None:
        # Le bouton de déclenchement n'apparaît que si une image est présente
        if st.button("🚀 Lancer le diagnostic Apollo", type="primary", use_container_width=True):
            with st.spinner("Analyse des pixels par le réseau convolutif..."):
                try:
                    # Conversion de l'image en flux binaire
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                    
                    # Requête HTTP vers le Backend FastAPI
                    response = requests.post(API_URL, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("status") == "success":
                            st.success("Analyse matricielle terminée !")
                            
                            classe_detectee = result.get("class_name")
                            confiance = result.get("confidence", 0.0) * 100
                            
                            # Affichage stylisé des métriques de prédiction
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.info(f"**Pathologie / Statut :**\n\n### {classe_detectee}")
                            with res_col2:
                                st.metric(label="Indice de Confiance", value=f"{confiance:.2f} %")
                            
                            # Jauge visuelle de certitude
                            st.progress(int(confiance))
                            
                            # Section recommandations automatiques basiques
                            st.markdown("### 📋 Recommandation Agronomique")
                            if "Mildiou" in str(classe_detectee) or "Brûlure" in str(classe_detectee):
                                st.error("🚨 Alerte infection fongique détectée. Isoler la parcelle et limiter l'irrigation par aspersion.")
                            elif "Saine" in str(classe_detectee) or "Sain" in str(classe_detectee):
                                st.success("✅ Culture saine. Poursuivre le programme de surveillance standard.")
                            else:
                                st.warning("⚠️ Anomalie détectée. Inspecter visuellement la parcelle pour confirmer le stress de la plante.")
                                
                        else:
                            st.error(f"Erreur modèle : {result.get('message')}")
                    else:
                        st.error(f"Le serveur API a répondu avec le code : {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Liaison rompue avec le serveur IA. Vérifie que ton terminal Backend FastAPI est actif sur le port 8000.")
                except Exception as e:
                    st.error(f"Erreur système : {str(e)}")
    else:
        # Message d'attente neutre et propre
        st.info("💡 En attente d'une image de culture pour initialiser le traitement du réseau de neurones.")

st.markdown("---")
st.caption("📊 Agronova Apollo Pro v3.0 | Infrastructure distribuée & optimisée pour l'agrotechnologie.")