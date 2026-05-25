import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Agronova Apollo - API d'Inférence IA")

# Configuration CORS pour permettre au Frontend de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet les connexions de n'importe où sur le Cloud
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Dictionnaire de correspondance des pathologies (À adapter selon ton dataset)
LES_CLASSES = {
    0: "Feuille Saine",
    1: "Mildiou (Mildew)",
    2: "Rouille (Rust)",
    3: "Oïdium",
    4: "Tavelure",
    5: "Pourriture Grise (Botrytis)",
    6: "Flétrissement Bactérien",
    7: "Strie Bactérienne / Cercosporiose (Classe 7)"  # Celle détectée sur ta capture !
}

# 2. Chargement du modèle MobileNetV2
MODEL_PATH = "model_mobilenetv2_plants.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modèle {MODEL_PATH} chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "status": "Opérationnel",
        "moteur_ia": "MobileNetV2",
        "environnement": "Cloud Évolutif"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Le modèle IA n'est pas disponible sur le serveur."}
    
    try:
        # Lire l'image envoyée par le client Streamlit
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Prétraitement de l'image (MobileNetV2 prend du 224x224)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0  # Normalisation [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension de batch
        
        # Faire la prédiction
        predictions = model.predict(img_array)
        predicted_class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        
        # Récupérer le nom de la maladie
        nom_pathologie = LES_CLASSES.get(predicted_class_id, f"Pathologie Inconnue (ID: {predicted_class_id})")
        
        return {
            "class_id": str(predicted_class_id),
            "pathologie": nom_pathologie,
            "confidence": f"{confidence:.2f} %"
        }
        
    except Exception as e:
        return {"error": f"Erreur lors du traitement de l'image : {str(e)}"}

# 3. Lancement du serveur avec adaptation dynamique pour l'Open Cloud
if __name__ == "__main__":
    # Render ou toute autre plateforme attribue un PORT dynamiquement
    port = int(os.environ.get("PORT", 8000))
    # On écoute sur 0.0.0.0 pour être visible depuis l'extérieur du réseau local
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)