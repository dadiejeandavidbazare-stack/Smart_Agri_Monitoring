import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les alertes inutiles de TF
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. CHEMINS
MODEL_PATH = 'models/agri_model_v1.h5'
DATA_DIR = 'data/raw/plantvillage dataset/color'

def run_test():
    try:
        # 2. CHARGEMENT
        print("🟢 Étape 1 : Chargement du modèle...")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print("🟢 Étape 2 : Récupération des noms de maladies...")
        class_names = sorted(os.listdir(DATA_DIR))
        
        # 3. CHOIX D'UNE IMAGE (On prend une image de pomme saine par exemple)
        # On cherche le dossier 'Apple___healthy' ou un autre présent
        test_folder = class_names[0] 
        img_name = os.listdir(os.path.join(DATA_DIR, test_folder))[0]
        img_path = os.path.join(DATA_DIR, test_folder, img_name)
        print(f"🟢 Étape 3 : Analyse de l'image : {img_path}")

        # 4. PRÉTRAITEMENT
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 5. PRÉDICTION
        prediction = model.predict(img_array, verbose=0)
        idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        print("-" * 30)
        print(f"🎯 RÉSULTAT : {class_names[idx]}")
        print(f"📊 CONFIANCE : {confidence:.2f}%")
        print("-" * 30)

        # 6. AFFICHAGE (Optionnel, juste pour confirmer visuellement)
        plt.imshow(img)
        plt.title(f"IA Result: {class_names[idx]}")
        plt.axis('off')
        print("📸 Ferme la fenêtre de l'image pour terminer le script.")
        plt.show()

    except Exception as e:
        print(f"❌ ERREUR : {e}")

if __name__ == "__main__":
    run_test()