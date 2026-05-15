import tensorflow as tf
import numpy as np
import os

def load_trained_model(model_path):
    """Charge le modèle sauvegardé."""
    return tf.keras.models.load_model(model_path)

def get_class_names(data_dir):
    """Récupère et trie les noms des maladies."""
    return sorted(os.listdir(data_dir))

def make_prediction(model, img_path, class_names):
    """Prend une image et retourne le diagnostic et la confiance."""
    # Chargement et prétraitement standard
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prédiction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100
    
    return class_names[class_idx], confidence