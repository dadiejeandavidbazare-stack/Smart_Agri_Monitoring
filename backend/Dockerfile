# Image de base légère avec Python 3.11
FROM python:3.11-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code (l'API et le modèle .h5)
COPY . .

# Exposition du port de FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "app_ia_api:app", "--host", "0.0.0.0", "--port", "8000"]