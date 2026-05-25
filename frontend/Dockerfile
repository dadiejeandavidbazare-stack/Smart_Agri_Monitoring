# Image de base légère avec Python 3.11
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposition du port de Streamlit
EXPOSE 8501

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app_web_client.py", "--server.port", "8501", "--server.address", "0.0.0.0"]