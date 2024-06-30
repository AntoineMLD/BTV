import streamlit as st
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ultralytics import YOLO
import cv2
import base64

# Fonction pour charger et configurer le modèle YOLO
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    model.to("cpu")  # Forcer l'utilisation du CPU
    return model

model = load_model()

# Fonction pour dessiner les prédictions sur l'image
def draw_predictions(image, predictions):
    for prediction in predictions:
        bbox = prediction['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        conf = prediction['confidence']
        cls = prediction['class']
        label = f"{cls} {conf:.2f}"
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Fonction pour redimensionner l'image
def resize_image(image, size=(640, 640)):
    return image.resize(size)

# Fonction de prédiction locale
def predict_locally(image):
    resized_image = resize_image(image)
    image_tensor = torch.tensor(np.array(resized_image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        results = model(image_tensor)

    predictions = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls = box.cls.item()
            predictions.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': int(cls)
            })
    return predictions

# Configuration des tentatives de nouvelle connexion pour les requêtes HTTP
def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Interface utilisateur Streamlit
st.title("Détection d'objets avec YOLO")
st.write("Téléchargez une image et obtenez les prédictions de détection d'objets.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    static_image = Image.open(uploaded_file).convert("RGB")
    st.image(static_image, caption='Image statique téléchargée.', use_column_width=True)

    # Convertir l'image en chaîne de caractères pour l'envoi
    buffered = BytesIO()
    static_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Tentative de prédiction via l'API Flask
    try:
        response = requests_retry_session().post('http://127.0.0.1:5000/predict', json={'image': img_str}, timeout=60)
        if response.status_code == 200:
            predictions = response.json()['predictions']
            st.write("Prédictions pour l'image statique (API) :")
            st.write(predictions)

            image_with_predictions = draw_predictions(np.array(static_image), predictions)
            st.image(image_with_predictions, caption='Image statique avec détections (API)', use_column_width=True)
        else:
            st.error(f"Erreur lors de la prédiction sur l'image statique: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        st.write("Tentative de prédiction locale...")
        predictions = predict_locally(static_image)

        st.write("Prédictions pour l'image statique (locale) :")
        st.write(predictions)

        image_with_predictions = draw_predictions(np.array(static_image), predictions)
        st.image(image_with_predictions, caption='Image statique avec détections (locale)', use_column_width=True)
