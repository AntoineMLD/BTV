import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Charger le modèle
model = YOLO('models/modelbest.pt')
score_image_path = 'images/summary.png'

# Fonction pour faire une prédiction et dessiner les boîtes de délimitation
def predict_and_draw_boxes(image_path):
    results = model.predict(image_path)
    
    # Charger l'image avec OpenCV
    image_cv2 = cv2.imread(image_path)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = int(box.cls)
            confidence = float(box.conf)

            # Dessiner la boîte sur l'image
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv2, f'{label}: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image_cv2, results

# Interface utilisateur Streamlit
st.title("Démonstration du modèle de Computer Vision Pool Detection")
# st.image("images/logo.png", width=200)  # Insérer l'URL de ton logo ici
st.write("Téléchargez une image pour obtenir une prédiction de piscines de qualité")

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    st.write("Classification en cours...")
    image_path = "temp_image.png"
    image.save(image_path)

    image_with_boxes, results = predict_and_draw_boxes(image_path)

    st.image(image_with_boxes, caption='Image avec prédictions', use_column_width=True)


    # Afficher l'image des scores mAP
    score_image = Image.open(score_image_path)
    st.image(score_image, caption='Résumé des métriques du modèle', use_column_width=True)





