from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import base64
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

print("Chargement du modèle...")
model = YOLO('best.pt')
model.to("cpu")
print("Modèle chargé avec succès")

@app.route('/', methods=['GET'])
def home():
    app.logger.info("Route principale accédée")
    return "API is running", 200

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Requête de prédiction reçue")
    try:
        data = request.json
        if not data or 'image' not in data:
            raise ValueError("Données d'image manquantes dans la requête")
        
        app.logger.info(f"Données reçues : {len(data['image'])} caractères")
        
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Impossible de décoder l'image")
        
        results = model(image)
        
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
        
        app.logger.info(f"Prédiction terminée. Nombre de prédictions: {len(predictions)}")
        return jsonify({"predictions": predictions})
    except Exception as e:
        app.logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    app.logger.info("Route de test accédée")
    return jsonify({"message": "Test successful"}), 200

if __name__ == '__main__':
    app.logger.info("Démarrage du serveur API sur http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
