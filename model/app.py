from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
from dotenv import load_dotenv
import logging
import traceback
import time
import psutil
import shutil


load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



# Définir le chemin de sauvegarde
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection_results')
os.makedirs(SAVE_DIR, exist_ok=True)
app.logger.info(f"Chemin complet du répertoire de sauvegarde : {os.path.abspath(SAVE_DIR)}")

# Charger le modèle YOLO
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas.")
model = YOLO(model_path)

# Forcer l'utilisation du CPU
device = "cpu"
model.to(device)

# Configurer les options de sauvegarde
save_options = {
    'project': SAVE_DIR,
    'name': 'detect',
    'save': True,
    'save_txt': True,
    'save_conf': True,
}

app.logger.info(f"Utilisation du device: {device}")
app.logger.info(f"Classes du modèle : {model.names}")

AZURE_MAPS_ACCOUNT_KEY = os.getenv('AZURE_MAPS_ACCOUNT_KEY')



@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Une erreur non gérée s'est produite : %s", traceback.format_exc())
    return jsonify(error=str(e)), 500

@app.route('/')
def index():
    app.logger.info("Page d'accueil demandée")
    return render_template('index.html', account_key=AZURE_MAPS_ACCOUNT_KEY)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        app.logger.info("Début de la détection")
        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        app.logger.info(f"Taille de l'image reçue : {img.shape}")
        app.logger.info(f"Utilisation mémoire actuelle : {psutil.virtual_memory().percent}%")
        app.logger.info(f"Utilisation CPU actuelle : {psutil.cpu_percent()}%")

        # Sauvegarder l'image capturée pour vérification
        capture_path = os.path.join(SAVE_DIR, f'capture_{int(time.time())}.jpg')
        cv2.imwrite(capture_path, img)
        app.logger.info(f"Image capturée sauvegardée sous : {capture_path}")

        results = model.predict(img, device=device, conf=0.25, iou=0.45, **save_options)
        
        app.logger.info(f"Résultats bruts : {results}")
        
        pool_count = 0
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            pool_count = len(results[0].boxes)
            app.logger.info(f"Nombre de piscines détectées : {pool_count}")
            annotated_image = results[0].plot()
            
            # Sauvegarder l'image annotée
            result_path = os.path.join(SAVE_DIR, f'result_{int(time.time())}.jpg')
            cv2.imwrite(result_path, annotated_image)
            app.logger.info(f"Image résultat sauvegardée sous : {result_path}")
            
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
        else:
            app.logger.info("Aucune piscine détectée")
            img_str = data['image']

        app.logger.info("Détection terminée avec succès")
        return jsonify({
            'pool_count': pool_count,
            'image': f"data:image/jpeg;base64,{img_str}"
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la détection : {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify(error=str(e)), 500
    
@app.route('/delete_images', methods=['POST'])
def delete_images():
    try:
        app.logger.info("Suppression des images sauvegardées")
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
            os.makedirs(SAVE_DIR)
        return jsonify(success=True)
    except Exception as e:
        app.logger.error(f"Erreur lors de la suppression des images : {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.logger.info("Juste avant de lancer l'application")
    try:
        app.logger.info("Tentative de démarrage de l'application")
        app.run(debug=False, port=5001)
    except Exception as e:
        app.logger.error(f"Erreur lors du démarrage de l'application: {str(e)}")
        app.logger.error(traceback.format_exc())
    