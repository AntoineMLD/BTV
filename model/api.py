from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os

app = FastAPI()

# Charger le modèle YOLO
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Erreur: le fichier modèle {model_path} n'existe pas.")
else:
    model = YOLO(model_path)
    model.to("cpu")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire l'image depuis le contenu binaire
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Faire une prédiction sur l'image
        results = model.predict(image, device="cpu")

        # Vérifier que les résultats contiennent des boîtes englobantes
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            # Convertir l'image avec les détections en bytes
            annotated_img_bytes = cv2.imencode('.jpg', results[0].plot())[1].tobytes()

            # Renvoyer l'image annotée en réponse
            return StreamingResponse(BytesIO(annotated_img_bytes), media_type="image/jpeg")
        else:
            raise HTTPException(status_code=500, detail="Aucune boîte englobante détectée dans les résultats.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
