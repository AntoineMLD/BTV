import torch

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Effectuer une détection sur une image d'exemple
img = 'https://ultralytics.com/images/zidane.jpg'  # ou le chemin vers une image locale
results = model(img)

# Afficher les résultats
results.show()  # ou results.save() pour enregistrer l'image avec les prédictions
