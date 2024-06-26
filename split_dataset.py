import os
import shutil
import random

# Chemins des dossiers d'images et de labels
train_images_path = 'dataset/train/images'
train_labels_path = 'dataset/train/labels'

# Chemins des nouveaux dossiers
train_images_output = 'dataset/train_split/images/train'
val_images_output = 'dataset/train_split/images/val'
train_labels_output = 'dataset/train_split/labels/train'
val_labels_output = 'dataset/train_split/labels/val'

# Création des dossiers s'ils n'existent pas déjà
os.makedirs(train_images_output, exist_ok=True)
os.makedirs(val_images_output, exist_ok=True)
os.makedirs(train_labels_output, exist_ok=True)
os.makedirs(val_labels_output, exist_ok=True)

# Ratio de division
val_ratio = 0.2

# Récupération de toutes les images
images = [f for f in os.listdir(train_images_path) if os.path.isfile(os.path.join(train_images_path, f))]

# Vérification que chaque image a un fichier de label correspondant
images_with_labels = [img for img in images if os.path.isfile(os.path.join(train_labels_path, os.path.splitext(img)[0] + '.txt'))]

# Mélange des images
random.shuffle(images_with_labels)

# Calcul du nombre d'images pour le jeu de validation
num_val_images = int(len(images_with_labels) * val_ratio)

# Division des images
val_images = images_with_labels[:num_val_images]
train_images = images_with_labels[num_val_images:]

def move_files(image_list, source_img_path, source_lbl_path, dest_img_path, dest_lbl_path):
    for image in image_list:
        # Déplacer l'image
        shutil.copy(os.path.join(source_img_path, image), os.path.join(dest_img_path, image))
        
        # Déplacer le label correspondant
        label = os.path.splitext(image)[0] + '.txt'
        shutil.copy(os.path.join(source_lbl_path, label), os.path.join(dest_lbl_path, label))

# Déplacer les fichiers vers les dossiers de train et de validation
move_files(train_images, train_images_path, train_labels_path, train_images_output, train_labels_output)
move_files(val_images, train_images_path, train_labels_path, val_images_output, val_labels_output)

print(f"Nombre d'images de train : {len(train_images)}")
print(f"Nombre d'images de validation : {len(val_images)}")
