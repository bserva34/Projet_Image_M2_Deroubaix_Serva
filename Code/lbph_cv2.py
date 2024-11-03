import cv2
import os
import numpy as np

# Chemin vers le dossier contenant les images de formation
dataset_path = 'BDD/40_personnes_train'

# Initialiser le reconnaisseur LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialiser les listes pour stocker les images, leurs labels et les chemins des images
images = []
labels = []
image_paths = []  # Liste pour stocker les chemins des images

# Associer chaque dossier de personne à un label numérique
label = 0  # Initialisation du label

# Parcourir les dossiers de 40 personnes
for person_name in sorted(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for file_name in os.listdir(person_path):
            image_path = os.path.join(person_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(label)
                image_paths.append(image_path)  # Enregistrer le chemin de l'image
                label += 1 
        print(f"Données ajoutées pour {person_name} avec le label {label}")
         # Incrémenter le label pour la prochaine personne

# Vérifier si les listes ne sont pas vides
if len(images) == 0 or len(labels) == 0:
    print("Erreur : aucune donnée d'entraînement trouvée.")
else:
    # Convertir les listes en tableaux NumPy
    images = np.array(images)
    labels = np.array(labels)

    # Entraîner le modèle LBPH
    recognizer.train(images, labels)

    # Enregistrer le modèle
    model_path = 'model.yml'
    recognizer.save(model_path)

    print("Modèle LBPH entraîné avec succès.")
    # Enregistrer les chemins des images pour référence future
    np.save('image_paths.npy', image_paths)
