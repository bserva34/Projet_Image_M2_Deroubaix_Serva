import cv2
import os
import numpy as np

# Chemin vers le dossier contenant les images de formation
dataset_path = 'BDD/40_personnes_train'

# Initialiser le reconnaisseur LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialiser les listes pour stocker les images et leurs labels
images = []
labels = []

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
        print(f"Données ajoutées pour {person_name} avec le label {label}")
        label += 1  # Incrémenter le label pour la prochaine personne

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
