import cv2
import numpy as np
import argparse

# Créer un parseur d'arguments
parser = argparse.ArgumentParser(description='Tester une image avec le modèle LBPH et enregistrer le résultat.')
parser.add_argument('input_image_path', type=str, help='Chemin de l\'image à tester')
parser.add_argument('output_image_path', type=str, help='Chemin pour enregistrer l\'image résultante')
args = parser.parse_args()

# Charger le modèle pré-entraîné
model_path = 'model.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Charger les chemins des images enregistrés
image_paths = np.load('image_paths.npy', allow_pickle=True)

# Charger l'image à détecter à partir de l'argument
input_image_path = args.input_image_path
image = cv2.imread(input_image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Si l'image est déjà un visage, on peut traiter directement l'image
face_roi = cv2.resize(gray_image, (92, 112))  # Redimensionner l'image si nécessaire
gray_image = face_roi
label, confidence = recognizer.predict(face_roi)

# Calculer le pourcentage de confiance
percentage = 100 - confidence
print(f"Label: {label}, Confiance: {percentage:.2f}%")

# Charger l'image correspondante de la base de données
best_match_image_path = image_paths[label]  # Obtenir le chemin de l'image correspondante
best_match_image = cv2.imread(best_match_image_path, cv2.IMREAD_GRAYSCALE)
best_match_image = cv2.resize(best_match_image, (92, 112))

# Ajouter une marge entre les images
space = 100  # Espacement en pixels
empty_space = np.ones((112, space, 3), dtype=np.uint8) * 255  # Espace vide blanc

# Annoter les images
annotated_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
annotated_match_image = cv2.cvtColor(best_match_image, cv2.COLOR_GRAY2BGR)

# Combiner les images avec l'espace entre elles
combined_image = np.hstack((annotated_image, empty_space, annotated_match_image))

# Ajouter le pourcentage de correspondance en dessous des images
text_height = 30  # Hauteur pour le texte
text_image = np.ones((text_height, combined_image.shape[1], 3), dtype=np.uint8) * 255  # Fond blanc pour le texte
cv2.putText(text_image, f'Correspondance: {percentage:.2f}%', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Empiler le texte sous les images
final_image = np.vstack((combined_image, text_image))

# Enregistrer l'image résultante à l'emplacement spécifié
cv2.imwrite(args.output_image_path, final_image)
