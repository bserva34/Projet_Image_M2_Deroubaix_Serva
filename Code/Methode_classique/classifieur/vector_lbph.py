import cv2
import numpy as np
import argparse

# Configuration de l'argument parser
parser = argparse.ArgumentParser(description='Tester une image avec le modèle LBPH et enregistrer le résultat.')
parser.add_argument('input_image_path', type=str, help='Chemin de l\'image à tester')
args = parser.parse_args()

def compute_lbph_vector(image_path):
    # Initialisation du modèle LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("L'image ne peut pas être chargée.")
    
    # Entraîner le modèle sur une image unique
    recognizer.train([image], np.array([0]))
    
    # Extraire l'histogramme LBPH via le modèle entraîné
    hist = recognizer.getHistograms()[0]  # Prendre l'histogramm
    
    return hist

# Chemin de l'image de test
test_image_path = args.input_image_path

# Calculer le vecteur LBPH
test_vector = compute_lbph_vector(test_image_path)

output_file_path='acc_vector_lbph.dat'

# Écrire le vecteur dans un fichier
with open(output_file_path, 'w') as f:
    # Convertir le vecteur en une ligne de texte
    f.write(f"{' '.join(map(str, test_vector.flatten()))}\n")


print(f"Vecteur LBPH enregistré dans le fichier : {output_file_path}")
