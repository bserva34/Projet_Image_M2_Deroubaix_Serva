import cv2
import os
import numpy as np

# Chemin vers le modèle sauvegardé
model_path = 'model.yml'
# Chemin vers les données d'entraînement pour relire les images
dataset_path = 'BDD/40_personnes_train'

# Charger le modèle LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Fonction pour calculer l'histogramme LBPH pour une image
def calculate_lbph_histogram(image, radius=1, neighbors=8, grid_x=8, grid_y=8):
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])  # Calcul de l'histogramme LBP
    lbp = cv2.normalize(lbp, lbp).flatten()  # Normaliser et aplatir l'histogramme
    return lbp

# Dictionnaire pour stocker les histogrammes pour chaque label
label_histograms = {}

# Calcul des histogrammes pour chaque image dans le dossier d'entraînement
for label, person_name in enumerate(sorted(os.listdir(dataset_path))):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        person_histograms = []

        for file_name in os.listdir(person_path):
            image_path = os.path.join(person_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Calculer l'histogramme LBPH pour chaque image
                hist = calculate_lbph_histogram(image)
                person_histograms.append(hist)

        # Calculer la moyenne des histogrammes pour le label
        if person_histograms:
            mean_histogram = np.mean(person_histograms, axis=0)
            label_histograms[label] = mean_histogram
            print(f"Histogramme moyen calculé pour le label {label} ({person_name})")

# Enregistrer les histogrammes moyens dans un fichier .dat
output_file = 'mean_histograms.dat'
with open(output_file, 'w') as f:
    for label, mean_histogram in label_histograms.items():
        histogram_str = ' '.join(map(str, mean_histogram))
        f.write(f"{label}: {histogram_str}\n")

print(f"Histogrammes moyens enregistrés dans {output_file}")
