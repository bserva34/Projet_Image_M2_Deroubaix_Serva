import cv2
import os
import numpy as np
import argparse

# Argument pour fournir l'image à tester
parser = argparse.ArgumentParser(description="")
parser.add_argument("input_bdd_path", type=str, help="Dossier bdd")
args = parser.parse_args()

# Chemin vers le dossier contenant les images de formation
dataset_path = args.input_bdd_path
# Chemin vers le fichier de sortie pour les vecteurs caractéristiques moyens
output_dat_file = 'lbph_bdd.dat'


# Initialiser le reconnaisseur LBPH pour extraire les vecteurs caractéristiques
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Ouvrir le fichier pour écrire les vecteurs caractéristiques moyens
with open(output_dat_file, 'w') as dat_file:
    cpt = 0  # Compteur pour les dossiers
    for person_name in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            person_images = []  # Liste pour stocker les images de cette personne
            person_histograms = []  # Liste pour stocker les vecteurs caractéristiques

            for file_name in os.listdir(person_path):
                image_path = os.path.join(person_path, file_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Entraîner sur une image unique pour obtenir son histogramme
                    recognizer.train([image], np.array([0]))  # Label factice
                    hist = recognizer.getHistograms()[0]  # Extraire l'histogramme
                    person_histograms.append(hist)
                    person_images.append(image)
                    dat_file.write(f"{person_name} {' '.join(map(str, hist.flatten()))}\n")

            if person_histograms:
                # Calculer la moyenne des vecteurs caractéristiques pour cette personne
                average_histogram = np.mean(person_histograms, axis=0)

                # Enregistrer le vecteur moyen dans le fichier .dat
                #dat_file.write(f"{person_name} {' '.join(map(str, average_histogram.flatten()))}\n")
                print(f"Vecteur moyen enregistré pour {person_name} dans {output_dat_file}")

            cpt += 1  # Passer au dossier suivant

print("Tous les vecteurs caractéristiques moyens ont été enregistrés.")
