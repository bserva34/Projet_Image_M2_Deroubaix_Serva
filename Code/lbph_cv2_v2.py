import cv2
import os
import numpy as np

# Chemin vers le dossier contenant les images de formation
dataset_path = 'BDD/40_personnes_train'
# Chemin vers le dossier où enregistrer les images moyennes
output_path = 'output_images_moyennes'
# Chemin vers le fichier de sortie pour les vecteurs caractéristiques moyens
output_dat_file = 'output_vectors_moyens.dat'

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_path):
    os.makedirs(output_path)

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
                    # Ajouter l'image à la liste de la personne
                    person_images.append(image)

                    # Extraire le vecteur caractéristique LBPH de l'image
                    hist = recognizer.compute(image)  # Utilisez `compute` pour obtenir l'histogramme
                    person_histograms.append(hist)

            if person_histograms:
                # Calculer la moyenne des vecteurs caractéristiques pour cette personne
                average_histogram = np.mean(person_histograms, axis=0)

                # Enregistrer le vecteur moyen dans le fichier .dat
                dat_file.write(f"{person_name} {average_histogram.tolist()}\n")
                print(f"Vecteur moyen enregistré pour {person_name} dans {output_dat_file}")

                # Créer et enregistrer l'image moyenne pour la personne (optionnel)
                average_image = np.mean(person_images, axis=0).astype(np.uint8)
                average_image_path = os.path.join(output_path, f"{cpt}.png")
                cv2.imwrite(average_image_path, average_image)
                print(f"Image moyenne créée pour {person_name} et enregistrée sous {average_image_path}")

            cpt += 1  # Passer au dossier suivant

print("Tous les vecteurs caractéristiques moyens ont été enregistrés.")
