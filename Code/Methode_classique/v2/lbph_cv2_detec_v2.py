import cv2
import numpy as np
import os
import argparse

# Argument pour fournir l'image à tester
parser = argparse.ArgumentParser(description="Tester une image avec le modèle LBPH et comparer aux vecteurs moyens.")
parser.add_argument("input_image_path", type=str, help="Chemin de l'image à tester")
args = parser.parse_args()

# Chemin vers le fichier de vecteurs moyens
output_dat_file = "lbph_bdd.dat"

# Fonction pour charger les vecteurs moyens depuis le fichier
def load_average_vectors(file_path):
    vectors = []
    names = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            name = parts[0]  # Le label est le premier élément
            
            # Convertir les vecteurs (après le label) en tableau NumPy
            vector = np.array([float(x) for x in parts[1:]])
            names.append(name)
            vectors.append(vector)
    
    #print(f"Vecteurs moyens chargés : {len(vectors)} vecteurs trouvés")
    return names, np.array(vectors)

# Fonction pour obtenir le vecteur LBPH d'une image
def compute_lbph_vector(image_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("L'image ne peut pas être chargée.")
    
    # Entraîner le modèle sur une image unique pour extraire son histogramme
    recognizer.train([image], np.array([0]))
    hist = recognizer.getHistograms()[0]
    return np.array(hist).flatten()

# Fonction pour calculer la distance euclidienne et trouver la correspondance la plus proche
def find_closest_match(test_vector, vectors):
    distances = np.linalg.norm(vectors - test_vector, axis=1)  # Distances euclidiennes
    min_index = np.argmin(distances)
    return min_index, distances[min_index]

# Charger les vecteurs moyens
names, average_vectors = load_average_vectors(output_dat_file)

# Calculer le vecteur LBPH pour l'image de test
test_image_path = args.input_image_path
test_vector = compute_lbph_vector(test_image_path)

# Trouver l'indice et la distance de la correspondance la plus proche
closest_index, closest_distance = find_closest_match(test_vector, average_vectors)

# Afficher le résultat
threshold = 4.0  # Seuil à ajuster en fonction des performances souhaitées
if closest_distance < threshold:
    print(f"{names[closest_index]} : {closest_distance:.2f}")
else:
    print("Le visage testé n'appartient pas à la BDD.")
