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

# distance euclédienne
def find_closest_match(test_vector, vectors):
    distances = np.linalg.norm(vectors - test_vector, axis=1)  # Distances euclidiennes
    min_index = np.argmin(distances)
    return min_index, distances[min_index]

# Fonction pour calculer la distance cosinus
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0  # Retourner 0 si l'un des vecteurs est nul
    return dot_product / (norm1 * norm2)

# Fonction pour trouver la correspondance la plus proche en utilisant la similarité cosinus
def find_closest_match_cosine(test_vector, vectors):
    similarities = [cosine_similarity(test_vector, vector) for vector in vectors]
    max_index = np.argmax(similarities)  # On cherche la similarité maximale
    return max_index, similarities[max_index]

# Fonction pour normaliser une distance euclidienne en un score (inverse)
def normalize_euclidean_distance(distance):
    return 1 / (1 + distance)

# Fonction pour calculer le score global en combinant les deux méthodes
def find_combined_match(test_vector, vectors, weight_euclidean=0.5, weight_cosine=0.5):
    scores = []
    for vector in vectors:
        # Distance euclidienne inversée
        euclidean_distance = np.linalg.norm(test_vector - vector)
        euclidean_score = normalize_euclidean_distance(euclidean_distance)

        # Similarité cosinus
        cosine_score = cosine_similarity(test_vector, vector)

        # Combinaison pondérée
        combined_score = (weight_euclidean * euclidean_score) + (weight_cosine * cosine_score)
        scores.append(combined_score)

    # Trouver l'indice avec le score combiné maximum
    max_index = np.argmax(scores)
    return max_index, scores[max_index]



# Charger les vecteurs moyens
names, average_vectors = load_average_vectors(output_dat_file)

# Calculer le vecteur LBPH pour l'image de test
test_image_path = args.input_image_path
test_vector = compute_lbph_vector(test_image_path)

# Trouver l'indice et la distance de la correspondance la plus proche
#closest_index, closest_distance = find_closest_match_cosine(test_vector, average_vectors)

# # Afficher le résultat
# threshold = 4.0  # Seuil à ajuster en fonction des performances souhaitées
# if closest_distance < threshold:
#     print(f"{names[closest_index]} : {closest_distance:.2f}")
# else:
#     print("Le visage testé n'appartient pas à la BDD.")

closest_index, combined_score = find_combined_match(test_vector, average_vectors, weight_euclidean=0.4, weight_cosine=0.6)

print(f"{names[closest_index]} : {combined_score:.2f}")