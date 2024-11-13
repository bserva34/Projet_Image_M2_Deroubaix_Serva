import cv2
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Tester une image avec le modèle LBPH et enregistrer le résultat.')
parser.add_argument('input_image_path', type=str, help='Chemin de l\'image à tester')
args = parser.parse_args()

# Chemin vers le fichier de vecteurs moyens
output_dat_file = 'output_vectors_moyens.dat'

# Fonction pour charger les vecteurs moyens depuis le fichier
# Fonction pour charger les vecteurs moyens depuis le fichier
def load_average_vectors(file_path):
    vectors = []
    names = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            name = parts[0]  # Le nom est le premier élément
            
            # Enlever les crochets, les virgules et les caractères vides, puis convertir en float
            vector = np.array([float(x.replace('[', '').replace(']', '').replace(',', '')) for x in parts[1:] if x])  
            names.append(name)
            vectors.append(vector)
    
    print(f"Vecteurs moyens chargés : {len(vectors)} vecteurs trouvés")
    return names, vectors


# Fonction pour obtenir le vecteur LBPH d'une image
def compute_lbph_vector(image_path):
    #recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=1, grid_y=1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("L'image ne peut pas être chargée.")
    
    # Entraîner le modèle sur une image unique
    recognizer.train([image], np.array([0]))
    
    # Extraire l'histogramme LBPH via le modèle "trained"
    hist = recognizer.getHistograms()[0]  # Prendre l'histogramme
    hist_final = np.array(hist).flatten()
    
    #print(f"Taille du vecteur LBPH : {hist_final.shape}")
    return hist_final


# Fonction pour calculer la distance euclidienne et trouver l'indice le plus proche
def find_closest_match(test_vector, vectors):
    min_distance = float('inf')
    min_index = -1
    for i, vector in enumerate(vectors):
        distance = np.linalg.norm(test_vector - vector)  # Calculer la distance euclidienne
        # test_vector = np.array(test_vector, dtype=np.float32)
        # vector = np.array(vector, dtype=np.float32) 
        #distance = cv2.compareHist(test_vector, vector, cv2.HISTCMP_CHISQR)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index, min_distance

# Chemin de l'image à tester
test_image_path = args.input_image_path

# Charger les vecteurs moyens
names, average_vectors = load_average_vectors(output_dat_file)

# Calculer le vecteur LBPH pour l'image de test
test_vector = compute_lbph_vector(test_image_path)

# Trouver l'indice et la distance de la correspondance la plus proche
closest_index, closest_distance = find_closest_match(test_vector, average_vectors)

# Afficher le résultat
if(closest_distance < 4.):
    print(f"La correspondance la plus proche est : {names[closest_index]} avec une distance de {closest_distance}")
else : 
    print("Le visage testé n'appartient pas à la BDD")
