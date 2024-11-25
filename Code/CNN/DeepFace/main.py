import yaml
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace

# Chemin du fichier YAML et de l'image d'entrée
yaml_file = "labeled_faces.yml"  # Fichier YAML contenant les labels et les embeddings
input_image_path = "image.jpg"  # Chemin de l'image à tester

# Charger le fichier YAML
with open(yaml_file, "r") as file:
    labeled_faces = yaml.safe_load(file)

# Vérifier si le fichier YAML est vide
if not labeled_faces:
    print("Le fichier YAML est vide ou non valide.")
    exit()

# Extraire le vecteur caractéristique (embedding) de l'image d'entrée
try:
    embeddings = DeepFace.represent(
        img_path=input_image_path,
        model_name="Dlib",
        detector_backend="dlib",
        enforce_detection=True,
        align=True
    )
    input_embedding = embeddings[0]["embedding"]
except Exception as e:
    print(f"Erreur lors de l'extraction des caractéristiques de l'image : {e}")
    exit()

# Initialisation des variables pour le label et la distance minimale
closest_label = None
closest_image = None
min_distance = float("inf")

# Parcourir les labels et leurs vecteurs dans le fichier YAML
for label, entries in labeled_faces.items():
    for entry in entries:
        embedding = np.array(entry["vector"])
        distance = cosine(input_embedding, embedding)  # Calcul de la distance cosinus

        if distance < min_distance:
            min_distance = distance
            closest_label = label
            closest_image = entry["image"]

# Afficher le résultat
if closest_label is not None and closest_image is not None:
    print(f"Label le plus proche : {closest_label}")
    print(f"Image associée : {closest_image}")
    print(f"Distance cosinus : {min_distance}")
else:
    print("Aucune correspondance trouvée.")
