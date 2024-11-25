import yaml
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace
import os

# Chemins des fichiers et dossiers
yaml_file = "labeled_faces.yml"  # Fichier YAML contenant les labels et les embeddings
test_images_folder = "testCNN"  # Dossier contenant les images à tester
output_file = "resultatTestCNN.dat"  # Fichier pour stocker les résultats

# Charger le fichier YAML
with open(yaml_file, "r") as file:
    labeled_faces = yaml.safe_load(file)

# Vérifier si le fichier YAML est vide
if not labeled_faces:
    print("Le fichier YAML est vide ou non valide.")
    exit()

# Ouvrir le fichier de sortie en mode écriture
with open(output_file, "w") as output:
    # Parcourir toutes les images du dossier testCNN
    for image_file in os.listdir(test_images_folder):
        # Construire le chemin complet de l'image
        image_path = os.path.join(test_images_folder, image_file)

        # Vérifier que le fichier est bien une image
        if not (image_file.endswith(".jpg") or image_file.endswith(".png")):
            continue

        try:
            # Extraire le vecteur caractéristique (embedding) de l'image d'entrée
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name="Dlib",
                detector_backend="dlib",
                enforce_detection=True,
                align=True
            )
            input_embedding = embeddings[0]["embedding"]
        except Exception as e:
            print(f"Erreur lors de l'extraction des caractéristiques de l'image {image_file} : {e}")
            continue

        # Initialisation des variables pour le label et la distance minimale
        closest_label = None
        closest_image = None
        min_distance = float("inf")
        closest_label_count = 0  # Nombre d'images pour le label trouvé

        # Parcourir les labels et leurs vecteurs dans le fichier YAML
        for label, entries in labeled_faces.items():
            for entry in entries:
                embedding = np.array(entry["vector"])
                distance = cosine(input_embedding, embedding)  # Calcul de la distance cosinus

                if distance < min_distance:
                    min_distance = distance
                    closest_label = label
                    closest_image = entry["image"]
                    closest_label_count = len(entries)  # Mettre à jour le nombre d'images pour ce label

        # Enregistrer le résultat dans le fichier de sortie
        if closest_label is not None and closest_image is not None:
            output.write(f"{image_file} {closest_label} {min_distance:.4f} {closest_image} {closest_label_count}\n")
            print(f"Résultat ajouté pour {image_file}: Label={closest_label}, Image correspondante={closest_image}, Distance={min_distance:.4f}, Nombre d'images={closest_label_count}")
        else:
            output.write(f"{image_file} Aucune_correspondance -1\n")
            print(f"Aucune correspondance trouvée pour {image_file}")
