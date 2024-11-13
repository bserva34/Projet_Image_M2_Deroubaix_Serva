from deepface import DeepFace
import cv2
import os
import yaml

# Chemins des dossiers d'entrée et du fichier YAML
input_folder = "to_crop"  # Dossier contenant les images
output_yaml = "labeled_faces.yml"  # Fichier YAML pour stocker les labels et vecteurs

# Créer le fichier YAML s'il n'existe pas
if not os.path.exists(output_yaml):
    with open(output_yaml, 'w') as f:
        yaml.dump({}, f)

# Charger le contenu actuel du fichier YAML
with open(output_yaml, 'r') as f:
    labeled_faces = yaml.safe_load(f) or {}

# Définir la marge autour du visage en pourcentage (35% pour agrandir le cadre)
margin_percentage = 0.0

# Taille d'affichage fixe pour cv2.imshow
display_size = (640, 640)

# Initialiser la fenêtre pour éviter les bugs
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", *display_size)

# Parcourir chaque image dans le dossier d'entrée
for image_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, image_name)
    # Lire l'image
    image = cv2.imread(img_path)
    if image is None:
        continue  # Passer les fichiers qui ne sont pas des images

    # Utiliser DeepFace pour détecter les visages
    try:
        faces = DeepFace.extract_faces(img_path, detector_backend='dlib', enforce_detection=True, align=False)
    except Exception as e:
        print(f"Erreur de détection sur {image_name}: {e}")
        continue

    # Parcourir chaque visage détecté
    for i, face in enumerate(faces):
        # Obtenir les coordonnées de `facial_area`
        x, y, face_width, face_height = face["facial_area"]['x'], face["facial_area"]['y'], face["facial_area"]['w'], face["facial_area"]['h']

        # Calculer la marge en pixels
        margin_x = int(face_width * margin_percentage)
        margin_y = int(face_height * margin_percentage)

        # Ajouter la marge autour du visage
        y1 = max(0, y - margin_y)
        y2 = min(image.shape[0], y + face_height + margin_y)
        x1 = max(0, x - margin_x)
        x2 = min(image.shape[1], x + face_width + margin_x)

        # Créer une image adaptée pour un affichage sans déformation
        h, w = image.shape[:2]
        scale = min(display_size[0] / w, display_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        canvas = cv2.copyMakeBorder(
            resized_image,
            (display_size[1] - new_h) // 2,
            (display_size[1] - new_h + 1) // 2,
            (display_size[0] - new_w) // 2,
            (display_size[0] - new_w + 1) // 2,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        # Ajuster les coordonnées du rectangle au nouvel espace
        x1_resized = int(x1 * scale + (display_size[0] - new_w) // 2)
        y1_resized = int(y1 * scale + (display_size[1] - new_h) // 2)
        x2_resized = int(x2 * scale + (display_size[0] - new_w) // 2)
        y2_resized = int(y2 * scale + (display_size[1] - new_h) // 2)

        # Dessiner le rectangle autour du visage
        cv2.rectangle(canvas, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 2)

        # Afficher l'image redimensionnée
        cv2.imshow("Image", canvas)
        cv2.waitKey(1)  # Nécessaire pour l'affichage dans certaines configurations

        # Demander un label pour le visage
        label = input(f"Entrez le label pour le visage détecté dans {image_name} (laisser vide pour ignorer, 'quit' pour quitter) : ").strip()

        # Si le label est 'quit', sauvegarder le YAML et quitter le programme
        if label.lower() == 'quit':
            with open(output_yaml, 'w') as f:
                yaml.dump(labeled_faces, f, default_flow_style=False)
            print("Données labellisées enregistrées et programme terminé.")
            cv2.destroyAllWindows()
            exit()

        # Si le label est vide, ignorer ce visage
        if not label:
            print("Visage ignoré.")
            continue

        # Obtenir le vecteur caractéristique pour le visage détecté
        try:
            cropped_face = image[y1:y2, x1:x2]
            embedding = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding pour {image_name}: {e}")
            continue

        # Enregistrer le visage et le label dans le fichier YAML
        if label not in labeled_faces:
            labeled_faces[label] = []

        # Vérifier si l'image et le vecteur existent déjà pour éviter les doublons
        new_entry = {
            "image": image_name,
            "vector": embedding
        }
        if new_entry not in labeled_faces[label]:
            labeled_faces[label].append(new_entry)

# Enregistrer les données labellisées dans le fichier YAML
with open(output_yaml, 'w') as f:
    yaml.dump(labeled_faces, f, default_flow_style=False)

cv2.destroyAllWindows()
print(f"Données labellisées enregistrées dans {output_yaml}.")
