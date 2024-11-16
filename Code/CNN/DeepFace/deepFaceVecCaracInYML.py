from deepface import DeepFace
import cv2
import os
import yaml

# Chemins des dossiers
input_folder = "to_crop"
processed_folder = "image_traitée"
faces_folder = "image_visage"
output_yaml = "labeled_faces.yml"

# Créer les dossiers si nécessaires
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(faces_folder, exist_ok=True)

# Charger ou initialiser le fichier YAML
if not os.path.exists(output_yaml):
    with open(output_yaml, 'w') as f:
        yaml.dump({}, f)
with open(output_yaml, 'r') as f:
    labeled_faces = yaml.safe_load(f) or {}

# Définir la marge autour du visage en pourcentage
margin_percentage = 0.0
display_size = (640, 640)

# Fenêtre OpenCV
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", *display_size)

# Fonction pour compter les occurrences d'un label dans le YAML
def count_label_occurrences(label):
    return len(labeled_faces.get(label, []))

# Parcourir chaque image dans le dossier d'entrée
for image_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, image_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Détection des visages avec DeepFace
    try:
        faces = DeepFace.extract_faces(img_path, detector_backend='dlib', enforce_detection=True, align=False)
    except Exception as e:
        print(f"Erreur de détection sur {image_name}: {e}")
        continue

    # Préparer une copie pour l'image finale avec rectangles des visages labellisés
    final_image = image.copy()
    labels_for_image = []

    for i, face in enumerate(faces):
        x, y, face_width, face_height = face["facial_area"]['x'], face["facial_area"]['y'], face["facial_area"]['w'], face["facial_area"]['h']
        margin_x = int(face_width * margin_percentage)
        margin_y = int(face_height * margin_percentage)
        y1, y2 = max(0, y - margin_y), min(image.shape[0], y + face_height + margin_y)
        x1, x2 = max(0, x - margin_x), min(image.shape[1], x + face_width + margin_x)

        # Redimensionner et afficher
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
        x1_resized = int(x1 * scale + (display_size[0] - new_w) // 2)
        y1_resized = int(y1 * scale + (display_size[1] - new_h) // 2)
        x2_resized = int(x2 * scale + (display_size[0] - new_w) // 2)
        y2_resized = int(y2 * scale + (display_size[1] - new_h) // 2)

        cv2.rectangle(canvas, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 2)
        cv2.imshow("Image", canvas)
        cv2.waitKey(1)

        # Label pour ce visage
        label = input(f"Entrez le label pour le visage détecté dans {image_name} (laisser vide pour ignorer, 'quit' pour quitter) : ").strip()
        if label.lower() == 'quit':
            with open(output_yaml, 'w') as f:
                yaml.dump(labeled_faces, f, default_flow_style=False)
            cv2.destroyAllWindows()
            print("Programme terminé.")
            exit()
        if not label:
            print("Visage ignoré.")
            continue

        # Générer l'embedding
        try:
            cropped_face = image[y1:y2, x1:x2]
            embedding = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding pour {image_name}: {e}")
            continue

        # Ajouter au YAML
        if label not in labeled_faces:
            labeled_faces[label] = []
        new_entry = {"image": image_name, "vector": embedding}
        if new_entry not in labeled_faces[label]:
            labeled_faces[label].append(new_entry)
        labels_for_image.append(f"{label}{count_label_occurrences(label) + 1}")

        # Dessiner le rectangle pour l'image finale
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Renommer et déplacer les fichiers
    if labels_for_image:
        new_name = "_".join(labels_for_image) + ".jpg"
        processed_path = os.path.join(processed_folder, new_name)
        faces_path = os.path.join(faces_folder, new_name)

        # Déplacer l'image traitée
        os.rename(img_path, processed_path)

        # Sauvegarder l'image finale avec rectangles
        cv2.imwrite(faces_path, final_image)

# Enregistrer les données YAML
with open(output_yaml, 'w') as f:
    yaml.dump(labeled_faces, f, default_flow_style=False)

cv2.destroyAllWindows()
print(f"Données enregistrées dans {output_yaml}.")
