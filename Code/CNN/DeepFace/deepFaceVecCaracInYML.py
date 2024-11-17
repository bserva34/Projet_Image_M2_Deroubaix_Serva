import cv2
import os
import yaml
from deepface import DeepFace
from tkinter import Tk, simpledialog
import shutil
import numpy as np

# Chemins des dossiers d'entrée et du fichier YAML
input_folder = "to_crop"  # Dossier contenant les images
output_yaml = "labeled_faces.yml"  # Fichier YAML pour stocker les labels et vecteurs

faces_img_traitee_path = "image_traitée"
faces_img_path = "image_visage"
to_review_img_path = "image_a_revoir"  # Dossier pour les images à revoir
os.makedirs(faces_img_traitee_path, exist_ok=True)
os.makedirs(faces_img_path, exist_ok=True)
os.makedirs(to_review_img_path, exist_ok=True)  # Créer le dossier image_a_revoir

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

# Fonction pour obtenir un label via Tkinter
def get_label():
    root = Tk()
    root.withdraw()  # Hide the root window
    label = simpledialog.askstring("Input", "Entrez le label pour ce visage:")
    root.destroy()
    return label

# Fonction pour redimensionner l'image tout en maintenant son rapport d'aspect
def resize_image_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculer les rapports de taille pour chaque dimension
    aspect_ratio = w / h
    if w > h:
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * aspect_ratio)
    
    # Redimensionner l'image en préservant l'aspect
    return cv2.resize(image, (new_w, new_h))

# Fonction pour comparer les vecteurs d'embedding (en utilisant la distance cosinus)
def is_duplicate_embedding(embedding, existing_embeddings):
    for existing_embedding in existing_embeddings:
        # Calculer la distance cosinus entre l'embedding actuel et les embeddings existants
        dist = np.linalg.norm(np.array(embedding) - np.array(existing_embedding))
        if dist < 1e-6:  # Si la distance est très faible, considérer comme doublon
            return True
    return False

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

    # Créer une copie de l'image pour dessiner les rectangles
    faces_img = image.copy()

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

        # Redimensionner l'image pour s'adapter à la fenêtre d'affichage tout en conservant le rapport d'aspect
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

        # Dessiner le rectangle vert autour du visage
        cv2.rectangle(canvas, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 2)

        # Afficher l'image redimensionnée
        cv2.imshow("Image", canvas)

        # Demander un label pour le visage via Tkinter
        label = get_label()

        # Si le label est 'quit', sauvegarder le YAML et quitter le programme
        if label is None or label.lower() == 'quit':
            with open(output_yaml, 'w') as f:
                yaml.dump(labeled_faces, f, default_flow_style=False)
            print("Données labellisées enregistrées et programme terminé.")
            cv2.destroyAllWindows()
            exit()

        # Si le label est vide, déplacer l'image dans le dossier "image_a_revoir"
        if not label:
            print(f"Image {image_name} déplacée vers {to_review_img_path}.")
            shutil.copy(img_path, os.path.join(to_review_img_path, image_name))  # Copier l'image au lieu de déplacer
            continue

        # Obtenir le vecteur caractéristique pour le visage détecté
        try:
            cropped_face = image[y1:y2, x1:x2]
            embedding = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding pour {image_name}: {e}")
            continue

        # Vérifier si le vecteur d'embedding est déjà dans le fichier YAML pour ce label
        if label not in labeled_faces:
            labeled_faces[label] = []

        # Vérifier si le vecteur existe déjà
        vector_exists = False
        for entry in labeled_faces[label]:
            if np.array_equal(np.array(entry['vector']), np.array(embedding)):
                vector_exists = True
                break

        if not vector_exists:
            # Incrémenter le nom de fichier pour éviter les doublons
            label_count = sum(1 for f in os.listdir("image_traitée") if f.startswith(label))
            new_name = f"{label}_{label_count + 1}.jpg"  # Incrémentation du nom de fichier
            new_path = os.path.join("image_traitée", new_name)

            # Enregistrer l'image modifiée avec les visages dans "image_visage"
            cv2.imwrite(os.path.join(faces_img_path, new_name), canvas)

            # Copier l'image dans "image_traitée" au lieu de la déplacer
            shutil.copy(img_path, new_path)

            # Ajouter l'image et l'embedding dans le YAML
            new_entry = {
                "image": new_name,  # Le nouveau nom du fichier dans "image_traitée"
                "vector": embedding
            }
            labeled_faces[label].append(new_entry)

    # Supprimer les images traitées après la boucle
    if os.path.exists(img_path):
        os.remove(img_path)

# Sauvegarder les données labellisées dans le fichier YAML
with open(output_yaml, 'w') as f:
    yaml.dump(labeled_faces, f, default_flow_style=False)

# Fermer toutes les fenêtres d'affichage
cv2.destroyAllWindows()
