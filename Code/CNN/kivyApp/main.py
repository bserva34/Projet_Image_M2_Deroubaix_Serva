from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
from deepface import DeepFace
from kivy.config import Config
from scipy.spatial.distance import cosine
import yaml 
Config.set('graphics', 'width', '450')
Config.set('graphics', 'height', '800')

with open("../DeepFace/labeled_faces.yml", "r") as file:
    labeled_faces = yaml.safe_load(file)

class CameraApp(App):
    def build(self):
        self.is_detecting = False
        self.faces_data = []  # Pour stocker les visages détectés
        self.front_camera = True  # Par défaut, utiliser la caméra avant
        layout = BoxLayout(orientation="vertical")

        # Caméra (4/5 de l'écran)
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)

        self.camera.size_hint = (1, 0.88)
        layout.add_widget(self.camera)

        # Boutons (1/5 de l'écran)
        button_layout = BoxLayout(size_hint=(1, 0.12))
        change_camera_btn = Button(text="Changer de caméra")
        detect_btn = Button(text="Commencer la détection")
        screenshot_btn = Button(text="Screenshot")

        # Lier les boutons à des fonctions
        change_camera_btn.bind(on_press=self.change_camera)
        detect_btn.bind(on_press=self.toggle_detection)
        screenshot_btn.bind(on_press=self.take_screenshot)

        button_layout.add_widget(change_camera_btn)
        button_layout.add_widget(detect_btn)
        button_layout.add_widget(screenshot_btn)

        layout.add_widget(button_layout)
        return layout

    def change_camera(self, instance):
        self.front_camera = not self.front_camera
        self.camera.index = 1 if self.front_camera else 0

    def toggle_detection(self, instance):
        print("")
        if not self.is_detecting:
            self.is_detecting = True
            instance.text = "Arrêter la détection"
            Clock.schedule_interval(self.detect_faces, 1.0 / 30)  # Détection à 30 FPS
        else:
            self.is_detecting = False
            instance.text = "Commencer la détection"
            Clock.unschedule(self.detect_faces)

    def detect_faces(self, dt):
        texture = self.camera.texture
        if not texture:
            return
        buffer = texture.pixels
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)

        # Préparation de l'image pour la détection (sans inversion)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        try:
            # Détection des visages avec DeepFace
            self.faces_data = DeepFace.extract_faces(
                img_path=img_bgr, detector_backend="dlib", enforce_detection=True, align=False
            )
        except Exception as e:
            print(f"Erreur lors de la détection des visages : {e}")
            self.faces_data = []

        # Dessiner les rectangles et le texte sur l'image originale (img_bgr)
        for face in self.faces_data:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )

            # Dessiner un rectangle orange autour du visage
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 165, 255), 2)

            # Ajouter un label "Reconnaissance en cours" en dessous du rectangle
            label_text = "Reconnaissance en cours"
            font_scale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = x
            text_y = y + h + text_size[1] + 5  # Positionner sous le rectangle

            # Ajouter une bordure noire autour du texte
            # Texte noir (bordure)
            cv2.putText(img_bgr, label_text, (text_x-1, text_y-1), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_bgr, label_text, (text_x+1, text_y-1), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_bgr, label_text, (text_x-1, text_y+1), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_bgr, label_text, (text_x+1, text_y+1), font, font_scale, (0, 0, 0), thickness+2)

            # Texte principal (couleur de l'écriture)
            cv2.putText(img_bgr, label_text, (text_x, text_y), font, font_scale, (0, 165, 255), thickness)

        # Image pour l'affichage (inversée pour Kivy)
        img_flipped = cv2.flip(img_bgr, 0)

        # Mettre à jour la texture avec les rectangles et labels sur l'image inversée
        img_rgba = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(img_rgba.shape[1], img_rgba.shape[0]), colorfmt="rgba")
        texture.blit_buffer(img_rgba.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.camera.texture = texture


    def take_screenshot(self, instance):
        # Récupérer une seule frame de la caméra
        texture = self.camera.texture
        if not texture:
            print("Aucune texture disponible.")
            return

        buffer = texture.pixels
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)

        # Convertir l'image en BGR pour OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        try:
            # Détection des visages avec DeepFace
            self.faces_data = DeepFace.extract_faces(
                img_path=img_bgr, detector_backend="dlib", enforce_detection=True, align=False
            )
        except Exception as e:
            print(f"Erreur lors de la détection des visages : {e}")
            self.faces_data = []

        # Seuil pour considérer une correspondance valide
        threshold = 0.4  # Distance cosinus maximum pour une correspondance acceptable

        for face in self.faces_data:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )

            # Extraire l'encodage du visage avec DeepFace
            try:
                embeddings = DeepFace.represent(
                    img_path=img_bgr,
                    model_name="Facenet",
                    detector_backend="dlib",
                    enforce_detection=False,
                    align=False
                )
                input_embedding = embeddings[0]["embedding"]
            except Exception as e:
                print(f"Erreur lors de l'extraction des caractéristiques : {e}")
                continue

            # Trouver le label le plus proche
            closest_label = None
            min_distance = float("inf")

            for label, entries in labeled_faces.items():
                for entry in entries:
                    embedding = np.array(entry["vector"])
                    distance = cosine(input_embedding, embedding)  # Calcul de la distance cosinus

                    if distance < min_distance:
                        min_distance = distance
                        closest_label = label

            # Définir la couleur et le texte du cadre en fonction du seuil
            if min_distance < threshold:
                label_text = closest_label
                color = (0, 255, 0)  # Orange pour les correspondances valides
            else:
                label_text = "Inconnu"
                color = (0, 0, 255)  # Rouge pour les inconnus

            # Dessiner un rectangle autour du visage
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)

            # Ajouter un label sous le rectangle
            font_scale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = x
            text_y = y + h + text_size[1] + 5

            # Ajouter une bordure noire autour du texte
            cv2.putText(img_bgr, label_text, (text_x - 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(img_bgr, label_text, (text_x + 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(img_bgr, label_text, (text_x - 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(img_bgr, label_text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 2)

            # Texte principal (couleur)
            cv2.putText(img_bgr, label_text, (text_x, text_y), font, font_scale, color, thickness)

        # Sauvegarder l'image annotée
        screenshot_path = "labeled_screenshot.png"
        cv2.imwrite(screenshot_path, img_bgr)
        print(f"Capture d'écran enregistrée sous '{screenshot_path}'")

        # Mettre à jour l'affichage de la caméra avec l'image annotée
        img_flipped = cv2.flip(img_bgr, 0)  # Inverser verticalement pour l'affichage Kivy
        img_rgba = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(img_rgba.shape[1], img_rgba.shape[0]), colorfmt="rgba")
        texture.blit_buffer(img_rgba.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.camera.texture = texture


if __name__ == "__main__":
    CameraApp().run()



