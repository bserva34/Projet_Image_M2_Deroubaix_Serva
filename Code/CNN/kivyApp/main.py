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
        if not self.is_detecting:
            self.is_detecting = True
            instance.text = "Arrêter la détection"
            Clock.schedule_interval(self.detect_faces, 1.0 / 30)  # Détection à 30 FPS
        else:
            self.is_detecting = False
            instance.text = "Commencer la détection"
            Clock.unschedule(self.detect_faces)

    def compare_faces(self, input_embedding, labeled_faces, threshold=0.4):
        closest_label = "Inconnu"
        min_distance = float("inf")

        for label, entries in labeled_faces.items():
            for entry in entries:
                embedding = np.array(entry["vector"])
                distance = cosine(input_embedding, embedding)  # Calcul de la distance cosinus

                if distance < min_distance:
                    min_distance = distance
                    closest_label = label

        # Définir le texte et la couleur selon le seuil
        if min_distance < threshold:
            label_text = closest_label
            color = (0, 255, 0)  # Vert pour une correspondance valide
        else:
            label_text = "Inconnu"
            color = (0, 0, 255)  # Rouge pour un visage non reconnu

        return label_text, color
        
    #compara la moyenne des distances à label en sachant qu'il faut changer le seuil
    # def compare_faces(self, input_embedding, labeled_faces, threshold=0.4):
    #     closest_label = "Inconnu"
    #     min_avg_distance = float("inf")  # Distance moyenne minimale initiale

    #     for label, entries in labeled_faces.items():
    #         distances = []  # Liste pour stocker les distances pour chaque label

    #         for entry in entries:
    #             embedding = np.array(entry["vector"])
    #             distance = cosine(input_embedding, embedding)  # Calcul de la distance cosinus
    #             distances.append(distance)

    #         # Calculer la distance moyenne pour ce label
    #         avg_distance = np.mean(distances)

    #         # Si la distance moyenne est inférieure à la distance minimale précédente, mettre à jour
    #         if avg_distance < min_avg_distance:
    #             min_avg_distance = avg_distance
    #             closest_label = label

    #     # Définir le texte et la couleur selon la distance moyenne et le seuil
    #     if min_avg_distance < threshold:
    #         label_text = closest_label
    #         color = (0, 255, 0)  # Vert pour une correspondance valide
    #     else:
    #         label_text = "Inconnu"
    #         color = (0, 0, 255)  # Rouge pour un visage non reconnu

    #     return label_text, color


    def detect_faces(self, dt):
        texture = self.camera.texture
        if not texture:
            return

        buffer = texture.pixels
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        try:
            self.faces_data = DeepFace.extract_faces(
                img_path=img_bgr, detector_backend="dlib", enforce_detection=True, align=False
            )
        except Exception as e:
            print(f"Erreur lors de la détection des visages : {e}")
            self.faces_data = []

        for face in self.faces_data:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )

            try:
                embeddings = DeepFace.represent(
                    img_path=img_bgr,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=False
                )
                input_embedding = embeddings[0]["embedding"]

                # Utiliser la méthode compare_faces
                label_text, color = self.compare_faces(input_embedding, labeled_faces)
            except Exception as e:
                print(f"Erreur lors de la comparaison des visages : {e}")
                label_text, color = "Erreur", (0, 0, 255)

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        img_flipped = cv2.flip(img_bgr, 0)
        img_rgba = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(img_rgba.shape[1], img_rgba.shape[0]), colorfmt="rgba")
        texture.blit_buffer(img_rgba.tobytes(), colorfmt="rgba", bufferfmt="ubyte")

        if hasattr(self, "annotated_image"):
            self.annotated_image.texture = texture
        else:
            from kivy.uix.image import Image
            self.annotated_image = Image(texture=texture, size_hint=(1, 0.88))
            self.root.add_widget(self.annotated_image, index=0)

    def take_screenshot(self, instance):
        texture = self.camera.texture
        if not texture:
            print("Aucune texture disponible.")
            return

        buffer = texture.pixels
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        try:
            self.faces_data = DeepFace.extract_faces(
                img_path=img_bgr, detector_backend="dlib", enforce_detection=True, align=False
            )
        except Exception as e:
            print(f"Erreur lors de la détection des visages : {e}")
            self.faces_data = []

        for face in self.faces_data:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )

            try:
                embeddings = DeepFace.represent(
                    img_path=img_bgr,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=False
                )
                input_embedding = embeddings[0]["embedding"]

                label_text, color = self.compare_faces(input_embedding, labeled_faces)
            except Exception as e:
                print(f"Erreur lors de la comparaison des visages : {e}")
                label_text, color = "Erreur", (0, 0, 255)

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        screenshot_path = "labeled_screenshot.png"
        cv2.imwrite(screenshot_path, img_bgr)
        print(f"Capture d'écran enregistrée sous '{screenshot_path}'")

if __name__ == "__main__":
    CameraApp().run()




