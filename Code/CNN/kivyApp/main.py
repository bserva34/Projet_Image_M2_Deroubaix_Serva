from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import yaml
from kivy.config import Config

# Configurer la taille de la fenêtre
Config.set('graphics', 'width', '450')
Config.set('graphics', 'height', '800')

# Charger les visages étiquetés
with open("../DeepFace/labeled_faces.yml", "r") as file:
    labeled_faces = yaml.safe_load(file)

class CameraApp(App):
    def build(self):
        self.is_detecting = False
        self.temp_image_path = "temp_frame.jpg"

        # Layout principal
        main_layout = BoxLayout(orientation="vertical")

        # Caméra en haut
        self.camera = Camera(play=True, resolution=(640, 480))
        self.camera.size_hint = (1, 0.4)
        main_layout.add_widget(self.camera)

        # Boutons au milieu
        button_layout = BoxLayout(size_hint=(1, 0.1))
        cnn_btn = Button(text="RecoCNN")
        lbph_btn = Button(text="RecoLBPH")
        screenshot_btn = Button(text="Screenshot")

        cnn_btn.bind(on_press=self.toggle_detection)
        lbph_btn.bind(on_press=self.reco_lbph)  # Ne fait rien pour l'instant
        screenshot_btn.bind(on_press=self.save_screenshot)

        button_layout.add_widget(cnn_btn)
        button_layout.add_widget(lbph_btn)
        button_layout.add_widget(screenshot_btn)
        main_layout.add_widget(button_layout)

        # Image OpenCV en bas
        self.annotated_image = Image(size_hint=(1, 0.5))
        main_layout.add_widget(self.annotated_image)

        return main_layout

    def toggle_detection(self, instance):
        if not self.is_detecting:
            self.is_detecting = True
            instance.text = "Arrêter RecoCNN"
            Clock.schedule_interval(self.update_temp_frame, 1.0 / 30)  # 30 FPS
        else:
            self.is_detecting = False
            instance.text = "RecoCNN"
            Clock.unschedule(self.update_temp_frame)

    def reco_lbph(self, instance):
        # Placeholder pour le bouton RecoLBPH
        print("RecoLBPH non implémenté pour l'instant.")

    def save_screenshot(self, instance):
        # Enregistrer l'image temporaire sous screen.jpg
        cv2.imwrite("screen.jpg", cv2.imread(self.temp_image_path))
        print("Image sauvegardée sous screen.jpg")

    def update_temp_frame(self, dt):
        # Capturer une frame de la caméra
        texture = self.camera.texture
        if not texture:
            return

        buffer = texture.pixels
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(self.temp_image_path, img_bgr)  # Sauvegarder comme image temporaire

        # Appliquer la détection si RecoCNN est actif
        if self.is_detecting:
            self.detect_faces()

        # Mettre à jour l'image OpenCV
        self.update_annotated_image(img_bgr)

    def detect_faces(self):
        try:
            img_bgr = cv2.imread(self.temp_image_path)
            faces_data = DeepFace.extract_faces(
                img_path=img_bgr, detector_backend="dlib", enforce_detection=True, align=False
            )

            for face in faces_data:
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
                        align=False,
                    )
                    input_embedding = embeddings[0]["embedding"]
                    label_text, color = self.compare_faces(input_embedding, labeled_faces)
                except Exception as e:
                    print(f"Erreur lors de la comparaison : {e}")
                    label_text, color = "Erreur", (0, 0, 255)

                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imwrite(self.temp_image_path, img_bgr)  # Mettre à jour le fichier temporaire
        except Exception as e:
            print(f"Erreur lors de la détection des visages : {e}")

    def compare_faces(self, input_embedding, labeled_faces, threshold=0.4):
        closest_label = "Inconnu"
        min_distance = float("inf")
        for label, entries in labeled_faces.items():
            for entry in entries:
                embedding = np.array(entry["vector"])
                distance = cosine(input_embedding, embedding)
                if distance < min_distance:
                    min_distance = distance
                    closest_label = label
        if min_distance < threshold:
            return closest_label, (0, 255, 0)  # Vert
        return "Inconnu", (0, 0, 255)  # Rouge

    def update_annotated_image(self, img_bgr):
        # Mettre à jour l'image affichée en bas
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(img_rgba.shape[1], img_rgba.shape[0]), colorfmt="rgba")
        texture.blit_buffer(img_rgba.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.annotated_image.texture = texture

if __name__ == "__main__":
    CameraApp().run()
