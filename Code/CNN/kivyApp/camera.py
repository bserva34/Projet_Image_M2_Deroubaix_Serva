from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
from deepface import DeepFace


class CameraApp(App):
    def build(self):
        self.is_detecting = False
        self.faces_data = []  # Pour stocker les visages détectés
        self.front_camera = True  # Par défaut, utiliser la caméra avant
        layout = BoxLayout(orientation="vertical")

        # Caméra (4/5 de l'écran)
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)
        self.camera.size_hint = (1, 0.8)
        layout.add_widget(self.camera)

        # Boutons (1/5 de l'écran)
        button_layout = BoxLayout(size_hint=(1, 0.2))
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

        # Image pour l'affichage (inversée pour Kivy)
        img_flipped = cv2.flip(img_bgr, 0)

        # Dessiner les rectangles orange sur l'image inversée
        for face in self.faces_data:
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )
            # Adapter les coordonnées pour l'image inversée
            y_flipped = img_flipped.shape[0] - (y + h)
            cv2.rectangle(img_flipped, (x, y_flipped), (x + w, y_flipped + h), (0, 165, 255), 2)

        # Mettre à jour la texture avec les rectangles
        img_rgba = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(img_rgba.shape[1], img_rgba.shape[0]), colorfmt="rgba")
        texture.blit_buffer(img_rgba.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.camera.texture = texture

    def take_screenshot(self, instance):
        if not self.faces_data:
            print("Aucun visage détecté pour la capture d'écran.")
            return

        # Capture d'écran basée sur l'image originale (non inversée)
        texture = self.camera.texture
        if texture:
            buffer = texture.pixels
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(texture.height, texture.width, -1)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img_flipped = cv2.flip(img_bgr, 0)

            # Enregistrer l'image
            cv2.imwrite("screenshot.png", img_flipped)
            print("Capture d'écran enregistrée sous 'screenshot.png'")


if __name__ == "__main__":
    CameraApp().run()