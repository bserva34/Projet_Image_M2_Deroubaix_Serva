import threading
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
import cv2
import numpy as np
import yaml
from deepface import DeepFace
from scipy.spatial.distance import cosine  # Import pour calculer la distance cosinus


class CameraApp(App):
    def build(self):
        # Layout principal
        main_layout = BoxLayout(orientation="horizontal")

        # Boutons à gauche
        button_layout = BoxLayout(orientation="vertical", size_hint=(0.2, 1))
        self.cnn_btn = Button(text="CNN (Off)")
        self.lbph_btn = Button(text="LBPH")
        self.screenshot_btn = Button(text="Screenshot")

        # Associer les boutons à des fonctions
        self.cnn_active = False  # État du bouton CNN
        self.cnn_btn.bind(on_press=self.toggle_cnn)
        self.lbph_btn.bind(on_press=self.toggle_lbph)
        self.screenshot_btn.bind(on_press=self.take_screenshots)

        button_layout.add_widget(self.cnn_btn)
        button_layout.add_widget(self.lbph_btn)
        button_layout.add_widget(self.screenshot_btn)
        main_layout.add_widget(button_layout)

        # Layout des images à droite
        image_layout = BoxLayout(orientation="vertical", size_hint=(0.8, 1))
        self.reel_time_image = Image()
        self.reco_facial_image = Image()
        image_layout.add_widget(self.reel_time_image)
        image_layout.add_widget(self.reco_facial_image)
        main_layout.add_widget(image_layout)

        # Charger la caméra
        self.capture = cv2.VideoCapture(0)  # Index 0 pour la webcam
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        Clock.schedule_interval(self.update_reel_time_image, 1.0 / 24)  # 24 FPS
        Clock.schedule_interval(self.update_reco_facial_image, 1.0 / 12)  # 12 FPS mais tentative de reconnaissance

        # Charger les vecteurs caractéristiques
        self.labeled_faces = self.load_yaml("../DeepFace/labeled_faces.yml")

        # Thread pour éviter le blocage de l'interface
        self.processing_thread = None
        self.processing_lock = threading.Lock()

        return main_layout

    def load_yaml(self, file_path):
        # Charger les données depuis un fichier YAML
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
            return data
        except Exception as e:
            print(f"Erreur lors du chargement du fichier YAML : {e}")
            return {}

    def toggle_lbph(self, instance):
        # Fonction vide appelée par les boutons
        print(f"{instance.text} button pressed (function not implemented).")

    def toggle_cnn(self, instance):
        # Activer ou désactiver le mode CNN
        self.cnn_active = not self.cnn_active
        self.cnn_btn.text = "CNN (On)" if self.cnn_active else "CNN (Off)"
        print(f"CNN mode {'enabled' if self.cnn_active else 'disabled'}.")

    def update_reel_time_image(self, dt):
        # Capture et affichage pour l'image à FPS élevé
        ret, frame = self.capture.read()
        if ret:
            if self.cnn_active:
                self.process_and_display(frame, self.reel_time_image, False)
            else:
                self.display_frame(frame, self.reel_time_image)

    def update_reco_facial_image(self, dt):
        # Démarrer un thread pour traiter la reconnaissance faciale si CNN actif
        if self.cnn_active and (self.processing_thread is None or not self.processing_thread.is_alive()):
            self.processing_thread = threading.Thread(target=self.process_reco_facial_frame)
            self.processing_thread.start()
        elif not self.cnn_active:
            ret, frame = self.capture.read()
            if ret:
                self.display_frame(frame, self.reco_facial_image)

    def process_reco_facial_frame(self):
        # Traitement de la reconnaissance faciale pour l'image du bas
        with self.processing_lock:
            ret, frame = self.capture.read()
            if not ret:
                return
            self.process_and_display(frame, self.reco_facial_image, True)

    def process_and_display(self, frame, image_widget, reco_facial):
        # Extraction des visages et affichage
        try:
            # Extraction des visages
            faces_data = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="dlib",
                enforce_detection=False,
                align=False,
            )
            color=(255, 0, 0) 
            # Dessiner les rectangles autour des visages et ajouter le label
            for face in faces_data:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]

                # Couper le visage
                cropped_face = frame[y:y + h, x:x + w]

                # Générer l'embedding et comparer avec le YAML
                if reco_facial:
                    try:
                        embedding = DeepFace.represent(
                            cropped_face, detector_backend="skip",
                            model_name="Dlib", enforce_detection=True, align=True
                        )[0]["embedding"]

                        label, color, min_distance = self.compare_faces(embedding, self.labeled_faces)

                        # Dessiner le rectangle et ajouter le label
                        label_text = f"{label} ({min_distance:.2f})"
                        cv2.putText(frame, label_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


                    except Exception as e:
                        print(f"Erreur lors de la génération de l'embedding : {e}")
                        continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        except Exception as e:
            print(f"Erreur lors de la reconnaissance faciale : {e}")

        # Mettre à jour l'affichage
        self.display_frame(frame, image_widget)

    def compare_faces(self, input_embedding, labeled_faces, threshold=0.04):
        # Comparer l'embedding avec les visages étiquetés
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

        return label_text, color, min_distance

    @mainthread
    def display_frame(self, frame, image_widget):
        # Convertir l'image OpenCV en texture Kivy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        buffer = frame_rgb.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="rgba")
        texture.blit_buffer(buffer, colorfmt="rgba", bufferfmt="ubyte")
        image_widget.texture = texture

    def take_screenshots(self, instance):
        # Sauvegarder les deux flux d'image visibles avec les rectangles
        self.save_widget_texture(self.reel_time_image, "screen_extract_face.jpg")
        self.save_widget_texture(self.reco_facial_image, "screen_reco_facial.jpg")

    def save_widget_texture(self, widget, filename):
        # Convertir la texture du widget en image et sauvegarder
        texture = widget.texture
        if texture is not None:
            width, height = texture.size
            buffer = texture.pixels  # Obtenir les pixels bruts
            frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 4)  # Convertir en image
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convertir en BGR pour OpenCV
            cv2.imwrite(filename, frame_bgr)
            print(f"Screenshot saved: {filename}")

    def on_stop(self):
        # Libérer la caméra à la fermeture de l'application
        self.capture.release()


if __name__ == "__main__":
    CameraApp().run()
