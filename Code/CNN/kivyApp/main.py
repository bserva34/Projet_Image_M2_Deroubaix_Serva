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
from scipy.spatial.distance import cosine
import copy

class CameraApp(App):
    def build(self):
        # Layout principal
        main_layout = BoxLayout(orientation="horizontal")

        # Boutons à gauche
        button_layout = BoxLayout(orientation="vertical", size_hint=(0.2, 1))
        self.cnn_btn = Button(text="CNN (Off)")
        self.lbph_btn = Button(text="LBPH (Off)")
        self.screenshot_btn = Button(text="Screenshot")

        # Associer les boutons à des fonctions
        self.cnn_active = False
        self.lbph_active = False
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
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Initialisation des visages et des données
        self.faces_data_reel_time = []
        self.faces_data_reco_facial = []
        self.faces_data_lbph = []
        self.last_embedding_time = 0
        self.last_extract_time = 0
        self.last_lbph_time =0
        self.detection_interval = 0.15 # Délai (en secondes) entre deux générations d'embeddings
        self.embedding_interval = 1.0  # Délai (en secondes) entre deux générations d'embeddings
        self.lbph_interval = 0.15

        Clock.schedule_interval(self.update_reel_time_image, 1.0 / 30)  # 30 FPS
        Clock.schedule_interval(self.update_reco_facial_image, 1.0 / 30)  # 30 FPS pour affichage secondaire

        # Charger les vecteurs caractéristiques
        self.labeled_faces = self.load_yaml("../DeepFace/labeled_faces.yml")

        self.name_lbph, self.vector_lbph = self.load__vectors_lbph("lbph_bdd.dat")

        return main_layout

    def load_yaml(self, file_path):
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
            return data
        except Exception as e:
            print(f"Erreur lors du chargement du fichier YAML : {e}")
            return {}

    def load__vectors_lbph(self, file_path):
        vectors = []
        names = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                name = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                names.append(name)
                vectors.append(vector)

        return names, np.array(vectors)

    def compute_lbph_vector(self, gray_image):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([gray_image], np.array([0]))
        hist = recognizer.getHistograms()[0]
        return np.array(hist).flatten()


    def find_closest_match(self, test_vector, vectors):
        distances = np.linalg.norm(vectors - test_vector, axis=1)  # Distances euclidiennes
        min_index = np.argmin(distances)
        return min_index, distances[min_index]

    def toggle_lbph(self, instance):
        self.lbph_active = not self.lbph_active
        if self.lbph_active:
            self.cnn_active = False
            self.cnn_btn.text = "CNN (Off)"
        self.lbph_btn.text = "LBPH (On)" if self.lbph_active else "LBPH (Off)"
        print(f"LBPH mode {'enabled' if self.lbph_active else 'disabled'}.")

    def toggle_cnn(self, instance):
        self.cnn_active = not self.cnn_active
        if self.cnn_active:
            self.lbph_active = False
            self.lbph_btn.text = "LBPH (Off)"
        self.cnn_btn.text = "CNN (On)" if self.cnn_active else "CNN (Off)"
        print(f"CNN mode {'enabled' if self.cnn_active else 'disabled'}.")

    def update_reel_time_image(self, dt):
        ret, frame = self.capture.read()
        if ret:
            if self.cnn_active or self.lbph_active:
                #current_frame_count = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = time.time()
                # Appeler extract_faces toutes les 10 frames
                #if current_frame_count % self.detection_interval == 0:
                if current_time - self.last_extract_time > self.detection_interval:
                    self.faces_data_reel_time = self.detect_faces(frame)
                    self.last_extract_time = current_time

                # Dessiner les rectangles avec les données existantes
                self.draw_faces(frame, self.faces_data_reel_time)

                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reel_time_image)

            else:
                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reel_time_image)



    def update_reco_facial_image(self, dt):
        ret, frame = self.capture.read()
        if ret:
            if self.cnn_active :
                #current_frame_count = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = time.time()

                # Générer les embeddings toutes les secondes
                #if current_frame_count % self.embedding_interval == 0:
                if current_time - self.last_embedding_time > self.embedding_interval:
                    self.faces_data_reco_facial = copy.deepcopy(self.faces_data_reel_time)
                    #self.faces_data_reco_facial=self.faces_data_reel_time
                    self.update_embeddings(self.faces_data_reco_facial, frame)
                    self.last_embedding_time = current_time

                # Dessiner les rectangles avec les labels et distances
                self.draw_faces(frame, self.faces_data_reco_facial, show_labels=True)
                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reco_facial_image)
            elif self.lbph_active:
                current_time = time.time()

                if current_time - self.last_lbph_time > self.lbph_interval:
                    self.faces_data_lbph = copy.deepcopy(self.faces_data_reel_time)
                    self.update_lbph(self.faces_data_lbph, frame)
                    self.last_lbph_time = current_time

                # Dessiner les rectangles avec les labels et distances
                self.draw_faces(frame, self.faces_data_lbph, show_labels=True)
                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reco_facial_image)

            else:
                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reco_facial_image)

    def detect_faces(self, frame):
        try:
            faces_data = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="dlib",
                enforce_detection=False,
                align=False,
            )
            processed_faces = []
            for face in faces_data:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]
                processed_faces.append({
                    "rect": (x, y, w, h),
                    "embedding": None,
                    "label": "Inconnu",
                    "distance": None,
                    "color": (255, 0, 0),
                })
            return processed_faces
        except Exception as e:
            print(f"Erreur lors de l'extraction des visages : {e}")
            return []

    def update_lbph(self, faces_data, frame):
        for face in faces_data:
            x, y, w, h = face["rect"]
            cropped_face = frame[y:y + h, x:x + w]
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

            # Calculer l'histogramme du visage crop
            test_vector = self.compute_lbph_vector(cropped_face)

            # Trouver la correspondance la plus proche
            min_index, min_distance = self.find_closest_match(test_vector, self.vector_lbph)

            # Récupérer le label correspondant à l'index (nom)
            label = self.name_lbph[min_index]
            
            # Mettre à jour les informations du dictionnaire face
            face.update({
                "embedding": test_vector,        # L'histogramme aplati
                "label": label,                  # Nom associé à l'index
                "distance": min_distance,        # Distance minimale (similarité)
                "color": (0, 255, 0),            # Couleur verte
            })

    def update_embeddings(self, faces_data, frame):
        for face in faces_data:
            x, y, w, h = face["rect"]
            cropped_face = frame[y:y + h, x:x + w]
            try:
                embedding = DeepFace.represent(
                    cropped_face,
                    detector_backend="skip",
                    model_name="Dlib",
                    enforce_detection=True,
                    align=True,
                )[0]["embedding"]
                label, color, min_distance = self.compare_faces(embedding, self.labeled_faces)
                face.update({
                    "embedding": embedding,
                    "label": label,
                    "distance": min_distance,
                    "color": color,
                })
            except Exception as e:
                print(f"Erreur lors de la génération de l'embedding : {e}")

    def draw_faces(self, frame, faces_data, show_labels=False):
        for face in faces_data:
            x, y, w, h = face["rect"]
            color = face["color"]
            label = face["label"]
            distance = face["distance"]

            # Dessiner le rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Ajouter le label si demandé
            if show_labels and distance is not None:
                label_text = f"{label} ({distance:.2f})"
                cv2.putText(frame, label_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def display_frame(self, frame, image_widget):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        buffer = frame_rgb.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="rgba")
        texture.blit_buffer(buffer, colorfmt="rgba", bufferfmt="ubyte")
        image_widget.texture = texture

    def take_screenshots(self, instance):
        self.save_widget_texture(self.reel_time_image, "screen_extract_face.jpg")
        self.save_widget_texture(self.reco_facial_image, "screen_reco_facial.jpg")

    def save_widget_texture(self, widget, filename):
        texture = widget.texture
        if texture is not None:
            width, height = texture.size
            buffer = texture.pixels
            frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 4)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame_bgr = cv2.flip(frame_bgr, 0)  # 0 pour retourner verticalement
            cv2.imwrite(filename, frame_bgr)
            print(f"Screenshot saved: {filename}")

    def compare_faces(self, input_embedding, labeled_faces, threshold=0.075):
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
            color = (0, 255, 0) 
        else:
            color = (0, 0, 255)
            closest_label = "Inconnu"
        return closest_label, color, min_distance

    def on_stop(self):
        self.capture.release()


if __name__ == "__main__":
    CameraApp().run()
