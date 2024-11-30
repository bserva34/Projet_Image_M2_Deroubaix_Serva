import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
import cv2
import numpy as np
import yaml
from deepface import DeepFace
from scipy.spatial.distance import cosine
import copy
import os
from datetime import datetime

class CameraApp(App):
    def build(self):
        # Layout principal
        main_layout = BoxLayout(orientation="horizontal")

        # Boutons à gauche
        button_layout = BoxLayout(orientation="vertical", size_hint=(0.2, 1))
        self.cnn_btn = Button(text="CNN (Off)")
        self.lbph_btn = Button(text="LBPH (Off)")
        self.import_btn = Button(text="Import IMG")
        self.screenshot_btn = Button(text="Screenshot")
        self.surveillance_btn = Button(text="Activer la surveillance")

        self.screenshot_mode = True

        # Associer les boutons à des fonctions
        self.cnn_active = False
        self.lbph_active = False
        self.import_active = False
        self.surveillance_active = False
        self.screenshot_btn.bind(on_press=self.take_screenshots)
        self.cnn_btn.bind(on_press=self.toggle_cnn)
        self.lbph_btn.bind(on_press=self.toggle_lbph)
        self.import_btn.bind(on_press=self.toggle_import)
        self.surveillance_btn.bind(on_press=self.toggle_surveillance)
        
        self.threshold_CNN = 0.06
        self.threshold_LBPH = 3.0

        self.detected_labels = []

         # Ajout des sliders pour les seuils
        cnn_slider_layout = BoxLayout(orientation="vertical")
        self.cnn_slider_label = Label(text=f"Threshold CNN: {self.threshold_CNN:.3f}", size_hint=(1, 0.2))
        self.cnn_slider = Slider(min=0.01, max=0.10, value=self.threshold_CNN, size_hint=(1, 0.8))
        self.cnn_slider.bind(value=self.update_cnn_threshold)
        cnn_slider_layout.add_widget(self.cnn_slider_label)
        cnn_slider_layout.add_widget(self.cnn_slider)

        lbph_slider_layout = BoxLayout(orientation="vertical")
        self.lbph_slider_label = Label(text=f"Threshold LBPH: {self.threshold_LBPH:.2f}", size_hint=(1, 0.2))
        self.lbph_slider = Slider(min=1.0, max=5.0, value=self.threshold_LBPH, size_hint=(1, 0.8))
        self.lbph_slider.bind(value=self.update_lbph_threshold)
        lbph_slider_layout.add_widget(self.lbph_slider_label)
        lbph_slider_layout.add_widget(self.lbph_slider)

        # Ajout des widgets au layout des boutons
        button_layout.add_widget(self.cnn_btn)
        button_layout.add_widget(cnn_slider_layout)
        button_layout.add_widget(self.lbph_btn)
        button_layout.add_widget(lbph_slider_layout)
        button_layout.add_widget(self.import_btn)
        button_layout.add_widget(self.screenshot_btn)
        button_layout.add_widget(self.surveillance_btn)

        main_layout.add_widget(button_layout)

        # Layout des images à droite
        self.image_layout = BoxLayout(orientation="vertical", size_hint=(0.8, 1))
        #self.reel_time_image = Image()
        self.reco_facial_image = Image()
        #image_layout.add_widget(self.reel_time_image)
        self.image_layout.add_widget(self.reco_facial_image)
        main_layout.add_widget(self.image_layout)

        # Charger la caméra
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Initialisation des visages et des données
        self.faces_data_reel_time = []
        self.faces_data_reco_facial = []
        self.faces_data_lbph = []
        self.last_embedding_time = 0
        self.last_swap_time = 0
        self.last_extract_time = 0
        self.last_lbph_time =0
        self.detection_interval = 0.15 # Délai (en secondes) entre deux générations d'embeddings
        self.embedding_interval = 1.0  # Délai (en secondes) entre deux générations d'embeddings
        self.lbph_interval = 0.15

        self.previous_threshold_CNN = -1.0
        self.previous_threshold_LBPH = -1.0

        self.image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.initial_directory = os.getcwd()
        self.extract_img=False

        Clock.schedule_interval(self.update_reel_time_image, 1.0 / 30)  # 30 FPS
        Clock.schedule_interval(self.update_reco_facial_image, 1.0 / 30)  # 30 FPS pour affichage secondaire

        # Charger les vecteurs caractéristiques
        self.labeled_faces = self.load_yaml("labeled_faces_withCeleb.yml")

        self.name_lbph, self.vector_lbph = self.load__vectors_lbph("lbph_bdd.dat")

        return main_layout

    def update_cnn_threshold(self, instance, value):
        self.threshold_CNN = value
        self.cnn_slider_label.text = f"Threshold CNN: {value:.3f}"

    def update_lbph_threshold(self, instance, value):
        self.threshold_LBPH = value
        self.lbph_slider_label.text = f"Threshold LBPH: {value:.2f}"

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

    def toggle_import(self, instance=None):
        self.import_active = not self.import_active
        if self.import_active:
            self.import_btn.text = "Retour mode caméra"
            self.open_filechooser()
        else:
            self.import_btn.text = "Import IMG"

    def toggle_surveillance(self, instance=None):
        # Activer ou désactiver la surveillance
        self.surveillance_active = not self.surveillance_active
        self.surveillance_btn.text = "Surveillance en cours" if self.surveillance_active else "Activer la surveillance"

        if self.surveillance_active:
            # Activer le mode CNN
            self.cnn_active = True
            self.lbph_active = False
            self.cnn_btn.text = "CNN (On)"
            self.lbph_btn.text = "LBPH (Off)"
            self.screenshot_mode = False
            
            # Réinitialiser le tableau des labels détectés
            self.detected_labels = []
            
            print("Surveillance activée. Réinitialisation des labels détectés.")
        else:
            # Désactiver CNN
            self.cnn_active = False
            self.lbph_active = False
            self.cnn_btn.text = "CNN (Off)"
            self.screenshot_mode = True
            print("Surveillance désactivée.")

    def check_surveillance(self,label):
        if self.detected_labels:
            labels = [face for face in self.detected_labels]  # Récupérer tous les labels
            for i in labels:
                if(i==label):
                    return False
        return True


    def open_filechooser(self):
        # Create a FileChooser widget
        file_chooser = FileChooserListView(path=os.getcwd(), filters=['*.png', '*.jpg', '*.jpeg'])

        # Create buttons for file selection
        btn_select = Button(text="Select", size_hint=(1, 0.2))
        btn_cancel = Button(text="Cancel", size_hint=(1, 0.2))

        # Layout for the popup content
        popup_layout = BoxLayout(orientation="vertical")
        popup_layout.add_widget(file_chooser)
        popup_btns = BoxLayout(size_hint=(1, 0.2))
        popup_btns.add_widget(btn_cancel)
        popup_btns.add_widget(btn_select)
        popup_layout.add_widget(popup_btns)

        # Create a popup
        self.popup = Popup(title="Select a File", content=popup_layout, size_hint=(0.9, 0.9))
        self.popup.open()

        # Bind buttons
        btn_select.bind(on_release=lambda _: self.on_file_selected(file_chooser.selection))
        btn_cancel.bind(on_release=lambda _: self.on_file_not_selected())

    def on_file_not_selected(self):
        self.import_active = False
        self.import_btn.text = "Import IMG"
        self.popup.dismiss()


    def on_file_selected(self, selection):
        if selection:
            self.image_path = selection[0]
            print(f"You have selected: {self.image_path}")  # Debugging statement
            
            # Load and display the image
            self.image = cv2.imread(self.image_path)
            self.image = self.resize_with_padding(self.image, 640, 480)
            frame = cv2.flip(self.image, 0)  # Simulate flipping the image
            self.display_frame(frame, self.reco_facial_image)

            # Reset flags
            self.extract_img = False
            self.previous_threshold_CNN = -1.0
            self.previous_threshold_LBPH = -1.0
            self.lbph_active = False
            self.cnn_active = False

            # Update button states
            self.import_btn.text = "Retour mode camera"
            self.popup.dismiss()
        else:
            print("No file selected!")
            self.import_btn.text = "Import IMG"
            self.popup.dismiss()

    def resize_with_padding(self, image, target_width, target_height):
        original_height, original_width = image.shape[:2]
        aspect_ratio_original = original_width / original_height
        aspect_ratio_target = target_width / target_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = target_width
            new_height = int(target_width / aspect_ratio_original)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio_original)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        delta_w = target_width - new_width
        delta_h = target_height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]  # Remplissage noir
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return padded_image

    def update_reel_time_image(self, dt):
        if self.import_active:
            frame = self.image
            self.faces_data_reel_time = self.detect_faces(frame)
            self.extract_img = True
        else :
            ret, frame = self.capture.read()
            if ret:
                if self.cnn_active or self.lbph_active:
                    current_time = time.time()
                    if current_time - self.last_extract_time > self.detection_interval:
                        self.faces_data_reel_time = self.detect_faces(frame)
                        self.last_extract_time = current_time

    def update_reco_facial_image(self, dt):
        if self.import_active:
            frame = copy.deepcopy(self.image)
            if self.cnn_active :
                if self.threshold_CNN != self.previous_threshold_CNN:
                    self.faces_data_reco_facial = copy.deepcopy(self.faces_data_reel_time)
                    self.update_embeddings(self.faces_data_reco_facial, frame)
                    self.previous_threshold_CNN = self.threshold_CNN
                self.draw_faces(frame, self.faces_data_reco_facial, show_labels=True)
                frame = cv2.flip(frame, 0)
                self.display_frame(frame, self.reco_facial_image)
            elif self.lbph_active :
                if self.threshold_LBPH != self.previous_threshold_LBPH:
                    self.faces_data_lbph = copy.deepcopy(self.faces_data_reel_time)
                    self.update_lbph(self.faces_data_lbph, frame)
                    self.previous_threshold_LBPH = self.threshold_LBPH
                self.draw_faces(frame, self.faces_data_lbph, show_labels=True)
                frame = cv2.flip(frame, 0)
                self.display_frame(frame, self.reco_facial_image)
            else:
                frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                self.display_frame(frame, self.reco_facial_image)
        else :
            ret, frame = self.capture.read()
            if ret:
                if self.cnn_active :
                    current_time = time.time()
                    # Générer les embeddings toutes les secondes
                    if current_time - self.last_embedding_time > self.embedding_interval:
                        self.faces_data_reco_facial = copy.deepcopy(self.faces_data_reel_time)
                        self.update_embeddings(self.faces_data_reco_facial, frame)
                        self.last_embedding_time = current_time
                    elif current_time - self.last_swap_time > self.detection_interval:
                        search_radius = 50 
                        for face_reco in self.faces_data_reco_facial:
                            reco_x, reco_y, reco_w, reco_h = face_reco["rect"]
                            reco_center = (reco_x + reco_w // 2, reco_y + reco_h // 2)
                            closest_face = None
                            min_distance = float('inf')

                            # Limiter la recherche aux visages dans un rayon donné
                            for face_real in self.faces_data_reel_time:
                                real_x, real_y, real_w, real_h = face_real["rect"]
                                real_center = (real_x + real_w // 2, real_y + real_h // 2)

                                # Filtrer par distance approximative avant de calculer
                                if abs(reco_center[0] - real_center[0]) > search_radius or abs(reco_center[1] - real_center[1]) > search_radius:
                                    continue

                                # Calculer la distance exacte
                                distance = ((real_center[0] - reco_center[0]) ** 2 + (real_center[1] - reco_center[1]) ** 2) ** 0.5

                                if distance < min_distance:
                                    min_distance = distance
                                    closest_face = face_real

                            # Mettre à jour la position si un visage proche est trouvé
                            if closest_face:
                                face_reco["rect"] = closest_face["rect"]
                        self.last_swap_time = current_time
                    # Dessiner les rectangles avec les labels et distances
                    self.draw_faces(frame, self.faces_data_reco_facial, show_labels=True)
                    frame = cv2.flip(frame, 0)  # 0 pour retourner verticalement
                    self.display_frame(frame, self.reco_facial_image)
                    
                    if self.surveillance_active:
                        if self.faces_data_reco_facial:
                            labels = [face["label"] for face in self.faces_data_reco_facial]  # Récupérer tous les labels
                            for l in labels:
                                if (self.check_surveillance(l)):
                                    self.detected_labels.append(l)
                                    self.take_screenshots(None)

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
            couleur = (0, 255, 0)
            x, y, w, h = face["rect"]
            
            # Extraire et recadrer le visage
            cropped_face = frame[y:y + h, x:x + w]
            
            # Dimensions de l'image recadrée
            face_h, face_w = cropped_face.shape[:2]
            aspect_ratio = face_w / face_h
            
            # Redimensionner sans déformer
            target_size = 100  # Taille cible
            if aspect_ratio > 1:  # Largeur plus grande
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:  # Hauteur plus grande ou égale
                new_height = target_size
                new_width = int(target_size * aspect_ratio)
            
            resized_face = cv2.resize(cropped_face, (new_width, new_height))

            # Créer une image de 100x100 avec fond blanc
            final_face = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255  # Fond blanc

            # Calculer la position pour centrer le visage redimensionné
            y_offset = (target_size - new_height) // 2
            x_offset = (target_size - new_width) // 2

            # Placer l'image redimensionnée au centre
            final_face[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_face

            # Convertir en niveaux de gris pour l'analyse LBPH
            cropped_face = cv2.cvtColor(final_face, cv2.COLOR_BGR2GRAY)
            
            # Calculer le vecteur LBPH
            test_vector = self.compute_lbph_vector(cropped_face)

            # Trouver la correspondance la plus proche
            min_index, min_distance = self.find_closest_match(test_vector, self.vector_lbph)
            
            if min_distance > self.threshold_LBPH:
                label = "Inconnu"
                couleur = (0, 0, 255)
            else:
                label = self.name_lbph[min_index]
            
            # Mettre à jour les informations du visage
            face.update({
                "embedding": test_vector,        
                "label": label,                  
                "distance": min_distance,        
                "color": couleur,            
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
        mode = self.screenshot_mode
        folder = "screen" if mode else "surveillance"
        if not os.path.exists(folder):
            os.makedirs(folder)

        
        # Obtenir tous les labels des visages détectés
        labels = ["unknown"]
        if self.faces_data_reco_facial:
            labels = [face["label"] for face in self.faces_data_reco_facial]  # Récupérer tous les labels

        # Créer une chaîne avec tous les labels, séparés par des underscores
        label_str = "_".join(labels)

        # Générer un nom de fichier basé sur les labels et l'heure actuelle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/{label_str}_{timestamp}.jpg"

        # Sauvegarder la texture du widget reco_facial_image
        self.save_widget_texture(self.reco_facial_image, filename)


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

    def compare_faces(self, input_embedding, labeled_faces):
        closest_label = "Inconnu"
        min_distance = float("inf")

        for label, entries in labeled_faces.items():
            for entry in entries:
                embedding = np.array(entry["vector"])
                distance = cosine(input_embedding, embedding)

                if distance < min_distance:
                    min_distance = distance
                    closest_label = label
        if min_distance < self.threshold_CNN:
            color = (0, 255, 0) 
        else:
            color = (0, 0, 255)
            closest_label = "Inconnu"
        return closest_label, color, min_distance

    def on_stop(self):
        self.capture.release()


if __name__ == "__main__":
    CameraApp().run()
