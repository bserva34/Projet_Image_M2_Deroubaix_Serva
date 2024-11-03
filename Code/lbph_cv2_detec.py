import cv2
import numpy as np

# Charger le modèle pré-entraîné
model_path = 'model.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Charger l'image à détecter
#input_image_path = 'BDD/40_personnes_ensembles/s1_2.pgm'  #images présente dans le modèle 
input_image_path = 'BDD/40_personnes_test/s1_last_9.pgm'  #images test
#input_image_path = 'BDD/archive2_pgm/subject01.centerlight.pgm' #images qui n'a rien à voir
image = cv2.imread(input_image_path)
image = cv2.resize(image, (92, 112))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Si l'image est déjà un visage, on peut traiter directement l'image
face_roi = cv2.resize(gray_image, (92, 112))  # Redimensionner le visage à 100x100 pixels
label, confidence = recognizer.predict(face_roi)

# Calculer le pourcentage de confiance
percentage = 100 - confidence  # Assurez-vous que le pourcentage est positif
print(f"Label: {label}, Confiance: {percentage:.2f}%")

# Dessiner un rectangle (ou toute autre annotation si nécessaire)
cv2.rectangle(image, (0, 0), (100, 100), (255, 0, 0), 2)  # Vous pouvez ajuster les coordonnées selon vos besoins
cv2.putText(image, f'Person {label}: {percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Afficher l'image avec les visages détectés
cv2.imshow('Image', image)

# Attendre la fermeture de la fenêtre
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour fermer
        break

cv2.destroyAllWindows()
