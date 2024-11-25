Les images utilisés pour le YML ne sont pas disponible en public.

labeled_faces.yml
embedding = DeepFace.represent(cropped_face,detector_backend='skip', model_name="Dlib", enforce_detection=True, align=True)[0]["embedding"]

labeled_faces_FaceNet.yml
embedding = DeepFace.represent(cropped_face, model_name="Dlib", enforce_detection=False)[0]["embedding"]
