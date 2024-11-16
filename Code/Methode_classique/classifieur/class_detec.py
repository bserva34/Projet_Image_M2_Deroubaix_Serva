# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # Charger le modèle
# model_path = "classifier_model.pkl"
# classifier, scaler, pca = joblib.load(model_path)

# # Fonction pour normaliser un vecteur
# def normalize_vector(vector, scaler):
#     """Normalise un vecteur à l'aide du scaler."""
#     return scaler.transform([vector])[0]

# # Fonction pour réduire la dimension d'un vecteur
# def reduce_dimension(vector, pca):
#     """Réduit la dimension d'un vecteur à l'aide du PCA."""
#     return pca.transform([vector])[0]

# # Fonction pour prédire le label avec distances
# def predict_label_with_distance(feature_vector, classifier, scaler, pca, threshold=0.7):
#     normalized_vector = normalize_vector(feature_vector, scaler)
#     reduced_vector = reduce_dimension(normalized_vector, pca)

#     distances, indices = classifier.kneighbors([reduced_vector])
#     avg_distance = np.mean(distances[0])
#     probabilities = classifier.predict_proba([reduced_vector])[0]
#     max_prob = max(probabilities)

#     print(f"Probabilités : {probabilities}")
#     print(f"Max prob : {max_prob}, Distance moyenne : {avg_distance}")

#     if max_prob >= threshold and avg_distance < 200.:  # Ajuste si nécessaire
#         label = classifier.classes_[np.argmax(probabilities)]
#         return label, max_prob, avg_distance
#     else:
#         return None, max_prob, avg_distance

# # Prédire le label
# label, confidence, avg_distance = predict_label_with_distance(example_vector, classifier, scaler, pca, threshold=0.7)
# if label:
#     print(f"Label prédit : {label} avec une confiance de {confidence:.2f}")
# else:
#     print("Aucune correspondance avec un niveau de confiance suffisant.")


import numpy as np
import joblib

# Charger le modèle
model_path = "classifier_model.pkl"
classifier = joblib.load(model_path)

# Fonction pour prédire le label d'un vecteur
def predict_label(feature_vector, classifier, threshold=0.4):
    probabilities = classifier.predict_proba([feature_vector])[0]
    max_prob = max(probabilities)
    #print(f"Max prob : {max_prob}")
    # if max_prob >= threshold:
    #     label = classifier.classes_[np.argmax(probabilities)]
    #     return label, max_prob
    # else:
    #     return None, max_prob
    label = classifier.classes_[np.argmax(probabilities)]
    return label, max_prob


def load_vec(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            vector = np.array([float(x) for x in parts[0:]])
    return vector


example_vector=load_vec('acc_vector_lbph.dat')




# Prédire le label
label, confidence = predict_label(example_vector, classifier)
# if label:
#     print(f"{label} : {confidence:.2f}")
# else:
#     print("0")
print(f"{label} : {confidence:.2f}")

