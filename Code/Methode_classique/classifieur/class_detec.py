import numpy as np
import joblib

# Charger le modèle
model_path = "classifier_model.pkl"
classifier = joblib.load(model_path)

# Fonction pour prédire le label d'un vecteur
def predict_label(feature_vector, classifier, threshold=0.4):
    probabilities = classifier.predict_proba([feature_vector])[0]
    max_prob = max(probabilities)
    # print(f"Max prob : {max_prob}")
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
#     print(f"Label prédit : {label} avec une confiance de {confidence:.2f}")
# else:
#     print("Aucune correspondance avec un niveau de confiance suffisant.")

print(f"{label} : {confidence:.2f}")