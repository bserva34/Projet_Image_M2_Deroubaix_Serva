import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Charger les données
def load_data(file_path):
    features = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            label = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            labels.append(label)
            features.append(vector)
    return np.array(features), np.array(labels)


file_path = "../v2/vectors_both.dat"
features, labels = load_data(file_path)

# # Diviser les données
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Normalisation
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Réduction de dimensions avec PCA
# pca = PCA(n_components=50)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# # Modèle k-NN
# classifier = KNeighborsClassifier(n_neighbors=7)
# classifier.fit(X_train, y_train)

# # Évaluer avec validation croisée
# scores = cross_val_score(classifier, X_train, y_train, cv=5)
# print(f"Validation croisée : Précision moyenne {np.mean(scores) * 100:.2f}%")

# # Sauvegarder le modèle
# joblib.dump((classifier, scaler, pca), "classifier_model.pkl")
# print("Modèle sauvegardé.")

features, labels = load_data(file_path)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
classifier = KNeighborsClassifier(n_neighbors=7)  # Tu peux ajuster le paramètre n_neighbors
classifier.fit(X_train, y_train)

# Évaluer le modèle
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Sauvegarder le modèle
model_path = "classifier_model.pkl"
joblib.dump(classifier, model_path)
print(f"Modèle sauvegardé sous {model_path}")