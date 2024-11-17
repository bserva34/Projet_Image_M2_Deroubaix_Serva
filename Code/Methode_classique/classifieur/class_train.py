import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
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

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Créer et entraîner le modèle
classifier = KNeighborsClassifier(n_neighbors=3)  # Tu peux ajuster le paramètre n_neighbors
classifier.fit(X_train, y_train)

# Évaluer le modèle
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Sauvegarder le modèle
model_path = "classifier_model.pkl"
joblib.dump(classifier, model_path)
print(f"Modèle sauvegardé sous {model_path}")


