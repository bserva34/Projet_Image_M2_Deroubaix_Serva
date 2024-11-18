import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import yaml

# Charger les données depuis un fichier YAML
def load_data(file_path):
    features = []
    labels = []
    
    with open(file_path, 'r') as f:
        # Charger le contenu du fichier YAML
        content = yaml.safe_load(f)
        
        # Parcourir chaque label et ses images avec vecteurs associés
        for label, items in content.items():
            # Vérifier si le label a au moins 2 vecteurs
            if len(items) >= 2:
                for item in items:
                    # Extraire le vecteur pour chaque image et le convertir en np.array
                    vector = np.array(item['vector'], dtype=float)
                    features.append(vector)
                    labels.append(label)
            else:
                print(f"Label '{label}' ignoré : moins de 2 vecteurs associés.")
    
    return np.array(features), np.array(labels)


# Définir le chemin du fichier de données
file_path = "../labeled_facesV16_11_2024_final_visage_non_aligner.yml"

# Charger les données depuis le fichier YAML
features, labels = load_data(file_path)

# Diviser les données en ensembles d'entraînement et de test
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Créer et entraîner le modèle
classifier = KNeighborsClassifier(n_neighbors=3z)  # Tu peux ajuster le paramètre n_neighbors
classifier.fit(X_train, y_train)

# Évaluer le modèle
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Sauvegarder le modèle
model_path = "classifier_model.pkl"
joblib.dump(classifier, model_path)
print(f"Modèle sauvegardé sous {model_path}")
