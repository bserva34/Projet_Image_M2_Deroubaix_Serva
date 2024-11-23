import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay

# Chemin du fichier résultat
result_file = "resultatTestCNN.dat"

# Seuils à tester
thresholds = [0.05, 0.06, 0.07, 0.08]

# Fonction pour extraire le début d'un nom
def extract_name(filename):
    return filename.split("_")[0].lower()

# Initialisation des matrices de confusion
confusion_matrices = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

# Charger les résultats depuis le fichier
try:
    with open(result_file, "r") as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"Le fichier {result_file} n'existe pas.")
    exit()

# Analyser chaque ligne du fichier
for line in lines:
    # Exemple de ligne : Adam_9.jpg Adam 0.0259
    parts = line.strip().split()
    if len(parts) != 3:
        continue  # Ignorer les lignes mal formées

    filename, label, distance_str = parts
    distance = float(distance_str)
    actual_name = extract_name(filename)
    is_unknown = actual_name.startswith("inconnu")

    # Traiter chaque seuil
    for threshold in thresholds:
        if is_unknown:
            # Cas où l'image est inconnue
            if distance > threshold:
                confusion_matrices[threshold]["TN"] += 1  # Correctement rejetée
            else:
                confusion_matrices[threshold]["FP"] += 1  # Incorrectement acceptée
        else:
            # Cas où l'image est connue
            if distance <= threshold:
                if actual_name == label.lower():
                    confusion_matrices[threshold]["TP"] += 1  # Correctement identifiée
                else:
                    confusion_matrices[threshold]["FP"] += 1  # Mauvais label
            else:
                confusion_matrices[threshold]["FN"] += 1  # Non reconnue

# Fonction pour générer et enregistrer des plots de matrices de confusion
def save_confusion_matrix_plot(matrix, threshold):
    # Construire la matrice de confusion
    confusion_matrix = np.array([
        [matrix["TP"], matrix["FN"]],
        [matrix["FP"], matrix["TN"]]
    ])
    
    # Calcul des métriques
    TP = matrix["TP"]
    FP = matrix["FP"]
    FN = matrix["FN"]
    TN = matrix["TN"]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Labels des axes
    display_labels = ["Positif", "Négatif"]
    
    # Création du plot
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    
    # Ajouter un titre avec le seuil
    plt.title(f"Matrice de Confusion (Threshold: {threshold})")
    
    # Ajouter les scores (Précision, Rappel, F1) sur l'image, au-dessus de la matrice
    metrics_text = (
        f"Précision: {precision:.2f}\n"
        f"Rappel: {recall:.2f}\n"
        f"F1-Score: {f1_score:.2f}"
    )
    # Placer le texte au-dessus de la matrice sans chevaucher le titre
    plt.figtext(
        0.5, 0.95, metrics_text, 
        ha='center', va='top', fontsize=12, color="darkblue"
    )
    
    # Enregistrer l'image
    output_filename = f"confusion_matrix_threshold_{str(threshold).replace('.', '')}.png"
    plt.savefig(output_filename)
    print(f"Image enregistrée : {output_filename}")
    plt.close()

# Générer et enregistrer les plots pour chaque seuil
for threshold, matrix in confusion_matrices.items():
    save_confusion_matrix_plot(matrix, threshold)
