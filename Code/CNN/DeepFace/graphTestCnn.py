import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch
import re

# Fonction pour lire le fichier .dat et extraire les données
def read_dat_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extraction des informations depuis la ligne
            parts = line.split()  # On suppose que les données sont séparées par des espaces
            image_name = parts[0]
            label = parts[1]
            distance = float(parts[2])
            closest_image = parts[3]  # Nom de l'image correspondante
            label_count = int(parts[4])  # Extraction du nombre d'images
            
            results.append({"image": image_name, "label": label, "distance": distance, 
                            "closest_image": closest_image, "label_count": label_count})
    
    return results

# Fonction pour extraire le nom de base avant "_chiffre.jpg"
def extract_name(image_name):
    match = re.match(r"(.+)_\d+\.jpg", image_name)
    return match.group(1).lower() if match else image_name.lower()

# Lire les résultats depuis le fichier .dat
results = read_dat_file('resultatTestCNN.dat')  # Remplacez par le chemin de votre fichier .dat

# Trier les résultats par distance (du plus petit au plus grand)
results.sort(key=lambda x: x['distance'])

# Préparation des couleurs et des données pour le tableau
colors = []
table_data = []
for result in results:
    image_name = result["image"]
    label = result["label"]
    closest_image = result["closest_image"]
    label_count = result["label_count"]
    image_base_name = extract_name(image_name)  # Utiliser la nouvelle fonction d'extraction

    # Déterminer la couleur de fond pour chaque cellule
    if label.lower() == image_base_name:
        row_colors = ['#89f068'] * 4  # Vert pour match
    elif image_base_name.startswith("inconnu") or image_base_name.startswith("inconnu2"):
        row_colors = ['#f4e07f'] * 4  # Jaune pour "inconnu"
    else:
        row_colors = ['#f93c2a'] * 4  # Rouge pour non-match
    
    # Ajouter un dégradé de bleu pour le nombre d'images du label dans la dernière case
    blue_gradient = plt.cm.Blues((label_count - min([r['label_count'] for r in results])) / 
                                 (max([r['label_count'] for r in results]) - min([r['label_count'] for r in results])))

    row_colors.append(blue_gradient)  # Dégradé de bleu pour le nombre d'images
    
    # Ajouter les données de cette ligne
    table_data.append([image_name, label, f"{result['distance']:.4f}", closest_image, label_count])
    colors.append(row_colors)

# Ajuster dynamiquement la hauteur de la figure en fonction du nombre de lignes
num_rows = len(table_data)
fig_height = max(6, 0.3 * num_rows)  # 0.3 unités de hauteur par ligne, minimum de 6

# Créer la figure avec une hauteur dynamique
fig, ax = plt.subplots(figsize=(12, fig_height))

# Désactiver les axes
ax.axis('off')

# Créer le tableau
table = ax.table(
    cellText=table_data,
    cellColours=colors,
    colLabels=["Image", "Label prédit", "Distance", "Image correspondante", "Nombre d'images"],
    loc='center'
)

# Ajuster l'apparence du tableau
table.auto_set_font_size(False)
table.set_fontsize(8)  # Réduire la taille de la police pour les tableaux très grands
table.scale(1.2, 1.2)  # Échelle ajustée

# Centrer le texte dans toutes les cellules
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(ha='center', va='center')
    cell.set_fontsize(8)  # Garder la police cohérente

# Ajouter la légende des couleurs
legend_labels = ['Reconnu', 'Inconnu', 'Pas Reconnu']
legend_colors = ['#89f068', '#f4e07f', '#f93c2a']
legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]

plt.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=3,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.05)  # Positionner la légende sous le tableau
)

# Ajouter une barre de couleur pour le dégradé bleu
sm = plt.cm.ScalarMappable(
    cmap="Blues",
    norm=mcolors.Normalize(vmin=min([r['label_count'] for r in results]), vmax=max([r['label_count'] for r in results]))
)
sm.set_array([])

fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',
    fraction=0.02,
    pad=0.08,
    label="Nombre d'images du label dans la BDD"
)

# Ajouter un titre
plt.title("Résultats de correspondance d'image avec labels", fontsize=14)

# Ajuster la taille des colonnes en fonction du contenu
table.auto_set_column_width([0, 1, 2, 3, 4])

# Afficher le tableau
plt.show()
