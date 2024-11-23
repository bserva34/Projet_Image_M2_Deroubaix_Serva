import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch

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
    image_base_name = image_name.split('_')[0].lower()  # Extraire la partie avant le "_"

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

# Création du tableau avec matplotlib
fig, ax = plt.subplots(figsize=(12, 8))  # Augmenter la taille pour tenir compte de la nouvelle colonne

# Désactiver les axes
ax.axis('off')

# Créer le tableau
table = ax.table(cellText=table_data, cellColours=colors, colLabels=["Image", "Label prédit", "Distance", "Image correspondante", "Nombre d'images"], loc='center')

# Ajuster l'apparence du tableau
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Centrer le texte dans toutes les cellules
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(ha='center', va='center')  # Centrer horizontalement et verticalement
    cell.set_fontsize(10)  # Fixer la taille de la police

# Ajuster la position du titre plus haut
plt.subplots_adjust(top=0.85, bottom=0.2)  # Ajuster l'espace en bas pour laisser place à la légende

# Ajouter une légende pour la couleur des cases (Vert, Jaune, Rouge)
legend_labels = ['Reconnu', 'Inconnu', 'Pas Reconnu']
legend_colors = ['#89f068', '#f4e07f', '#f93c2a']
legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]

# Placer la légende sous le tableau, en dehors de la zone du graphique
plt.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

# Ajouter une légende pour la couleur du dégradé de bleu
sm = plt.cm.ScalarMappable(cmap="Blues", norm=mcolors.Normalize(vmin=min([r['label_count'] for r in results]), vmax=max([r['label_count'] for r in results])))

sm.set_array([])

# Afficher la légende du dégradé bleu
fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.02, pad=0.04, label="Nombre d'images du label dans la BDD")

# Ajouter un titre
plt.title("Résultats de correspondance d'image avec labels", fontsize=14)

# Ajuster la taille des colonnes en fonction du contenu (texte ou titres les plus longs)
table.auto_set_column_width([0, 1, 2, 3, 4])  # Ajuster la largeur des colonnes selon le contenu

# Afficher le graphique
plt.show()
