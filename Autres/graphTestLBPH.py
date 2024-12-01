import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import re

# Fonction pour lire les données depuis un fichier texte
def read_input_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extraire les informations nécessaires (image, label, distance)
            match = re.match(r"(.+)\s->\s(.+)\s:\s([\d.]+)", line.strip())
            if match:
                image_name = match.group(1)
                label = match.group(2)
                distance = float(match.group(3))
                results.append({"image": image_name, "label": label, "distance": distance})
    return results

# Fonction pour extraire le nom de base avant "_chiffre.jpg"
def extract_name(image_name):
    match = re.match(r"(.+)_\d+\.jpg", image_name)
    return match.group(1).lower() if match else image_name.lower()

# Fonction principale pour afficher le tableau
def display_results(file_path):
    # Lire les résultats depuis le fichier
    results = read_input_file(file_path)
    
    # Trier les résultats par distance (du plus petit au plus grand)
    results.sort(key=lambda x: x['distance'])

    # Préparer les données pour le tableau
    colors = []
    table_data = []
    for result in results:
        image_name = result["image"]
        label = result["label"]
        distance = result["distance"]
        image_base_name = extract_name(image_name)

        # Déterminer la couleur de fond pour chaque cellule
        if label.lower() == image_base_name:
            row_colors = ['#138727'] * 3  # Vert pour match
        elif image_base_name.startswith("inconnu"):
            row_colors = ['#f4e07f'] * 3  # Jaune pour "inconnu"
        else:
            row_colors = ['#f93c2a'] * 3  # Rouge pour non-match

        # Ajouter les données de cette ligne
        table_data.append([image_name, label, f"{distance:.4f}"])
        colors.append(row_colors)

    # Ajuster dynamiquement la hauteur de la figure en fonction du nombre de lignes
    num_rows = len(table_data)
    fig_height = max(6, 0.3 * num_rows)  # 0.3 unités de hauteur par ligne, minimum de 6

    # Créer la figure avec une hauteur dynamique
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Désactiver les axes
    ax.axis('off')

    # Créer le tableau
    table = ax.table(
        cellText=table_data,
        cellColours=colors,
        colLabels=["Image", "Label prédit", "Distance"],
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
    legend_colors = ['#138727', '#f4e07f', '#f93c2a']
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]

    plt.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05)  # Positionner la légende sous le tableau
    )

    # Ajouter un titre
    plt.title("Résultats de correspondance d'image avec labels", fontsize=14)

    # Ajuster la taille des colonnes en fonction du contenu
    table.auto_set_column_width([0, 1, 2])

    # Afficher le tableau
    plt.show()


# Appeler la fonction principale avec un fichier d'entrée
display_results('res_lbph_test.txt')
