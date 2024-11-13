# import re
# import statistics

# def extraire_pourcentages(nom_fichier):
#     pourcentages = []
    
#     with open(nom_fichier, 'r') as fichier:
#         for ligne in fichier:
#             matches = re.findall(r'(\d+\.\d+)%', ligne)
#             pourcentages.extend([float(valeur) for valeur in matches])
    
#     return pourcentages

# def analyse_pourcentages(pourcentages):
#     if not pourcentages:
#         return None, None, None, None
    
#     moyenne = sum(pourcentages) / len(pourcentages)
#     max_valeur = max(pourcentages)
#     min_valeur = min(pourcentages)
#     mediane = statistics.median(pourcentages)
    
#     return moyenne, max_valeur, min_valeur, mediane

# # Nom du fichier à lire
# nom_fichier = 'res_lbph.txt'  # Remplacez par le chemin de votre fichier

# # Extraction des pourcentages
# pourcentages = extraire_pourcentages(nom_fichier)

# if pourcentages:
#     moyenne, max_valeur, min_valeur, mediane = analyse_pourcentages(pourcentages)
    
#     print(f"Moyenne: {moyenne:.2f}%")
#     print(f"Max: {max_valeur:.2f}%")
#     print(f"Min: {min_valeur:.2f}%")
#     print(f"Médiane: {mediane:.2f}%")
# else:
#     print("Aucune valeur trouvée dans le fichier.")

import re
import statistics

def extraire_distances(nom_fichier):
    """Extrait les valeurs de distance des résultats dans un fichier texte."""
    distances = []
    
    with open(nom_fichier, 'r') as fichier:
        for ligne in fichier:
            # Chercher un motif de distance à la fin de la ligne (exemple : "distance de 1.3771785702535027")
            match = re.search(r'distance de ([\d.]+)$', ligne)
            if match:
                # Ajouter la valeur numérique dans la liste des distances
                distances.append(float(match.group(1)))
    
    return distances

def analyse_distances(distances):
    """Calcule les statistiques de la liste des distances."""
    if not distances:
        return None, None, None, None
    
    moyenne = sum(distances) / len(distances)
    max_valeur = max(distances)
    min_valeur = min(distances)
    mediane = statistics.median(distances)
    
    return moyenne, max_valeur, min_valeur, mediane

# Nom du fichier à lire
nom_fichier = 'res_lbph_v2.txt'  # Remplacez par le chemin de votre fichier

# Extraction des valeurs de distance
distances = extraire_distances(nom_fichier)

# Calcul des statistiques
if distances:
    moyenne, max_valeur, min_valeur, mediane = analyse_distances(distances)
    
    print(f"Moyenne: {moyenne:.2f}")
    print(f"Max: {max_valeur:.2f}")
    print(f"Min: {min_valeur:.2f}")
    print(f"Médiane: {mediane:.2f}")
else:
    print("Aucune valeur de distance trouvée dans le fichier.")
