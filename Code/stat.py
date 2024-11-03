import re
import statistics

def extraire_pourcentages(nom_fichier):
    pourcentages = []
    
    with open(nom_fichier, 'r') as fichier:
        for ligne in fichier:
            matches = re.findall(r'(\d+\.\d+)%', ligne)
            pourcentages.extend([float(valeur) for valeur in matches])
    
    return pourcentages

def analyse_pourcentages(pourcentages):
    if not pourcentages:
        return None, None, None, None
    
    moyenne = sum(pourcentages) / len(pourcentages)
    max_valeur = max(pourcentages)
    min_valeur = min(pourcentages)
    mediane = statistics.median(pourcentages)
    
    return moyenne, max_valeur, min_valeur, mediane

# Nom du fichier à lire
nom_fichier = 'res_lbph.txt'  # Remplacez par le chemin de votre fichier

# Extraction des pourcentages
pourcentages = extraire_pourcentages(nom_fichier)

if pourcentages:
    moyenne, max_valeur, min_valeur, mediane = analyse_pourcentages(pourcentages)
    
    print(f"Moyenne: {moyenne:.2f}%")
    print(f"Max: {max_valeur:.2f}%")
    print(f"Min: {min_valeur:.2f}%")
    print(f"Médiane: {mediane:.2f}%")
else:
    print("Aucune valeur trouvée dans le fichier.")
