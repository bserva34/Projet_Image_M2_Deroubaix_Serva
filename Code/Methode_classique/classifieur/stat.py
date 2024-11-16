import numpy as np

# Chemin vers le fichier de résultats
file_path = "results.txt"

def parse_results(file_path):
    values = []
    similar_count = 0
    different_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Extraire les parties de la ligne
                parts = line.strip().split(' ')
                name_before_underscore = parts[0].split('_')[0]
                name_after_arrow = parts[2].strip(':')
                value = float(parts[-1])
                
                # Ajouter la valeur pour les statistiques
                values.append(value)
                
                # Comparer les noms
                if name_before_underscore == name_after_arrow:
                    similar_count += 1
                else:
                    different_count += 1
            except Exception as e:
                print(f"Erreur lors du traitement de la ligne : {line.strip()}. Erreur : {e}")

    # Calculer les statistiques
    values = np.array(values)
    stats = {
        "moyenne": np.mean(values),
        "min": np.min(values),
        "max": np.max(values),
        "mediane": np.median(values),
    }
    
    return stats, similar_count, different_count

# Appeler la fonction et afficher les résultats
stats, similar_count, different_count = parse_results(file_path)

print(f"Moyenne : {stats['moyenne']:.2f}")
print(f"Min : {stats['min']:.2f}")
print(f"Max : {stats['max']:.2f}")
print(f"Médiane : {stats['mediane']:.2f}")
print()
print(f"Similaires : {similar_count}")
print(f"Différents : {different_count}")
