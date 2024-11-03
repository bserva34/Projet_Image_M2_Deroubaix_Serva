import os
import shutil

# Définir le répertoire de destination
destination_dir = 'output'

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Parcourir les dossiers s1 à s40
for i in range(1, 41):
    source_dir = f's{i}'
    if os.path.exists(source_dir) and os.path.isdir(source_dir):
        # Obtenir la liste des fichiers dans le dossier
        files = [file_name for file_name in os.listdir(source_dir) 
                 if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm'))]
        
        # Vérifier s'il y a des fichiers à traiter
        if files:
            # Trier les fichiers par nom pour s'assurer que la première et la dernière image sont bien celles voulues
            files.sort()

            # Prendre la première et la dernière image
            first_image = files[0]
            last_image = files[-1]

            # Définir les chemins source pour la première et la dernière image
            first_image_path = os.path.join(source_dir, first_image)
            last_image_path = os.path.join(source_dir, last_image)

            # Copier la première image dans le dossier de destination
            shutil.copy(first_image_path, os.path.join(destination_dir, f's{i}_first_{first_image}'))
            print(f"Copiée: {first_image} depuis {source_dir} vers {destination_dir}")

            # Copier la dernière image dans le dossier de destination
            shutil.copy(last_image_path, os.path.join(destination_dir, f's{i}_last_{last_image}'))
            print(f"Copiée: {last_image} depuis {source_dir} vers {destination_dir}")

            # Supprimer les images d'origine
            os.remove(first_image_path)
            os.remove(last_image_path)
            print(f"Supprimée: {first_image} et {last_image} de {source_dir}")

print("Opération terminée : première et dernière images copiées et supprimées de chaque dossier.")
