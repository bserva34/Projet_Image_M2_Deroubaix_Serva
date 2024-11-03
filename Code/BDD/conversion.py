import os
from PIL import Image

# Chemin du dossier contenant les images
dossier_images = "archive_2"  # Remplace par le chemin de ton dossier

# Créer un dossier de sortie pour les images .pgm
dossier_sortie = os.path.join(dossier_images, "archive2_pgm")
os.makedirs(dossier_sortie, exist_ok=True)

# Parcours des fichiers du dossier
for fichier in os.listdir(dossier_images):
    chemin_complet = os.path.join(dossier_images, fichier)
    
    try:
        # Ouvrir l'image
        image = Image.open(chemin_complet)
        
        # Conversion en mode 'L' pour un format PGM en niveaux de gris
        image = image.convert("L")
        
        # Nom du fichier de sortie avec l'extension .pgm
        fichier_pgm = fichier + ".pgm"  # Garde le nom de base et ajoute .pgm
        chemin_sortie = os.path.join(dossier_sortie, fichier_pgm)
        
        # Sauvegarder l'image en .pgm
        image.save(chemin_sortie, "PPM")
        print(f"Converted {fichier} to {fichier_pgm}")
        
    except Exception as e:
        print(f"Could not convert {fichier}: {e}")

print("Conversion terminée !")
