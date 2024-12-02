from cx_Freeze import setup, Executable
import os

# Spécifiez le fichier principal de votre application
script_principal = "detec_face.py"

# Liste des fichiers supplémentaires (fichiers .yml et .dat)
include_files = [
    'labeled_faces_withCeleb.yml',
    'lbph_bdd.dat',
]

# Configuration de l'exécutable
executables = [
    Executable(script_principal, base=None)  # base=None pour console, "Win32GUI" pour GUI
]

# Configuration de cx_Freeze
setup(
    name="MonAppli",
    version="0.1",
    description="Une application Python convertie en exécutable",
    options={
        "build_exe": {
            "packages": ["os"],  # Ajoutez ici tous les packages nécessaires (par ex. "cv2", "numpy")
            "include_files": include_files,  # Liste des fichiers supplémentaires
        }
    },
    executables=executables
)
