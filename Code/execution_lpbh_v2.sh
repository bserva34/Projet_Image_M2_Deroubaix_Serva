#!/bin/bash

clear

# Dossier contenant les images à tester
input_folder="BDD/40_personnes_test"
result_file="res_lbph_v2.txt"

# Vider le fichier de résultats s'il existe déjà
> "$result_file"

# Parcourir toutes les images dans le dossier
for input_image in "$input_folder"/*; do
    # Extraire le nom de fichier sans le chemin
    input_image_name=$(basename "$input_image")

    # Exécuter le script Python avec seulement l'argument de l'image d'entrée et capturer la sortie
    output=$(python3 lbph_cv2_detec_v2.py "$input_image" 2>&1)

    # Vérifier le code de retour de la commande précédente
    if [ $? -eq 0 ]; then
        # Écrire directement la sortie du script Python dans le fichier de résultats
        echo "$output" >> "$result_file"
        echo "Image $input_image_name traitée avec succès."
    else
        echo "Erreur lors du traitement de $input_image_name" >> "$result_file"
    fi
done
