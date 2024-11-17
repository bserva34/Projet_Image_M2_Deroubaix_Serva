#!/bin/bash

clear

# Dossier contenant les images à tester
input_folder="BDD/40_personnes_test"
output_folder="res_lbph"
result_file="res_lbph.txt"

# Créer le dossier de sortie s'il n'existe pas
mkdir -p "$output_folder"

# Vider le fichier de résultats s'il existe déjà
> "$result_file"

# Parcourir toutes les images dans le dossier
for input_image in "$input_folder"/*; do
    # Extraire le nom de fichier sans le chemin
    input_image_name=$(basename "$input_image")

    # Chemin de sortie pour l'image enregistrée avec l'extension .png
    base_name="${input_image_name%.pgm}"  # Enlève l'extension .pgm
    output_image_path="$output_folder/result_$base_name.png"

    # Exécuter le script Python avec les arguments et capturer la sortie
    output=$(python3 lbph_cv2_detec.py "$input_image" "$output_image_path" 2>&1)

    # Vérifier le code de retour de la commande précédente
    if [ $? -eq 0 ]; then
        # Extraire le label et le pourcentage de confiance
        label=$(echo "$output" | grep -oP "Label:\s*\K[0-9]+")
        confidence=$(echo "$output" | grep -oP "Confiance:\s*\K[0-9]+.[0-9]+")

        if [ -n "$label" ] && [ -n "$confidence" ]; then
            echo "Image testée: $input_image_name, Résultat enregistré: $output_image_path, Label: $label, Confiance: $confidence%" >> "$result_file"
            echo "Image $input_image_name traitée avec succès. Confiance: $confidence%."
        else
            echo "Erreur : impossible d'extraire le label et la confiance pour $input_image_name." >> "$result_file"
        fi
    else
        echo "Erreur lors du traitement de $input_image_name" >> "$result_file"
    fi
done
