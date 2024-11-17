#!/bin/bash

# Chemin vers le dossier contenant les images
input_folder="../../BDD/40_personnes_test" # Remplace par le chemin réel
# Fichier de résultats
result_file="results.txt"

# Vider le fichier de résultats s'il existe déjà
> "$result_file"

# Parcourir toutes les images dans le dossier
for input_image in "$input_folder"/*; do
    # Extraire le nom de fichier sans le chemin
    input_image_name=$(basename "$input_image")

    # Appeler le script vector_lbph.py avec le chemin de l'image
    python3 vector_lbph.py "$input_image"

    # Vérifier le code de retour pour voir si vector_lbph a bien fonctionné
    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'exécution de vector_lbph pour $input_image_name" >> "$result_file"
        continue
    fi

    # Appeler le script class_detec.py avec le chemin de l'image
    output=$(python3 class_detec.py "$input_image" 2>&1)

    # Vérifier le code de retour pour voir si class_detec a bien fonctionné
    if [ $? -eq 0 ]; then
        # Écrire le nom de l'image et le résultat dans le fichier
        echo "$input_image_name -> $output" >> "$result_file"
        echo "Image $input_image_name traitée avec succès."
    else
        # En cas d'erreur, écrire un message dans le fichier de résultats
        echo "$input_image_name -> Erreur lors du traitement" >> "$result_file"
        echo "Erreur lors du traitement de $input_image_name."
    fi
done

echo "Traitement terminé. Résultats enregistrés dans $result_file."
