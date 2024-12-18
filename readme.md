# Projet de Reconnaissance Faciale

Ce projet implémente une application de reconnaissance faciale utilisant DeepFace (CNN) et LBPH.

## Prérequis

```bash
  pip install deepface
  pip install opencv-python
  pip install numpy==1.23.3
  pip install scipy
  pip install kivy
  pip install pyyaml
  pip install tensorflow==2.10.0
```

## Application

Pour lancer l'application de reconnaissance faciale, exécutez ce script Python :
```bash
  python3 Application/detec_face.py
```

## Fonctionnalité

  - Reconnaissance faciale par CNN (deepface/dlib) sur une base de donnée local (YML) avec un seuil variable. 
  - Reconnaissance faciale par LBPH (opencv) sur une base de donnée local (dat) avec un seuil variable. 
  - 2 Modes: Caméra & Image importé. 
  - Option surveillance pour récuperer l'heure.
  - Ajout de visage au YML intégré à l'application
  - Screenshot de l'image courante. 
  
## Ajout de visage à la base de donnée

Pour LBPH: avoir les imagettes triées et labélisées de taille 100*100px formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp.
```bash
  python3 Application/lbph_bdd_generation.py dossier_avec_imagette_a_ajouter
```

Pour CNN : 
 - Directement dans l'application
 - Via un autre code python qui prend un dossier d'image :n'importe quelle photo fonctionne, avec les formats suivants : .jpg, .jpeg, .png, .bmp, .tiff, .webp.
```bash
  mkdir Application/Img_to_embed
  #Ajouter les images des visages à ajouter dans le YML au dossier créé
  python3 Application/deepFaceVecCaracInYML.py 
```
Ouverture d'une application pour labélisé les visages: entrer le nom pour labéliser, cancel pour terminer et vide pour ignorer.
