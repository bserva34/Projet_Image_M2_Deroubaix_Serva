# Projet CNN avec Deepface

Ce projet implémente un système de reconnaissance faciale basé sur un réseau de neurones convolutionnel (CNN) en utilisant la bibliothèque Deepface. Le projet utilise également `opencv-python`

## Prérequis

Avant de commencer, assurez-vous d'avoir `Python 3` installé sur votre machine.

## Installation

Les bibliothèques nécessaires peuvent être installées avec les commandes suivantes :

```bash
pip install deepface
pip install opencv-python  


```
# Projet CNN avec InsightFace

Ce projet implémente un système de reconnaissance faciale basé sur un réseau de neurones convolutionnel (CNN) en utilisant la bibliothèque InsightFace. Le projet utilise également `numpy`, `opencv-python`, et `onnxruntime` pour l'inférence optimisée.

## Prérequis

Avant de commencer, assurez-vous d'avoir `Python 3` installé sur votre machine.

## Installation

Les bibliothèques nécessaires peuvent être installées avec les commandes suivantes :

```bash
pip install insightface[torch]   # Installe InsightFace avec support pour PyTorch
pip install torch torchvision    # Installe PyTorch et torchvision
pip install numpy                # Installe NumPy
pip install numpy==1.19.5        # Installe NumPy
pip install opencv-python        # Installe OpenCV pour la gestion des images
pip install onnxruntime          # Installe ONNX Runtime pour l'inférence
pip install onnxruntime-gpu      # Si vous souhaitez utiliser un GPU pour l'inférence
