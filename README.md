# GTI411 - Laboratoire 2 : Imagerie Numerique et Filtrage

## Description du projet

Ce projet s'inscrit dans le cadre du cours GTI411. L'objectif principal est de developper un systeme capable de detecter et d'identifier des panneaux de signalisation routiere dans des images complexes.

Conformement aux exigences du laboratoire, aucune methode basee sur l'apprentissage profond (Deep Learning, YOLO, etc.) n'est utilisee. La detection repose exclusivement sur des algorithmes de vision par ordinateur classiques offerts par la bibliotheque OpenCV.

Le projet est divise en deux parties :

- **Partie 1** : Detection manuelle. L'utilisateur selectionne une region d'interet (ROI) sur l'image et le programme identifie le panneau present dans cette zone.
- **Partie 2** : Detection automatique. Le programme scanne l'image entiere et identifie tous les panneaux de maniere autonome.

## Architecture

Afin de maintenir un code propre, lisible et facile a maintenir, le projet respecte le patron d'architecture MVC (Modele-Vue-Controleur) :

- **models/** : Contient la logique d'acces aux donnees et de traitement.
  - `chargeur_images.py` : Gestion du chargement des images de test et des templates de reference.
  - `similarite.py` : Algorithme de reconnaissance de panneaux basé sur le filtrage spatial (amplitude du gradient de Sobel), l'analyse des couleurs en espace HSV, et le template matching multi-échelle (corrélation normalisée).
- **views/** : Gere l'interface utilisateur.
  - `interface.py` : Interface graphique principale developpee avec Tkinter.
  - `affichage.py` : Fonctions utilitaires pour le rendu visuel avec OpenCV (dessin de boites englobantes, texte).
- **controllers/** : Orchestre la communication entre les vues et les modeles.
  - `partie1.py` : Logique de la selection manuelle et du traitement des evenements de la souris.
  - `partie2.py` : Logique d'analyse globale de l'image pour la detection automatique.

## Pre-requis et Installation

Assurez-vous d'avoir Python installe sur votre machine. Il est recommande d'utiliser un environnement virtuel.

1. Creation et activation de l'environnement virtuel :
   Pour Windows :

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   Pour Linux/MacOS :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Installation des dependances :
   ```bash
   pip install opencv-python numpy
   ```

## Execution

Pour lancer l'application, executez simplement le point d'entree a la racine du projet :

```bash
python main.py
```

Une fenetre graphique s'ouvrira, vous permettant de choisir une image de test et de selectionner le mode de detection (Partie 1 ou Partie 2).
