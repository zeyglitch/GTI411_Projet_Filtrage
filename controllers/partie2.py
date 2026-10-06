import cv2
import numpy as np
from models.similarite import trouver_meilleur_match
from views.affichage import dessiner_resultats_multiples


class DetectionAuto:
    """
    Partie 2 du lab : détection automatique des panneaux.
    On isole les zones d'intérêt avec filtrage HSV + seuillage,
    puis contours, NMS et matching avec les templates.
    """

    def __init__(self, chargeur):
        self.chargeur = chargeur

    def lancer(self):
        """Lance la détection auto sur l'image courante."""
        if not self.chargeur.templates:
            print("Pas de templates chargés")
            return

        if self.chargeur.image_courante is None:
            print("Aucune image chargée")
            return

        print("\n=== PARTIE 2 : Détection automatique ===")
        print("Analyse en cours...\n")

        # TODO: à implémenter
        # étapes prévues :
        # 1. convertir en HSV et isoler les couleurs (rouge, bleu, jaune...)
        # 2. seuillage pour les panneaux blancs
        # 3. combiner les masques
        # 4. findContours pour extraire les régions
        # 5. NMS pour virer les boîtes en doublon
        # 6. matcher chaque zone avec les templates

        print("Partie 2 pas encore implémentée, ça arrive!")
