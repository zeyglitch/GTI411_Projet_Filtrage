import os
import sys

# on ajoute le dossier du projet au path pour les imports entre packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from models.chargeur_images import ChargeurImages
from controllers.partie1 import DetectionManuelle
from controllers.partie2 import DetectionAuto
from views.interface import InterfacePrincipale


def main():
    # chemins vers les images
    racine = os.path.dirname(os.path.abspath(__file__))
    dossier_templates = os.path.join(racine, "images", "templates")
    dossier_tests = os.path.join(racine, "images", "tests_img")

    # chargement des templates
    chargeur = ChargeurImages(dossier_templates, dossier_tests)
    images_test = chargeur.lister_images_test()

    def lancer_partie1(selection):
        """Callback quand on clique sur Partie 1 dans l'interface."""
        # si c'est un chemin absolu -> image externe (parcourir)
        if os.path.isabs(selection):
            chargeur.charger_image_chemin(selection)
        else:
            chargeur.charger_image(selection)

        if chargeur.image_courante is not None:
            detecteur = DetectionManuelle(chargeur)
            detecteur.lancer()

    def lancer_partie2(selection):
        """Callback quand on clique sur Partie 2 dans l'interface."""
        if os.path.isabs(selection):
            chargeur.charger_image_chemin(selection)
        else:
            chargeur.charger_image(selection)

        if chargeur.image_courante is not None:
            detecteur = DetectionAuto(chargeur)
            detecteur.lancer()

    # on lance l'interface tkinter
    interface = InterfacePrincipale(images_test, lancer_partie1, lancer_partie2)
    interface.lancer()


if __name__ == "__main__":
    main()
