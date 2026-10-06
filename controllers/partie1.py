import cv2
from models.similarite import trouver_meilleur_match
from views.fenetre import FenetreImage


# les clics accidentels plus petits que ça (en pixels de l'image) sont ignorés
TAILLE_MIN_SELECTION = 15


class DetectionManuelle:
    """
    Partie 1 : sélection manuelle des panneaux à la souris.
    On utilise un mouse callback pour gérer le rectangle
    et la fenêtre (views/fenetre.py) pour l'affichage.
    """

    def __init__(self, chargeur):
        self.chargeur = chargeur
        self.en_train_de_dessiner = False
        self.point_depart = None
        self.point_fin = None
        self.selection_faite = False

    def _callback_souris(self, event, x, y):
        """Gère les events souris (x, y sont déjà en pixels de l'image)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.en_train_de_dessiner = True
            self.point_depart = (x, y)
            self.point_fin = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.en_train_de_dessiner:
            self.point_fin = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.en_train_de_dessiner:
            self.en_train_de_dessiner = False
            self.point_fin = (x, y)
            self.selection_faite = True

    def _rectangle(self):
        """Rectangle (x, y, w, h) entre les 2 points, même si on dessine de droite à gauche."""
        x1, y1 = self.point_depart
        x2, y2 = self.point_fin
        return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)

    def lancer(self):
        """Boucle principale de la détection manuelle."""
        if not self.chargeur.templates:
            print("Pas de templates, impossible de continuer")
            return

        if self.chargeur.image_courante is None:
            print("Aucune image chargée")
            return

        image = self.chargeur.image_courante
        titre = "Partie 1 - Selection manuelle (Echap: quitter | F: plein ecran | C: effacer)"
        fenetre = FenetreImage(titre, image, self._callback_souris)
        fenetre.ouvrir()

        print("\n=== PARTIE 1 : Détection manuelle ===")
        print("Dessinez un rectangle autour d'un panneau")
        print("Echap ou bouton X pour quitter, F pour le plein écran, C pour effacer\n")

        while True:
            # on affiche le rectangle en cours de dessin (en vert)
            en_cours = None
            if self.en_train_de_dessiner:
                en_cours = self._rectangle()

            fenetre.afficher(en_cours)
            touche = fenetre.lire_touche()

            if touche == 27 or not fenetre.est_ouverte():  # echap ou bouton X
                break

            # traitement de la sélection
            if self.selection_faite:
                self.selection_faite = False
                x, y, w, h = self._rectangle()

                # ignorer les clics accidentels (trop petits)
                if w < TAILLE_MIN_SELECTION or h < TAILLE_MIN_SELECTION:
                    continue

                # extraire le roi depuis l'image source (pas celle avec les annotations)
                roi = image[y:y+h, x:x+w]

                nom_match, score = trouver_meilleur_match(roi, self.chargeur.templates)

                if nom_match is None:
                    # score trop bas -> probablement pas un panneau
                    print(f"  -> Aucun panneau détecté (score max: {score:.2f})")
                    fenetre.ajouter_annotation((x, y, w, h), f"Inconnu ({score:.2f})", (0, 0, 220))
                else:
                    print(f"  -> {nom_match} (score: {score:.2f})")
                    fenetre.ajouter_annotation((x, y, w, h), f"{nom_match} ({score:.2f})")

        fenetre.fermer()
