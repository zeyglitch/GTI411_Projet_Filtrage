import cv2
from models.similarite import trouver_meilleur_match
from views.affichage import dessiner_resultat


# taille max pour l'affichage (pour que l'image rentre dans l'écran)
MAX_LARGEUR = 1200
MAX_HAUTEUR = 800


def redimensionner_si_necessaire(image):
    """
    Si l'image est trop grande pour l'écran, on la resize
    en gardant le ratio (proportions).
    Retourne l'image (éventuellement resizée) et le ratio appliqué.
    """
    h, w = image.shape[:2]

    if w <= MAX_LARGEUR and h <= MAX_HAUTEUR:
        return image, 1.0

    ratio = min(MAX_LARGEUR / w, MAX_HAUTEUR / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(image, (new_w, new_h))
    return resized, ratio


class DetectionManuelle:
    """
    Partie 1 : sélection manuelle des panneaux à la souris.
    On utilise un mouse callback pour gérer le rectangle
    et la fermeture de la fenêtre.
    """

    def __init__(self, chargeur):
        self.chargeur = chargeur
        self.en_train_de_dessiner = False
        self.point_depart = None
        self.point_fin = None
        self.selection_faite = False

    def _callback_souris(self, event, x, y, flags, param):
        """Gère les events souris pour dessiner le rectangle."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.en_train_de_dessiner = True
            self.point_depart = (x, y)
            self.point_fin = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.en_train_de_dessiner:
            self.point_fin = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.en_train_de_dessiner = False
            self.point_fin = (x, y)
            self.selection_faite = True

    def lancer(self):
        """Boucle principale de la détection manuelle."""
        if not self.chargeur.templates:
            print("Pas de templates, impossible de continuer")
            return

        if self.chargeur.image_courante is None:
            print("Aucune image chargée")
            return

        # si l'image est trop grande, on la redimensionne pour l'affichage
        img_source, ratio = redimensionner_si_necessaire(self.chargeur.image_courante)
        img_affichage = img_source.copy()

        nom_fenetre = "Partie 1 - Selection manuelle (Echap pour quitter)"

        cv2.namedWindow(nom_fenetre, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(nom_fenetre, self._callback_souris)

        print("\n=== PARTIE 1 : Détection manuelle ===")
        print("Dessinez un rectangle autour d'un panneau")
        print("Echap ou bouton X pour quitter\n")

        while True:
            # on affiche le rectangle en cours de dessin (en vert)
            img_temp = img_affichage.copy()
            if self.en_train_de_dessiner and self.point_depart and self.point_fin:
                cv2.rectangle(img_temp, self.point_depart, self.point_fin, (0, 255, 0), 2)

            cv2.imshow(nom_fenetre, img_temp)
            key = cv2.waitKey(30) & 0xFF

            # on vérifie si la fenêtre a été fermée (bouton X)
            try:
                if cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == 27:  # echap
                break

            # traitement de la sélection
            if self.selection_faite:
                self.selection_faite = False

                x1, y1 = self.point_depart
                x2, y2 = self.point_fin
                # normaliser les coords (au cas ou on dessine de droite à gauche)
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                # ignorer les clics accidentels (trop petits)
                if w < 15 or h < 15:
                    continue

                # extraire le roi depuis l'image source (pas celle avec les annotations)
                roi = img_source[y:y+h, x:x+w]

                # chercher le meilleur match
                nom_match, score = trouver_meilleur_match(roi, self.chargeur.templates)

                if nom_match is None:
                    # score trop bas -> probablement pas un panneau
                    print(f"  -> Aucun panneau détecté (score max: {score:.2f})")
                    dessiner_resultat(img_affichage, (x, y, w, h),
                                      "Inconnu", score, couleur=(0, 0, 200))
                else:
                    print(f"  -> {nom_match} (score: {score:.2f})")
                    dessiner_resultat(img_affichage, (x, y, w, h), nom_match, score)

        cv2.destroyAllWindows()
