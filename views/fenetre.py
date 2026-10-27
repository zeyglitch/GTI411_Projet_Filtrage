import cv2
import numpy as np
from views.affichage import generer_couleur, dessiner_resultats


class FenetreImage:
    """
    Fenêtre OpenCV qui affiche une image et ses annotations.
    L'image est redimensionnée à la taille de la fenêtre en gardant ses proportions,
    et les coordonnées de la souris sont reconverties en pixels de l'image.
    F = plein écran, C = effacer les cadres.
    """

    def __init__(self, titre, image, callback_souris=None):
        self.titre = titre
        self.image = image
        self.callback_souris = callback_souris
        self.annotations = []    # liste de (rect, texte, couleur) en pixels de l'image
        self.plein_ecran = False

        # taille de départ : l'image réduite pour rentrer dans un écran de 1280x800
        h, w = image.shape[:2]
        ratio = min(1.0, 1280 / w, 800 / h)
        self.taille = (int(w * ratio), int(h * ratio))

        # échelle et décalage de l'image dans la fenêtre (pour convertir les clics)
        self.echelle = ratio
        self.decalage = (0, 0)

    def ouvrir(self):
        # WINDOW_NORMAL pour pouvoir redimensionner la fenêtre à la main
        cv2.namedWindow(self.titre, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(self.titre, self.taille[0], self.taille[1])
        cv2.setMouseCallback(self.titre, self._souris)

    def est_ouverte(self):
        """False si la fenêtre a été fermée avec le bouton X."""
        try:
            return cv2.getWindowProperty(self.titre, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def fermer(self):
        cv2.destroyWindow(self.titre)

    def ajouter_annotation(self, rect, texte, couleur=None):
        if couleur is None:
            couleur = generer_couleur()
        self.annotations.append((rect, texte, couleur))

    def changer_image(self, nouvelle_image, annotations=None):
        """Permet de changer dynamiquement l'image et les annotations."""
        self.image = nouvelle_image
        if annotations is not None:
            self.annotations = list(annotations)

    def afficher(self, rect_en_cours=None):
        """Affiche l'image. rect_en_cours = (x, y, w, h) du rectangle qu'on est en train de tirer."""
        try:
            _, _, largeur, hauteur = cv2.getWindowImageRect(self.titre)
        except cv2.error:   # fenêtre déjà fermée
            largeur, hauteur = self.taille
        if largeur < 50 or hauteur < 50:   # fenêtre réduite : on garde l'ancienne taille
            largeur, hauteur = self.taille
        self.taille = (largeur, hauteur)

        # image centrée dans la fenêtre, bandes grises autour
        ih, iw = self.image.shape[:2]
        echelle = min(largeur / iw, hauteur / ih)
        nw, nh = int(iw * echelle), int(ih * echelle)
        ox, oy = (largeur - nw) // 2, (hauteur - nh) // 2
        self.echelle = echelle
        self.decalage = (ox, oy)

        ecran = np.full((hauteur, largeur, 3), 40, np.uint8)
        ecran[oy:oy + nh, ox:ox + nw] = cv2.resize(self.image, (nw, nh))

        # les cadres sont dessinés après le resize pour que le texte reste net
        a_dessiner = []
        for (x, y, w, h), texte, couleur in self.annotations:
            rect = (int(x * echelle) + ox, int(y * echelle) + oy, int(w * echelle), int(h * echelle))
            a_dessiner.append((rect, texte, couleur))
        dessiner_resultats(ecran, a_dessiner)

        # rectangle vert pendant qu'on dessine à la souris
        if rect_en_cours is not None:
            x, y, w, h = rect_en_cours
            p1 = (int(x * echelle) + ox, int(y * echelle) + oy)
            p2 = (int((x + w) * echelle) + ox, int((y + h) * echelle) + oy)
            cv2.rectangle(ecran, p1, p2, (0, 255, 0), 2)

        cv2.imshow(self.titre, ecran)

    def _souris(self, evenement, x, y, flags, param):
        if self.callback_souris is None:
            return
        # point de la fenêtre -> pixel de l'image d'origine (limité à l'image)
        ih, iw = self.image.shape[:2]
        ox, oy = self.decalage
        px = int((x - ox) / self.echelle)
        py = int((y - oy) / self.echelle)
        px = max(0, min(px, iw - 1))
        py = max(0, min(py, ih - 1))
        self.callback_souris(evenement, px, py)

    def lire_touche(self):
        """Attend 30 ms une touche. Gère F (plein écran) et C (effacer). Retourne la touche."""
        touche = cv2.waitKey(30) & 0xFF

        if touche in (ord('f'), ord('F')):
            self.plein_ecran = not self.plein_ecran
            mode = cv2.WINDOW_FULLSCREEN if self.plein_ecran else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.titre, cv2.WND_PROP_FULLSCREEN, mode)
        elif touche in (ord('c'), ord('C')):
            self.annotations = []

        return touche
