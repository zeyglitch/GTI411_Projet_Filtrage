import cv2
import numpy as np
from views.affichage import generer_couleur, dessiner_resultats


FOND = (40, 40, 40)   # couleur des bandes autour de l'image


class FenetreImage:
    """
    Fenêtre OpenCV redimensionnable qui affiche une image avec ses annotations.
    - l'image est mise à l'échelle pour remplir la fenêtre en gardant ses
      proportions et elle est centrée (bandes sombres sur les côtés)
    - les cadres et textes sont dessinés à la taille de la fenêtre, donc
      ils restent nets et lisibles même en plein écran
    - les coordonnées de la souris sont reconverties en pixels de l'image
    Touches : F = plein écran, C = effacer les annotations.
    """

    def __init__(self, titre, image, callback_souris=None, largeur_max=1280, hauteur_max=800):
        self.titre = titre
        self.image = image
        self.callback_souris = callback_souris
        self.annotations = []    # liste de (rect, texte, couleur) en pixels de l'image
        self.plein_ecran = False

        # taille de départ : l'image réduite pour rentrer à l'écran (jamais agrandie)
        h, w = image.shape[:2]
        ratio = min(1.0, largeur_max / w, hauteur_max / h)
        self.taille = (max(200, int(w * ratio)), max(200, int(h * ratio)))

        # position de l'image dans la fenêtre au dernier affichage
        self.echelle = ratio
        self.decalage = (0, 0)
        self._cache = None

    # ---------- ouverture / fermeture ----------

    def ouvrir(self):
        # FREERATIO : la fenêtre ne force pas le ratio de l'image (c'est notre
        # canvas qui gère les proportions), sinon la taille lue se dérègle
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

    # ---------- annotations ----------

    def ajouter_annotation(self, rect, texte, couleur=None):
        if couleur is None:
            couleur = generer_couleur()
        self.annotations.append((rect, texte, couleur))

    def effacer_annotations(self):
        self.annotations = []

    # ---------- dessin ----------

    def _taille_fenetre(self):
        """Taille actuelle de la zone d'affichage (si on ne peut pas la lire, la dernière connue)."""
        try:
            _, _, w, h = cv2.getWindowImageRect(self.titre)
        except cv2.error:
            return self.taille
        # fenêtre réduite dans la barre des tâches -> w et h valent 0
        if w < 50 or h < 50:
            return self.taille
        self.taille = (w, h)
        return self.taille

    def construire(self, largeur, hauteur):
        """
        Crée l'image à afficher pour une fenêtre largeur x hauteur :
        fond sombre, image centrée et mise à l'échelle, puis annotations.
        """
        ih, iw = self.image.shape[:2]
        echelle = min(largeur / iw, hauteur / ih)
        nw = max(1, int(iw * echelle))
        nh = max(1, int(ih * echelle))
        ox = (largeur - nw) // 2
        oy = (hauteur - nh) // 2

        # INTER_AREA est meilleur quand on rétrécit, LINEAR quand on agrandit
        methode = cv2.INTER_AREA if echelle < 1 else cv2.INTER_LINEAR
        fond = np.full((hauteur, largeur, 3), FOND, np.uint8)
        fond[oy:oy + nh, ox:ox + nw] = cv2.resize(self.image, (nw, nh), interpolation=methode)

        self.echelle = echelle
        self.decalage = (ox, oy)

        # les annotations sont dessinées après le redimensionnement (texte net)
        a_dessiner = []
        for (x, y, w, h), texte, couleur in self.annotations:
            rect_ecran = (int(x * echelle) + ox, int(y * echelle) + oy,
                          int(w * echelle), int(h * echelle))
            a_dessiner.append((rect_ecran, texte, couleur))
        dessiner_resultats(fond, a_dessiner)
        return fond

    def afficher(self, rect_en_cours=None):
        """Affiche l'image. rect_en_cours = (x, y, w, h) du rectangle qu'on est en train de tirer."""
        largeur, hauteur = self._taille_fenetre()

        # on ne recalcule que si la fenêtre ou les annotations ont changé
        cle = (largeur, hauteur, len(self.annotations))
        if self._cache is None or self._cache[0] != cle:
            self._cache = (cle, self.construire(largeur, hauteur))

        ecran = self._cache[1].copy()
        if rect_en_cours is not None:
            x, y, w, h = rect_en_cours
            ox, oy = self.decalage
            p1 = (int(x * self.echelle) + ox, int(y * self.echelle) + oy)
            p2 = (int((x + w) * self.echelle) + ox, int((y + h) * self.echelle) + oy)
            cv2.rectangle(ecran, p1, p2, (0, 255, 0), 2)

        cv2.imshow(self.titre, ecran)

    # ---------- souris et clavier ----------

    def ecran_vers_image(self, x, y):
        """Convertit un point de la fenêtre en pixel de l'image (limité à l'image)."""
        ih, iw = self.image.shape[:2]
        ox, oy = self.decalage
        px = int((x - ox) / self.echelle)
        py = int((y - oy) / self.echelle)
        return max(0, min(px, iw - 1)), max(0, min(py, ih - 1))

    def _souris(self, evenement, x, y, flags, param):
        if self.callback_souris is not None:
            px, py = self.ecran_vers_image(x, y)
            self.callback_souris(evenement, px, py)

    def lire_touche(self):
        """Attend 30 ms une touche. Gère F (plein écran) et C (effacer). Retourne la touche."""
        touche = cv2.waitKey(30) & 0xFF

        if touche in (ord('f'), ord('F')):
            self.plein_ecran = not self.plein_ecran
            mode = cv2.WINDOW_FULLSCREEN if self.plein_ecran else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.titre, cv2.WND_PROP_FULLSCREEN, mode)
        elif touche in (ord('c'), ord('C')):
            self.effacer_annotations()

        return touche
