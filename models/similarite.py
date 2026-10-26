import cv2
import numpy as np


# en dessous de ce score on considère qu'on n'a rien trouvé (panneau inconnu)
SEUIL_DETECTION = 0.33

# largeur de référence pour comparer les images à la même échelle
TAILLE_BASE = 100

# si le rapport largeur/hauteur du roi et du template diffère de plus de 40%,
# le score est divisé par 2 (zone mal découpée ou pas la bonne forme)
ECART_RATIO_MAX = 1.4


def signature_couleur(image):
    """
    Pourcentage de pixels de chaque couleur dans l'image.
    On travaille en HSV pour que l'éclairage ne fausse pas les couleurs :
    rouge, orange, jaune, vert, bleu, noir, blanc/gris.
    """
    # flou léger pour éviter que le bruit du capteur fausse la couleur d'un pixel
    flou = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(flou, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # on isole les pixels qui ont une vraie couleur (assez saturés et assez clairs)
    colore = (s > 90) & (v > 70)

    classes = [
        colore & ((h < 8) | (h >= 165)),    # rouge (fait le tour à 180 dans opencv)
        colore & (h >= 8) & (h < 20),       # orange
        colore & (h >= 20) & (h < 35),      # jaune
        colore & (h >= 35) & (h < 85),      # vert
        colore & (h >= 85) & (h < 135),     # bleu
        (~colore) & (v < 90),               # noir
        (~colore) & (v >= 90),              # blanc / gris clair
    ]
    return np.array([float(c.mean()) for c in classes])


def gradient_sobel(image):
    """
    Calcule l'amplitude du gradient avec le filtre de Sobel.
    Ça permet de faire ressortir les formes, les flèches, les lettres
    et les chiffres, peu importe si la photo est sombre ou claire.
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    # norme du vecteur gradient : sqrt(gx^2 + gy^2)
    mag = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def correlation_multiechelle(roi, template):
    """
    Fait glisser le template dans le roi (ou l'inverse) à plusieurs échelles.
    Comme le rectangle sélectionné a souvent un peu de marge ou est mal centré,
    tester plusieurs échelles (de 70% à 130%) permet de trouver la vraie correspondance.
    On compare les couleurs (BGR) et la forme des contours (Sobel).
    """
    # template ramené à 100 px de large en gardant ses proportions
    tw = TAILLE_BASE
    th = max(10, int(TAILLE_BASE * template.shape[0] / template.shape[1]))
    tmpl_bgr = cv2.GaussianBlur(cv2.resize(template, (tw, th)), (5, 5), 0)
    tmpl_sob = cv2.GaussianBlur(cv2.resize(gradient_sobel(template), (tw, th)), (5, 5), 0)

    roi_sob_brut = gradient_sobel(roi)

    meilleur_bgr = 0.0
    meilleur_sob = 0.0

    for echelle in [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]:
        rw = max(10, int(tw * echelle))
        rh = max(10, int(th * echelle))

        roi_bgr = cv2.GaussianBlur(cv2.resize(roi, (rw, rh)), (5, 5), 0)
        roi_sob = cv2.GaussianBlur(cv2.resize(roi_sob_brut, (rw, rh)), (5, 5), 0)

        # si le roi est plus grand, on fait glisser le template dedans
        if rw >= tw and rh >= th:
            res_b = cv2.matchTemplate(roi_bgr, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
            res_s = cv2.matchTemplate(roi_sob, tmpl_sob, cv2.TM_CCOEFF_NORMED)
        # si le template est plus grand, on fait glisser le roi dedans
        elif tw >= rw and th >= rh:
            res_b = cv2.matchTemplate(tmpl_bgr, roi_bgr, cv2.TM_CCOEFF_NORMED)
            res_s = cv2.matchTemplate(tmpl_sob, roi_sob, cv2.TM_CCOEFF_NORMED)
        else:
            # cas mixte : on remet le roi à la taille du template
            r_b = cv2.resize(roi_bgr, (tw, th))
            r_s = cv2.resize(roi_sob, (tw, th))
            res_b = cv2.matchTemplate(r_b, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
            res_s = cv2.matchTemplate(r_s, tmpl_sob, cv2.TM_CCOEFF_NORMED)

        meilleur_bgr = max(meilleur_bgr, float(np.max(res_b)))
        meilleur_sob = max(meilleur_sob, float(np.max(res_s)))

    return meilleur_bgr, meilleur_sob


def calculer_similarite(roi, template):
    """
    Score final entre 0 et 1 : à quel point le roi ressemble au template.
    1. On vérifie la compatibilité des couleurs vives en HSV.
    2. On combine 70% de contours (Sobel) et 30% d'apparence couleur (BGR).
    3. On divise par 2 si les proportions sont trop différentes.
    """
    # 1. couleurs en HSV
    sig_roi = signature_couleur(roi)
    sig_tmpl = signature_couleur(template)

    vives_tmpl = float(sig_tmpl[:5].sum())
    if vives_tmpl >= 0.15:
        # panneau coloré : les couleurs du template doivent être présentes dans le roi
        communes = float(np.minimum(sig_roi[:5], sig_tmpl[:5]).sum())
        ressemblance_couleur = communes / vives_tmpl
    else:
        # panneau noir et blanc (40, do not pass, keep right) :
        # si le roi est très coloré (ex: rouge vif), ce n'est pas un panneau noir et blanc
        vives_roi = float(sig_roi[:5].sum())
        if vives_roi > 0.35:
            ressemblance_couleur = 0.1
        else:
            ressemblance_couleur = 1.0

    # 2. corrélation multi-échelle (Sobel + BGR)
    corr_bgr, corr_sob = correlation_multiechelle(roi, template)

    # les arêtes de Sobel sont plus fiables (70%), la couleur confirme (30%)
    score = (0.70 * corr_sob + 0.30 * corr_bgr) * ressemblance_couleur

    # 3. proportions largeur / hauteur
    ratio_roi = roi.shape[1] / roi.shape[0]
    ratio_tmpl = template.shape[1] / template.shape[0]
    ecart = max(ratio_roi, ratio_tmpl) / min(ratio_roi, ratio_tmpl)
    if ecart > ECART_RATIO_MAX:
        score = score / 2

    return float(np.clip(score, 0.0, 1.0))


def trouver_meilleur_match(roi, templates):
    """
    Compare le roi à tous les templates et retourne le meilleur match.
    Si le score reste sous le seuil, retourne None (panneau inconnu ou faux positif).
    """
    meilleur_nom = None
    meilleur_score = -1.0

    for nom, template in templates.items():
        score = calculer_similarite(roi, template)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_nom = nom

    if meilleur_score < SEUIL_DETECTION:
        return None, meilleur_score

    return meilleur_nom, meilleur_score
