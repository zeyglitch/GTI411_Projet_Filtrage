import cv2
import numpy as np


# en dessous de ce score on considère qu'on n'a rien trouvé (panneau inconnu)
SEUIL_DETECTION = 0.30

# taille (en pixels) du plus grand côté pour comparer les images
TAILLE_ORB = 200
TAILLE_GLISSANT = 120


def signature_couleur(image):
    """
    Pourcentage de pixels de chaque "classe de couleur" dans l'image :
    rouge, orange, jaune, vert, bleu, noir, blanc/gris.
    On travaille en HSV (la teinte ne dépend presque pas de la lumière).
    Retourne un tableau de 7 valeurs qui additionnent 1.
    """
    flou = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(flou, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    colore = (s > 90) & (v > 70)

    classes = [
        colore & ((h < 8) | (h >= 165)),    # rouge
        colore & (h >= 8) & (h < 20),       # orange
        colore & (h >= 20) & (h < 35),      # jaune
        colore & (h >= 35) & (h < 85),      # vert
        colore & (h >= 85) & (h < 135),     # bleu
        (~colore) & (v < 90),               # noir
        (~colore) & (v >= 90),              # blanc / gris
    ]
    return np.array([c.mean() for c in classes])


def compter_matchs_orb(roi, template):
    """
    Compte les points ORB du roi qui retrouvent leur place dans le template.
    ORB repère des coins (lettres, flèches...) et ne dépend pas de l'échelle.
    Un match compte seulement s'il passe 3 tests :
      1. test du ratio : le meilleur match est nettement meilleur que le 2e
      2. il est mutuel : le template choisit aussi ce point du roi
      3. cohérence géométrique (RANSAC) : tous les matchs doivent suivre la
         même transformation (décalage + rotation + échelle). Les matchs
         au hasard entre deux textes différents ne la respectent pas.
    Retourne le nombre de matchs cohérents (inliers).
    """
    # même taille pour les deux, en gardant les proportions du template
    echelle = TAILLE_ORB / max(template.shape[:2])
    taille = (int(template.shape[1] * echelle), int(template.shape[0] * echelle))
    roi_gris = cv2.cvtColor(cv2.resize(roi, taille), cv2.COLOR_BGR2GRAY)
    tmpl_gris = cv2.cvtColor(cv2.resize(template, taille), cv2.COLOR_BGR2GRAY)

    # bord et patch plus petits que par défaut (31) : sinon ORB ne trouve presque
    # rien dans les petites zones (le bord de 31 pixels est ignoré)
    orb = cv2.ORB_create(500, edgeThreshold=15, patchSize=15)
    kp_roi, des_roi = orb.detectAndCompute(roi_gris, None)
    kp_tmpl, des_tmpl = orb.detectAndCompute(tmpl_gris, None)

    if des_roi is None or des_tmpl is None or len(des_tmpl) < 2:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    paires = bf.knnMatch(des_roi, des_tmpl, k=2)
    retour = bf.match(des_tmpl, des_roi)

    pts_roi = []
    pts_tmpl = []
    for i, p in enumerate(paires):
        if len(p) < 2:
            continue
        if p[0].distance < 0.75 * p[1].distance and retour[p[0].trainIdx].trainIdx == i:
            pts_roi.append(kp_roi[i].pt)
            pts_tmpl.append(kp_tmpl[p[0].trainIdx].pt)

    # il faut au moins 4 points pour estimer la transformation
    if len(pts_roi) < 4:
        return 0

    _, masque = cv2.estimateAffinePartial2D(np.float32(pts_roi), np.float32(pts_tmpl),
                                            method=cv2.RANSAC, ransacReprojThreshold=4)
    if masque is None:
        return 0
    return int(masque.sum())


def correlation_glissante(roi, template):
    """
    matchTemplate qui "glisse" le template dans un roi un peu plus grand
    (25 % de plus). Comme ça on tolère une zone sélectionnée un peu décalée.
    Retourne la meilleure corrélation entre 0 et 1.
    """
    echelle = TAILLE_GLISSANT / max(template.shape[:2])
    tw = int(template.shape[1] * echelle)
    th = int(template.shape[0] * echelle)

    tmpl_petit = cv2.GaussianBlur(cv2.resize(template, (tw, th)), (5, 5), 0)
    roi_grand = cv2.GaussianBlur(cv2.resize(roi, (int(tw * 1.25), int(th * 1.25))), (5, 5), 0)

    resultat = cv2.matchTemplate(roi_grand, tmpl_petit, cv2.TM_CCOEFF_NORMED)
    # max(0, ...) car la corrélation peut être négative
    return max(0.0, float(np.max(resultat)))


def calculer_similarite(roi, template):
    """
    Score entre 0 et 1 : à quel point le roi ressemble au template.
    Deux sources de preuves, on garde la meilleure :

    1) Score ORB (panneaux avec du texte ou des détails : Stop, Do Not Enter...)
       = nombre de matchs géométriquement cohérents.
    2) Score visuel (panneaux simples : Yield, Work...)
       = composition des couleurs x corrélation du template glissant.
       Seulement pour les templates colorés : pour un panneau noir et blanc
       la couleur ne donne aucune information (tous les panneaux noir et
       blanc se ressemblent).

    Deux garde-fous pour éviter de prendre un panneau inconnu pour un template :
    - si les couleurs vives du template ne sont pas dans le roi,
      on divise le score ORB par 2
    - si les proportions (largeur / hauteur) sont très différentes, on divise
      le score final par 2 (zone mal découpée ou panneau coupé par le bord)
    """
    # --- couleurs : est-ce qu'on retrouve les couleurs vives du template dans le roi ? ---
    # (0 à 1) : un template rouge avec un roi sans rouge donne 0.
    # On ignore le noir et le blanc : ils sont dans presque tous les panneaux.
    sig_roi = signature_couleur(roi)
    sig_tmpl = signature_couleur(template)
    template_colore = sig_tmpl[:5].sum() >= 0.2
    if template_colore:
        ressemblance_couleur = float(np.minimum(sig_roi[:5], sig_tmpl[:5]).sum() / sig_tmpl[:5].sum())
    else:
        ressemblance_couleur = 1.0

    # --- 1) score ORB ---
    nb_matchs = compter_matchs_orb(roi, template)
    score_orb = float(np.clip((nb_matchs - 3) / 12, 0, 1))
    if ressemblance_couleur < 0.4:
        score_orb *= 0.5

    # --- 2) score visuel (seulement si le template a des couleurs) ---
    score_visuel = 0.0
    if template_colore:
        corr = correlation_glissante(roi, template)
        score_visuel = ressemblance_couleur * float(np.clip((corr - 0.15) / 0.45, 0, 1))

    score = max(score_orb, score_visuel)

    # --- proportions ---
    ratio_roi = roi.shape[1] / roi.shape[0]
    ratio_tmpl = template.shape[1] / template.shape[0]
    if abs(np.log(ratio_roi / ratio_tmpl)) > 0.45:
        score *= 0.5

    return score


def trouver_meilleur_match(roi, templates):
    """
    Teste le roi contre tous les templates.
    Si le meilleur score est en dessous du seuil, retourne None.
    """
    meilleur_nom = None
    meilleur_score = -1

    for nom, template in templates.items():
        score = calculer_similarite(roi, template)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_nom = nom

    if meilleur_score < SEUIL_DETECTION:
        return None, meilleur_score

    return meilleur_nom, meilleur_score
