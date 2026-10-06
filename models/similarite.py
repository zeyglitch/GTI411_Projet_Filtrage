import cv2
import numpy as np


# en dessous de ce seuil on considère qu'on n'a rien trouvé
SEUIL_DETECTION = 0.12


def calculer_similarite(roi, template):
    """
    Compare un ROI avec un template en combinant 2 approches complémentaires :
    1) Approche "Classique" (Couleur + Forme) : très bien pour les panneaux simples et colorés (Stop, Yield)
    2) Approche "ORB" (Features) : indispensable pour les panneaux textuels (Do Not Pass, Keep Right)
       qui sont sinon confondus avec le fond blanc/sable.
    """
    
    # --- 1) SCORE ORB (très robuste pour le texte) ---
    orb = cv2.ORB_create(500)
    kp_roi, des_roi = orb.detectAndCompute(roi, None)
    kp_tmpl, des_tmpl = orb.detectAndCompute(template, None)
    
    score_orb = 0.0
    if des_roi is not None and des_tmpl is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_roi, des_tmpl)
        # on ne garde que les matchs très sûrs (faible distance)
        good_matches = [m for m in matches if m.distance < 50]
        # On plafonne le score à 1.0 (12 bons matchs = score parfait)
        score_orb = min(1.0, len(good_matches) / 12.0)

    # --- 2) SCORE CLASSIQUE (Histogramme HSV + Canny + Pixel direct) ---
    # resize le roi à la taille du template
    roi_resized = cv2.resize(roi, (template.shape[1], template.shape[0]))

    # flou gaussien pour réduire le bruit (vu en cours)
    roi_blur = cv2.GaussianBlur(roi_resized, (5, 5), 0)
    tmpl_blur = cv2.GaussianBlur(template, (5, 5), 0)

    # a. score couleur (HSV, on ignore V)
    roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    tmpl_hsv = cv2.cvtColor(tmpl_blur, cv2.COLOR_BGR2HSV)

    hist_roi = cv2.calcHist([roi_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_tmpl = cv2.calcHist([tmpl_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_roi, hist_roi)
    cv2.normalize(hist_tmpl, hist_tmpl)

    # max(0, ...) car TM_CCOEFF_NORMED et HISTCMP_CORREL peuvent être négatifs
    score_couleur = max(0, cv2.compareHist(hist_roi, hist_tmpl, cv2.HISTCMP_CORREL))

    # b. score forme (Canny + matchTemplate)
    roi_gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl_blur, cv2.COLOR_BGR2GRAY)

    roi_canny = cv2.Canny(roi_gray, 50, 150)
    tmpl_canny = cv2.Canny(tmpl_gray, 50, 150)
    score_forme = max(0, np.max(cv2.matchTemplate(roi_canny, tmpl_canny, cv2.TM_CCOEFF_NORMED)))

    # c. score pixel-à-pixel direct
    res_direct = cv2.matchTemplate(roi_blur, tmpl_blur, cv2.TM_CCOEFF_NORMED)
    score_direct = max(0, np.max(res_direct))

    # on combine les 3 scores classiques
    score_classique = 0.35 * score_couleur + 0.35 * score_forme + 0.30 * score_direct

    # --- RÉSULTAT FINAL ---
    # On prend le maximum entre l'approche classique et ORB.
    # ORB "sauve" les panneaux avec beaucoup de texte, 
    # et l'approche classique gère bien les formes simples !
    return max(score_classique, score_orb)


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
