import cv2
import numpy as np
import random


def generer_couleur():
    # couleur aléatoire assez vive pour bien se voir sur la photo
    return (random.randint(0, 255), random.randint(100, 255), random.randint(100, 255))


def taille_texte(largeur, hauteur):
    """Taille de la police selon la taille de la fenêtre (pas trop petit, pas énorme)."""
    return max(0.45, min(0.8, min(largeur, hauteur) / 1400))


def se_chevauchent(a, b):
    """True si les rectangles a et b = (x1, y1, x2, y2) se touchent."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def dessiner_etiquette(image, rect, texte, couleur, etiquettes):
    """
    Écrit le texte au-dessus du cadre, sur un fond noir pour qu'il soit lisible.
    Si ça chevauche une étiquette déjà dessinée (panneaux proches), on le met en dessous.
    """
    hauteur_img, largeur_img = image.shape[:2]
    x, y, w, h = rect

    echelle = taille_texte(largeur_img, hauteur_img)
    epaisseur = 1 if echelle < 0.7 else 2
    (tw, th), base = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)

    tx = max(0, min(x, largeur_img - tw - 8))
    ty = y - base - 4                     # au-dessus
    zone = (tx, ty - th - 4, tx + tw + 8, ty + base)
    if ty - th - 4 < 0 or any(se_chevauchent(zone, e) for e in etiquettes):
        ty = min(y + h + th + 6, hauteur_img - base)   # en dessous
        zone = (tx, ty - th - 4, tx + tw + 8, ty + base)

    cv2.rectangle(image, zone[:2], zone[2:], (0, 0, 0), -1)
    cv2.putText(image, texte, (tx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, epaisseur)
    etiquettes.append(zone)


def dessiner_resultats(image, resultats):
    """Dessine chaque (rect, texte, couleur) : le cadre puis son étiquette."""
    echelle = taille_texte(image.shape[1], image.shape[0])
    epaisseur = 2 if echelle < 0.7 else 3

    etiquettes = []
    for (x, y, w, h), texte, couleur in resultats:
        cv2.rectangle(image, (x, y), (x + w, y + h), couleur, epaisseur)
        dessiner_etiquette(image, (x, y, w, h), texte, couleur, etiquettes)
    return image


def creer_mosaique_etapes(image, masques_couleurs, masque_nettoye, boites_candidates, resultats_finaux):
    """
    Construit une mosaïque 2x2 montrant les étapes clés de la détection automatique :
    1. Masques HSV bruts des couleurs
    2. Masque nettoyé par morphologie mathématique (Fermeture + Ouverture)
    3. Boîtes candidates extraites des contours et filtrées
    4. Résultat final après corrélation et seuillage
    """
    # 1. Masques HSV bruts sur fond désaturé
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fond_gris = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR) // 3
    visuel_hsv = fond_gris.copy()
    couleurs_bgr = [
        (0, 0, 255),    # rouge
        (0, 140, 255),  # orange
        (0, 230, 255),  # jaune
        (230, 120, 0),  # bleu
        (240, 240, 240) # blanc
    ]
    for m, c in zip(masques_couleurs, couleurs_bgr):
        visuel_hsv[m > 0] = c

    # 2. Nettoyage morphologique en vert émeraude
    visuel_morpho = fond_gris.copy()
    visuel_morpho[masque_nettoye > 0] = (50, 255, 100)

    # 3. Boîtes candidates
    visuel_boites = image.copy()
    for i, (bx, by, bw, bh) in enumerate(boites_candidates, 1):
        cv2.rectangle(visuel_boites, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        cv2.putText(visuel_boites, f"Candidate #{i}", (bx, max(20, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # 4. Résultat final
    visuel_final = image.copy()
    a_dessiner = [(rect, f"{nom} ({score:.2f})", (0, 255, 0)) for rect, nom, score in resultats_finaux]
    dessiner_resultats(visuel_final, a_dessiner)

    # Sous-fonction pour créer un panneau propre avec son bandeau de titre
    def creer_panneau(img_source, titre):
        pw, ph = 620, 360
        resized = cv2.resize(img_source, (pw, ph - 35))
        panneau = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.rectangle(panneau, (0, 0), (pw, 35), (20, 20, 20), -1)
        cv2.putText(panneau, titre, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        panneau[35:ph, 0:pw] = resized
        cv2.rectangle(panneau, (0, 0), (pw - 1, ph - 1), (80, 80, 80), 1)
        return panneau

    p1 = creer_panneau(visuel_hsv, "1. Masques HSV (Rouge, Orange, Jaune, Bleu, Blanc)")
    p2 = creer_panneau(visuel_morpho, "2. Nettoyage Morphologique (Fermeture + Ouverture)")
    p3 = creer_panneau(visuel_boites, f"3. Boites candidates ({len(boites_candidates)} retenues)")
    p4 = creer_panneau(visuel_final, f"4. Resultat final ({len(resultats_finaux)} detecte(s))")

    ligne_haut = np.hstack([p1, p2])
    ligne_bas = np.hstack([p3, p4])
    return np.vstack([ligne_haut, ligne_bas])



