import cv2
import numpy as np


# --- paramètres (tout est en proportion de l'image, rien n'est fixé en pixels) ---
AIRE_MIN = 0.003      # une boite doit couvrir au moins 0.3% de l'image
AIRE_MAX = 0.30       # et au plus 30%
RATIO_MIN = 0.4       # largeur / hauteur
RATIO_MAX = 2.5
REMPLISSAGE_MIN = 0.35  # aire du contour / aire de la boite (vire les taches fines)


def isoler_regions(image):
    """
    Crée un masque binaire par couleur de panneau.
    Couleurs (rouge, orange, jaune, bleu) avec la teinte HSV,
    et blanc avec un seuillage (faible saturation + forte luminosité).
    On garde les masques séparés : si on les mélange, deux panneaux
    de couleurs différentes qui se touchent deviennent un seul gros blob.
    """
    flou = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(flou, cv2.COLOR_BGR2HSV)

    # rouge = deux morceaux car la teinte "fait le tour" à 180
    rouge = cv2.inRange(hsv, (0, 110, 70), (10, 255, 255)) | \
            cv2.inRange(hsv, (165, 110, 70), (180, 255, 255))

    # orange / jaune : saturation élevée sinon on prend le sable du fond
    orange = cv2.inRange(hsv, (10, 170, 120), (20, 255, 255))
    jaune = cv2.inRange(hsv, (20, 150, 120), (35, 255, 255))

    bleu = cv2.inRange(hsv, (100, 130, 70), (130, 255, 255))

    # blanc : peu de couleur et très clair
    blanc = cv2.inRange(hsv, (0, 0, 215), (180, 48, 255))

    return [rouge, orange, jaune, bleu, blanc]


def nettoyer_masque(masque):
    """
    Fermeture pour relier les morceaux d'un même panneau (texte, bordure...)
    puis ouverture pour enlever les petits points isolés.
    La taille du noyau dépend de la taille de l'image.
    """
    petit_cote = min(masque.shape[:2])
    taille = max(3, int(petit_cote * 0.008))
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taille, taille))

    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)
    return masque


def extraire_boites(masque):
    """
    findContours sur le masque, puis on garde seulement les contours
    qui peuvent être un panneau (taille, ratio, remplissage).
    Retourne une liste de boites (x, y, w, h).
    """
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aire_image = masque.shape[0] * masque.shape[1]

    boites = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aire_boite = w * h

        if aire_boite < AIRE_MIN * aire_image or aire_boite > AIRE_MAX * aire_image:
            continue

        ratio = w / h
        if ratio < RATIO_MIN or ratio > RATIO_MAX:
            continue

        if cv2.contourArea(c) / aire_boite < REMPLISSAGE_MIN:
            continue

        boites.append((x, y, w, h))

    return boites


def _intersection(a, b):
    """Aire de l'intersection entre deux boites (x, y, w, h)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def fusionner_boites(boites, seuil=0.5):
    """
    Enlève les doublons : si deux boites se chevauchent beaucoup
    (intersection / aire de la plus petite > seuil), on les fusionne
    en une seule boite qui contient les deux.
    On répète tant qu'il reste des fusions possibles.
    """
    boites = list(boites)
    changement = True

    while changement:
        changement = False
        for i in range(len(boites)):
            for j in range(i + 1, len(boites)):
                a, b = boites[i], boites[j]
                plus_petite = min(a[2] * a[3], b[2] * b[3])

                if _intersection(a, b) / plus_petite > seuil:
                    x = min(a[0], b[0])
                    y = min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    boites[i] = (x, y, x2 - x, y2 - y)
                    boites.pop(j)
                    changement = True
                    break
            if changement:
                break

    return boites
