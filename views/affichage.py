import cv2
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
