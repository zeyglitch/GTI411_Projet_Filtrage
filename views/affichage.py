import cv2
import numpy as np
import random


# dernière teinte utilisée : sert à ce que deux couleurs de suite soient bien différentes
_derniere_teinte = random.randint(0, 179)


def generer_couleur():
    """
    Couleur BGR aléatoire, vive et claire pour bien se voir sur l'image
    et sur le fond noir des étiquettes.
    La teinte est tirée au hasard mais toujours éloignée de la précédente
    (sinon on pourrait avoir 3 panneaux de suite en orange).
    """
    global _derniere_teinte
    _derniere_teinte = (_derniere_teinte + random.randint(30, 60)) % 180

    hsv = np.uint8([[[_derniere_teinte, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def taille_texte(largeur, hauteur):
    """
    Taille de la police selon la taille de la fenêtre :
    ni trop petit quand la fenêtre est petite, ni énorme en plein écran.
    """
    return max(0.45, min(0.8, min(largeur, hauteur) / 1400))


def _se_chevauchent(a, b):
    """True si les rectangles a et b = (x, y, w, h) se touchent."""
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def dessiner_etiquette(image, rect, texte, couleur, zones_occupees, autres_cadres=()):
    """
    Dessine le texte (fond noir) près du rectangle rect = (x, y, w, h).
    - l'étiquette reste toujours dans l'image (jamais coupée par le bord)
    - elle ne se pose ni sur une autre étiquette (zones_occupees), ni sur
      un autre cadre (autres_cadres) : on essaie des places de plus en plus
      loin au-dessus et en dessous du cadre
    - si elle est loin de son cadre, un petit trait les relie
    On ajoute la nouvelle étiquette à zones_occupees.
    """
    hauteur_img, largeur_img = image.shape[:2]
    x, y, w, h = rect

    echelle = taille_texte(largeur_img, hauteur_img)
    epaisseur = 1 if echelle < 0.7 else 2
    (tw, th), base = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)
    marge = 4
    lb = tw + 2 * marge         # largeur de l'étiquette
    hb = th + base + 2 * marge  # hauteur de l'étiquette

    # places essayées de la plus proche à la plus loin :
    # au-dessus (gauche puis droite) et en dessous (gauche puis droite)
    candidats = []
    for rang in range(5):
        candidats.append((x, y - hb * (rang + 1)))
        candidats.append((x + w - lb, y - hb * (rang + 1)))
        candidats.append((x, y + h + hb * rang))
        candidats.append((x + w - lb, y + h + hb * rang))
    # en dernier recours : dans le cadre
    candidats.append((x, y))
    candidats.append((x, y + h - hb))

    obstacles = list(zones_occupees) + list(autres_cadres)

    choix = None
    for cx, cy in candidats:
        # on la ramène dans l'image si elle dépasse
        cx = max(0, min(cx, largeur_img - lb))
        cy = max(0, min(cy, hauteur_img - hb))
        if not any(_se_chevauchent((cx, cy, lb, hb), o) for o in obstacles):
            choix = (cx, cy)
            break

    if choix is None:
        # vraiment aucune place : on la met au-dessus tant pis
        choix = (max(0, min(x, largeur_img - lb)), max(0, min(y - hb, hauteur_img - hb)))

    cx, cy = choix

    # trait entre l'étiquette et son cadre si elle n'est pas collée dessus
    px = max(x, min(cx + lb // 2, x + w))
    if cy + hb < y - 2:
        cv2.line(image, (px, cy + hb), (px, y), couleur, 1)
    elif cy > y + h + 2:
        cv2.line(image, (px, cy), (px, y + h), couleur, 1)

    cv2.rectangle(image, (cx, cy), (cx + lb, cy + hb), (0, 0, 0), -1)
    cv2.putText(image, texte, (cx + marge, cy + marge + th),
                cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, epaisseur)
    zones_occupees.append((cx, cy, lb, hb))


def dessiner_resultats(image, resultats):
    """
    Dessine tous les cadres puis toutes les étiquettes.
    resultats = liste de (rect, texte, couleur), rect = (x, y, w, h) en pixels de l'image.
    On dessine d'abord les cadres pour que les étiquettes les évitent.
    """
    echelle = taille_texte(image.shape[1], image.shape[0])
    epaisseur = 2 if echelle < 0.7 else 3

    for (x, y, w, h), texte, couleur in resultats:
        cv2.rectangle(image, (x, y), (x + w, y + h), couleur, epaisseur)

    zones_occupees = []
    for i, (rect, texte, couleur) in enumerate(resultats):
        autres = [r[0] for j, r in enumerate(resultats) if j != i]
        dessiner_etiquette(image, rect, texte, couleur, zones_occupees, autres)
    return image
