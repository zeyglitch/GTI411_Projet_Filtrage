import cv2
import random


def generer_couleur():
    """Couleur BGR aléatoire pas trop sombre."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def dessiner_resultat(image, rect, nom_match, score, couleur=None):
    """
    Dessine un rectangle + texte sur l'image.
    rect = (x, y, w, h)
    """
    x, y, w, h = rect

    if couleur is None:
        couleur = generer_couleur()

    # rectangle autour de la zone
    cv2.rectangle(image, (x, y), (x + w, y + h), couleur, 2)

    # texte avec fond noir pour la lisibilité
    texte = f"{nom_match} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - th - 10), (x + tw + 4, y), (0, 0, 0), -1)
    cv2.putText(image, texte, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

    return image


def dessiner_resultats_multiples(image, resultats):
    """
    Dessine plusieurs résultats sur l'image.
    resultats = liste de (rect, nom_match, score)
    """
    for rect, nom, score in resultats:
        couleur = generer_couleur()
        dessiner_resultat(image, rect, nom, score, couleur)
    return image
