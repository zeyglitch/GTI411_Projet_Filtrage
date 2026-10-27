import cv2
import numpy as np
from models.similarite import trouver_meilleur_match
from models.detection_zones import (isoler_regions, nettoyer_masque, extraire_boites,
                                    fusionner_boites, est_nette, touche_le_bord, intersection)
from views.fenetre import FenetreImage
from views.affichage import creer_mosaique_etapes


# un panneau coupé par le bord de l'image est peu fiable : on est plus exigeant
SCORE_MIN_BORD = 0.5


class DetectionAuto:
    """
    Partie 2 du lab : détection automatique des panneaux.
    On isole les zones d'intérêt avec filtrage HSV + seuillage,
    puis contours, fusion des boites et matching avec les templates.
    """

    def __init__(self, chargeur):
        self.chargeur = chargeur

    def detecter(self, image, retour_etapes=False):
        """
        Fait toute la détection sur une image (sans rien afficher).
        Retourne une liste de (boite, nom_template, score).
        Si retour_etapes=True, retourne aussi les données des étapes intermédiaires.
        """
        # 1. un masque par couleur (rouge, orange, jaune, bleu, blanc)
        masques_couleurs = isoler_regions(image)

        # 2. nettoyage + contours -> boites englobantes pour chaque masque
        boites = []
        masques_nettoyes = []
        for masque in masques_couleurs:
            m_net = nettoyer_masque(masque)
            masques_nettoyes.append(m_net)
            boites += extraire_boites(m_net)

        # masque combiné nettoyé pour la visualisation
        masque_nettoye_combine = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in masques_nettoyes:
            masque_nettoye_combine = cv2.bitwise_or(masque_nettoye_combine, m)

        # 3. une même zone peut ressortir dans 2 masques -> on fusionne les boites qui se superposent
        boites = fusionner_boites(boites)
        boites_candidates = list(boites)

        # 4. on compare chaque zone aux templates avec l'algo de la partie 1
        resultats = []
        for boite in boites:
            x, y, w, h = boite
            roi = image[y:y+h, x:x+w]

            # zone floue -> pas un panneau
            if not est_nette(roi):
                continue

            nom, score = trouver_meilleur_match(roi, self.chargeur.templates)

            # sous le seuil -> ce n'est pas un panneau connu
            if nom is None:
                continue

            # panneau coupé par le bord de l'image : il faut un meilleur score
            if touche_le_bord(boite, image) and score < SCORE_MIN_BORD:
                continue

            resultats.append((boite, nom, score))

        # 5. deux résultats qui se superposent = le même panneau vu deux fois
        #    On garde celui qui a le meilleur score
        resultats = self._garder_meilleurs(resultats)

        # 6. si le même template sort plusieurs fois, on garde seulement les
        #    détections presque aussi bonnes que la meilleure (au moins 50%)
        meilleur_par_nom = {}
        for rect, nom, score in resultats:
            meilleur_par_nom[nom] = max(score, meilleur_par_nom.get(nom, 0))

        resultats_finaux = [r for r in resultats if r[2] >= 0.5 * meilleur_par_nom[r[1]]]

        if retour_etapes:
            etapes = (masques_couleurs, masque_nettoye_combine, boites_candidates)
            return resultats_finaux, etapes
        return resultats_finaux

    def _garder_meilleurs(self, resultats):
        """Parmi les résultats qui se superposent beaucoup, garde le meilleur score."""
        gardes = []
        for rect, nom, score in sorted(resultats, key=lambda r: r[2], reverse=True):
            doublon = False
            for rect2, _, _ in gardes:
                plus_petite = min(rect[2] * rect[3], rect2[2] * rect2[3])
                if intersection(rect, rect2) / plus_petite > 0.5:
                    doublon = True
                    break
            if not doublon:
                gardes.append((rect, nom, score))
        return gardes

    def lancer(self, mode_demo=False):
        """Lance la détection auto sur l'image courante et affiche le résultat."""
        if not self.chargeur.templates:
            print("Pas de templates chargés")
            return

        if self.chargeur.image_courante is None:
            print("Aucune image chargée")
            return

        print("\n=== PARTIE 2 : Détection automatique ===")
        print("Analyse en cours...\n")

        image = self.chargeur.image_courante
        resultats, etapes = self.detecter(image, retour_etapes=True)

        for rect, nom, score in resultats:
            print(f"  -> {nom} (score: {score:.2f}) en {rect}")
        print(f"\n{len(resultats)} panneau(x) trouvé(s).")
        print("Astuce : Espace ou 'E' pour basculer entre les étapes et le résultat final")
        print("         F : plein écran | Echap : quitter\n")

        # Génération de la mosaïque des étapes
        masques_couleurs, masque_nettoye, boites_candidates = etapes
        mosaique = creer_mosaique_etapes(image, masques_couleurs, masque_nettoye,
                                         boites_candidates, resultats)

        # Préparation des annotations du résultat final
        annotations_finales = [(rect, f"{nom} ({score:.2f})", None) for rect, nom, score in resultats]

        titre = "Partie 2 - Detection automatique (Espace/E: Etapes | F: Plein ecran | Echap: Quitter)"
        vue_etapes = mode_demo
        img_depart = mosaique if vue_etapes else image

        fenetre = FenetreImage(titre, img_depart)
        if not vue_etapes:
            for rect, texte, coul in annotations_finales:
                fenetre.ajouter_annotation(rect, texte, coul)
        fenetre.ouvrir()

        while True:
            fenetre.afficher()
            touche = fenetre.lire_touche()

            if touche == 27 or not fenetre.est_ouverte():  # echap ou bouton X
                break

            # Espace ou touche E -> basculer entre mosaïque des étapes et résultat final
            if touche in (ord('e'), ord('E'), 32):
                vue_etapes = not vue_etapes
                if vue_etapes:
                    fenetre.changer_image(mosaique, [])
                else:
                    fenetre.changer_image(image, [])
                    for rect, texte, coul in annotations_finales:
                        fenetre.ajouter_annotation(rect, texte, coul)

        fenetre.fermer()
