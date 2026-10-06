import cv2
from models.similarite import trouver_meilleur_match
from models.detection_zones import (isoler_regions, nettoyer_masque, extraire_boites,
                                    fusionner_boites, est_nette, touche_le_bord, intersection)
from views.fenetre import FenetreImage


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

    def detecter(self, image):
        """
        Fait toute la détection sur une image (sans rien afficher).
        Retourne une liste de (boite, nom_template, score).
        """
        # 1. un masque par couleur (rouge, orange, jaune, bleu, blanc)
        # 2. nettoyage + contours -> boites englobantes pour chaque masque
        boites = []
        for masque in isoler_regions(image):
            masque = nettoyer_masque(masque)
            boites += extraire_boites(masque)

        # 3. une même zone peut ressortir dans 2 masques (ex: bordure rouge
        #    et intérieur blanc) -> on fusionne les boites qui se superposent
        boites = fusionner_boites(boites)

        # 4. on compare chaque zone aux templates avec l'algo de la partie 1
        resultats = []
        for boite in boites:
            x, y, w, h = boite
            roi = image[y:y+h, x:x+w]

            # zone floue (voiture, fond hors focus) -> pas un panneau
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
        #    (ex: un gros morceau de fond et le vrai panneau dedans).
        #    On garde celui qui a le meilleur score.
        resultats = self._garder_meilleurs(resultats)

        # 6. si le même template sort plusieurs fois, on garde seulement les
        #    détections presque aussi bonnes que la meilleure (au moins 50%).
        #    Comme ça deux vrais panneaux identiques restent, mais les
        #    fragments d'un autre panneau qui ressemblent un peu sont enlevés.
        meilleur_par_nom = {}
        for rect, nom, score in resultats:
            meilleur_par_nom[nom] = max(score, meilleur_par_nom.get(nom, 0))

        return [r for r in resultats if r[2] >= 0.5 * meilleur_par_nom[r[1]]]

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

    def lancer(self):
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
        resultats = self.detecter(image)

        for rect, nom, score in resultats:
            print(f"  -> {nom} (score: {score:.2f}) en {rect}")
        print(f"\n{len(resultats)} panneau(x) trouvé(s). Echap ou bouton X pour quitter")

        titre = "Partie 2 - Detection automatique (Echap: quitter | F: plein ecran)"
        fenetre = FenetreImage(titre, image)
        for rect, nom, score in resultats:
            fenetre.ajouter_annotation(rect, f"{nom} ({score:.2f})")
        fenetre.ouvrir()

        while True:
            fenetre.afficher()
            touche = fenetre.lire_touche()

            if touche == 27 or not fenetre.est_ouverte():  # echap ou bouton X
                break

        fenetre.fermer()
