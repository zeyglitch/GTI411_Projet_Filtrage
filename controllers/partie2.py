import cv2
from models.similarite import trouver_meilleur_match
from models.detection_zones import (isoler_regions, nettoyer_masque, extraire_boites,
                                    fusionner_boites, est_nette, touche_le_bord)
from views.affichage import dessiner_resultats_multiples
from controllers.partie1 import redimensionner_si_necessaire


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

        # 5. si le même template sort plusieurs fois, on garde seulement les
        #    détections presque aussi bonnes que la meilleure (au moins 50%).
        #    Comme ça deux vrais panneaux identiques restent, mais les
        #    fragments d'un autre panneau qui ressemblent un peu sont enlevés.
        meilleur_par_nom = {}
        for rect, nom, score in resultats:
            meilleur_par_nom[nom] = max(score, meilleur_par_nom.get(nom, 0))

        return [r for r in resultats if r[2] >= 0.5 * meilleur_par_nom[r[1]]]

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

        # on dessine sur une copie, puis on resize pour que ça rentre à l'écran
        image_resultat = dessiner_resultats_multiples(image.copy(), resultats)
        image_resultat, _ = redimensionner_si_necessaire(image_resultat)

        nom_fenetre = "Partie 2 - Detection automatique (Echap pour quitter)"
        cv2.namedWindow(nom_fenetre, cv2.WINDOW_AUTOSIZE)

        while True:
            cv2.imshow(nom_fenetre, image_resultat)
            key = cv2.waitKey(30) & 0xFF

            # fenêtre fermée avec le bouton X
            try:
                if cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == 27:  # echap
                break

        cv2.destroyAllWindows()
