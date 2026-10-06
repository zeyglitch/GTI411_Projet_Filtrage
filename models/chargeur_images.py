import cv2
import os


class ChargeurImages:
    """Gère le chargement des templates et des images de test."""

    def __init__(self, dossier_templates, dossier_tests):
        self.dossier_templates = dossier_templates
        self.dossier_tests = dossier_tests
        self.templates = {}
        self.image_courante = None

        self._charger_templates()

    def _charger_templates(self):
        """Charge tous les templates depuis le dossier."""
        if not os.path.exists(self.dossier_templates):
            print(f"Dossier templates introuvable : {self.dossier_templates}")
            return

        for fichier in os.listdir(self.dossier_templates):
            if fichier.lower().endswith((".png", ".jpg", ".jpeg")):
                nom = os.path.splitext(fichier)[0]
                chemin = os.path.join(self.dossier_templates, fichier)
                img = cv2.imread(chemin)
                if img is not None:
                    self.templates[nom] = img

        print(f"{len(self.templates)} templates chargés")

    def lister_images_test(self):
        """Liste les images dispo dans le dossier de test."""
        if not os.path.exists(self.dossier_tests):
            return []
        return sorted([f for f in os.listdir(self.dossier_tests)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    def charger_image(self, nom_fichier):
        """Charge une image de test par son nom de fichier."""
        chemin = os.path.join(self.dossier_tests, nom_fichier)
        self.image_courante = cv2.imread(chemin)
        if self.image_courante is None:
            print(f"Impossible de charger : {chemin}")
            return False
        return True

    def charger_image_chemin(self, chemin_complet):
        """Charge une image depuis un chemin absolu (pour les images externes)."""
        self.image_courante = cv2.imread(chemin_complet)
        if self.image_courante is None:
            print(f"Impossible de charger : {chemin_complet}")
            return False
        return True
