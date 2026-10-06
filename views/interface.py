import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os


class InterfacePrincipale:
    """
    Fenêtre principale en tkinter.
    Permet de choisir l'image et la partie à lancer,
    au lieu de passer par le terminal.
    """

    def __init__(self, images_test, callback_partie1, callback_partie2):
        self.callback_partie1 = callback_partie1
        self.callback_partie2 = callback_partie2
        self.images_test = list(images_test)
        self.chemin_externe = None

        # fenêtre principale
        self.root = tk.Tk()
        self.root.title("GTI411 - Lab 2 : Filtrage")
        self.root.geometry("420x280")
        self.root.resizable(False, False)

        self._creer_widgets()

    def _creer_widgets(self):
        """Met en place tous les éléments de l'interface."""
        # titre
        titre = ttk.Label(self.root, text="GTI411 - Lab 2 : Filtrage",
                          font=("Arial", 14, "bold"))
        titre.pack(pady=(15, 10))

        # section sélection d'image
        frame_img = ttk.LabelFrame(self.root, text="Image de test", padding=10)
        frame_img.pack(padx=20, fill="x")

        # combobox avec les images du dossier de test
        self.combo = ttk.Combobox(frame_img, values=self.images_test,
                                  state="readonly", width=30)
        if self.images_test:
            self.combo.current(0)
        self.combo.pack(side="left", padx=(0, 10))

        # bouton pour charger une image custom
        btn_parcourir = ttk.Button(frame_img, text="Parcourir...",
                                   command=self._parcourir)
        btn_parcourir.pack(side="right")

        # boutons des deux parties
        frame_parties = ttk.Frame(self.root)
        frame_parties.pack(pady=20)

        btn_p1 = ttk.Button(frame_parties, text="Partie 1 - Manuelle",
                            command=self._lancer_p1, width=22)
        btn_p1.grid(row=0, column=0, padx=10)

        btn_p2 = ttk.Button(frame_parties, text="Partie 2 - Automatique",
                            command=self._lancer_p2, width=22)
        btn_p2.grid(row=0, column=1, padx=10)

        # bouton quitter
        btn_quit = ttk.Button(self.root, text="Quitter", command=self.root.quit)
        btn_quit.pack(pady=(5, 15))

    def _parcourir(self):
        """Ouvre un file dialog pour choisir une image."""
        chemin = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("Tous", "*.*")]
        )
        if chemin:
            self.chemin_externe = chemin
            nom = os.path.basename(chemin)
            # on ajoute le nom au combo si pas déjà dedans
            if nom not in self.images_test:
                self.images_test.append(nom)
                self.combo['values'] = self.images_test
            self.combo.set(nom)

    def _get_selection(self):
        """Retourne le nom de l'image ou le chemin externe."""
        selection = self.combo.get()
        if not selection:
            messagebox.showwarning("Attention", "Sélectionnez d'abord une image!")
            return None
        # si c'est l'image externe, on retourne le chemin complet
        if self.chemin_externe and selection == os.path.basename(self.chemin_externe):
            return self.chemin_externe
        return selection

    def _lancer_p1(self):
        """Lance la partie 1."""
        selection = self._get_selection()
        if selection:
            self.root.withdraw()  # on cache la fenêtre tkinter
            self.callback_partie1(selection)
            self.root.deiconify()  # on la reaffiche après

    def _lancer_p2(self):
        """Lance la partie 2."""
        selection = self._get_selection()
        if selection:
            self.root.withdraw()
            self.callback_partie2(selection)
            self.root.deiconify()

    def lancer(self):
        """Lance la boucle principale de l'interface."""
        self.root.mainloop()
