import numpy as np


def wilkinson_matrix(n):
    """
    Génère la matrice de Wilkinson de taille n x n.

    Paramètres :
    - n : taille de la matrice (entier).

    Retourne :
    - W : matrice de Wilkinson (numpy array).
    """
    # Créer une matrice tridiagonale avec des 1 sur la diagonale supérieure et inférieure
    W = np.diag([1] * (n-1), k=-1) + np.diag([1] * (n-1), k=1)

    # Remplir la diagonale principale
    for i in range(n):
        W[i, i] = (i + 1) - (n / 2) - \
            0.5 if n % 2 == 0 else (i + 1) - ((n + 1) / 2)

    return W


if __name__ == "__main__":
    # Exemple d'utilisation
    n = 5  # Taille de la matrice
    W = wilkinson_matrix(n)
    print("Matrice de Wilkinson de taille", n, "x", n, ":\n", W)
