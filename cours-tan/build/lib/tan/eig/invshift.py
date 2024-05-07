import numpy as np
from scipy.linalg import lu, solve_triangular


def invshift(A, mu=0, tol=1e-6, nmax=10000, x0=None):
    """
    Evalue numériquement une valeur propre d'une matrice par la méthode de la puissance inverse.
    
    Paramètres :
    - A : Matrice carrée dont on cherche la valeur propre de module minimal ou la plus proche de MU.
    - mu : Réel ou complexe autour duquel on cherche la valeur propre (0 par défaut).
    - tol : Tolérance pour l'erreur absolue (1.e-6 par défaut).
    - nmax : Nombre maximal d'itérations (100 par défaut).
    - x0 : Vecteur initial (vecteur aléatoire par défaut).
    
    Retourne :
    - lambda : Valeur propre de la matrice A la plus proche de MU.
    - x : Vecteur propre associé à la valeur propre lambda.
    - iter : Numéro de l'itération à laquelle la valeur propre est calculée.
    """
    n, m = A.shape
    if n != m:
        raise ValueError('Seules les matrices carrées sont supportées')
    if x0 is None:
        x0 = np.random.rand(n)
    x0 = x0/ np.linalg.norm(x0)
    P,L, U = lu(A - mu * np.eye(n))
    np.isclose(P @ L @ U, A - mu * np.eye(n))
    z0 = solve_triangular(L, P.T @ x0, lower=True)
    pro = solve_triangular(U, z0)
    lamb = np.dot(x0, pro)
    err = tol * abs(lamb) + 1
    iter = 0
    while (err > tol * abs(lamb)) and (abs(lamb) != 0) and (iter <= nmax):
        x = pro/np.linalg.norm(pro)
        z = solve_triangular(L, P.T @ x, lower=True)
        pro = solve_triangular(U, z)
        lamb_new = np.dot(x,pro)
        err = abs(lamb_new - lamb)
        lamb = lamb_new

        iter += 1
    lamb = 1 / lamb + mu
    return lamb, x, iter
