import numpy as np
import scipy.linalg as la


def qrbasic(A, tol=1e-6, nmax=100):
    """
    Calcule les valeurs propres de la matrice A en utilisant des itérations QR.
    
    Paramètres :
    - A : la matrice carrée dont on cherche les valeurs propres.
    - tol : la tolérance pour le critère d'arrêt.
    - nmax : le nombre maximal d'itérations.
    
    Retourne :
    - D : un tableau des valeurs propres de A.
    - niter : le nombre d'itérations effectuées.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Seules les matrices carrées sont supportées.")

    T = A.copy()
    niter = 0
    test = np.linalg.norm(np.tril(T, -1), np.inf)

    while niter <= nmax and test >= tol:
        Q, R = np.linalg.qr(T)
        T = R @ Q
        niter += 1
        test = np.linalg.norm(np.tril(T, -1), np.inf)

    if niter > nmax:
        print("La méthode ne converge pas dans le nombre d'itérations maximum voulu.")

    D = np.diag(T)
    return D, niter
