import numpy as np

def jacobi1(A, b, x0, tol=1e-6, maxiter=100):
    """Résolvez le système linéaire Ax = b en utilisant la méthode de Jacobi.

    Paramètres
    ----------
    A : array_like
        La matrice du système linéaire.
    b : array_like
        Le vecteur de droite du système linéaire.
    x0 : semblable à un tableau
        L'estimation initiale de la solution.
    tol : float, optionnel
        La tolérance pour le critère d'arrêt.
    maxiter : int, optionnel
        Le nombre maximum d'itérations.

    Retourne
    -------
    x : array_like
        La solution du système linéaire.
    niter : int
        Le nombre d'itérations effectuées.
    inc : array_like
        L'incrément à chaque itération.
    """
    x = x0.copy()
    niter = 0
    inc = []
    while True:
        niter += 1
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i] # <1>
        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter,inc

def jacobi2( A, b, x0, tol=1e-6, maxiter=100):
    """Résolvez le système linéaire Ax = b en utilisant la méthode de Jacobi.

    Paramètres
    ----------
    A : array_like
        La matrice du système linéaire.
    b : array_like
        Le vecteur de droite du système linéaire.
    x0 : semblable à un tableau
        L'estimation initiale de la solution.
    tol : float, optionnel
        La tolérance pour le critère d'arrêt.
    maxiter : int, optionnel
        Le nombre maximum d'itérations.

    Retourne
    --------
    x : array_like
        La solution du système linéaire.
    niter : int
        Le nombre d'itérations effectuées.
    inc : array_like
        L'incrément à chaque itération.
    """
    x = x0.copy()
    niter = 0
    D=np.diag(np.diag(A))
    N = D-A
    B = np.linalg.inv(D).dot(N) # <2>
    g = np.linalg.inv(D).dot(b) 
    inc = []
    while True:
        niter += 1
        x_new = B.dot(x) + g # <3>
        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter, inc