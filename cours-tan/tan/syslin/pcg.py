import numpy as np

def pcg(A, b, P=None, x0=None, tol=1e-6, maxiter=100):
    """
    Calcule la solution du système linéaire Ax = b en utilisant la méthode du gradient conjugué préconditionné.
    
    Paramètres :
    ----------
    A : ndarray, shape (n, n)
        Matrice des coefficients.
    
    b : ndarray, shape (n,)
        Vecteur de droite.
    
    P : ndarray, shape (n, n), optionnel
        Matrice de préconditionnement. Par défaut, il s'agit de la matrice d'identité.
    
    x0 : ndarray, shape (n,), optionnel
        Estimation initiale de la solution. Par défaut, il s'agit d'un vecteur nul.
    
    maxiter : int, optionnel
        Nombre maximal d'itérations pour le solveur. La valeur par défaut est 100.
    
    tol : float, optionnel
        Tolérance pour le critère de convergence de la norme résiduelle. La valeur par défaut est 1e-6.
    
    Retourne :
    -------
    x : ndarray, shape (n,)
        Solution approximative de Ax = b.
    
    niter : int
        Nombre d'itérations effectuées.

    inc : liste
        L'incrément à chaque itération.

    res : list
        La norme résiduelle à chaque itération.

    Notes :
    -----
    La méthode du gradient conjugué préconditionné utilise la matrice P pour accélérer la convergence. La méthode itère
    tant que la norme résiduelle est supérieure à la tolérance spécifiée ou jusqu'à ce que le nombre maximal d'itérations soit atteint.
    """
    if P is None:
        P = np.eye(len(A))
    if x0 is None:
        x0 = np.zeros(len(A))
    niter = 0
    x = x0
    r = b - A @ x
    r0 = r
    z = np.linalg.solve(P, r)
    p = z
    res = [1]
    inc = []
    #while res[-1] > tol and niter < maxiter:
    while niter < maxiter:
        niter += 1
        pold = p
        zold = z
        rold = r
        xold = x
        alpha = np.dot(p, rold) / np.dot(p, A @ p)
        x = xold + alpha * p
        r = rold - alpha * A @ p
        z = np.linalg.solve(P, r)
        beta = np.dot(z, A @ pold ) / np.dot(pold, A @ pold)
        p =  z - beta * pold
        res.append(np.linalg.norm(r)/np.linalg.norm(r0))
        inc.append(np.linalg.norm(x - xold))
        if inc[-1] < tol:
            break
    return x, niter, inc, res