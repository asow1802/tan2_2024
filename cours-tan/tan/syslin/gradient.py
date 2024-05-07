import numpy as np

def gradient(A, b, P=None, x0=None, tol=1e-6, maxiter=100):
    """
    Calcule la solution du système linéaire Ax = b en utilisant la méthode de descente de gradient préconditionnée.
    
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
    La méthode de descente de gradient préconditionnée utilise la matrice P pour accélérer la convergence. La méthode itère 
    tant que la norme résiduelle est supérieure à la tolérance spécifiée ou jusqu'à ce que le nombre maximal d'itérations soit atteint.
    """
    if P is None:
        P = np.eye(len(A))
    if x0 is None:
        x0 = np.zeros(len(A))
    niter = 0
    x = x0.copy()
    r = b - A @ x
    r0 = r
    z = np.linalg.solve(P, r)
    res = [1]
    inc = []
    # nous utilisons la norme du résidu relatif  pour le critère d'arrêt
    # nous calculons également l'incrément
    while niter < maxiter:
        alpha = np.dot(r, z) / np.dot(A @ z, z)
        x_new = x + alpha * z
        r = r - alpha * A @ z
        z = np.linalg.solve(P, r)
        res.append(np.linalg.norm(r)/np.linalg.norm(r0))
        inc.append(np.linalg.norm(x_new - x))        
        if res[-1] < tol:
            break
        if inc[-1] < tol:
            break
        x = x_new
        niter = niter + 1
    return x, niter, inc, res

def richardson( A, b, x0=None, P=None, alpha=1.0, tol=1e-6, maxiter=100):
    """Résolvez le système linéaire Ax = b en utilisant la méthode de Richardson.
    
    Paramètres
    ----------
    A : array_like
        La matrice du système linéaire.
    b : array_like
        Le vecteur de droite du système linéaire.
    x0 : semblable à un tableau, optionnel
        L'estimation initiale de la solution. Par défaut, il s'agit d'un vecteur nul.
    P  : array_like, optionnel
        La matrice de préconditionnement. Par défaut, il s'agit de la matrice identité.
    alpha : float, optionnel
        Le paramètre de relaxation. La valeur par défaut est 1,0.
    tol : float, optionnel
        La tolérance pour le critère d'arrêt. La valeur par défaut est 1e-6.
    maxiter : int, optionnel
        Le nombre maximum d'itérations. La valeur par défaut est 100.
    
    Retourne
    -------
    x : array_like
        La solution du système linéaire.
    niter : int
        Le nombre d'itérations effectuées.
    inc : array_like
        L'incrément à chaque itération.
    """
    if x0 is None:
        x0 = np.zeros(len(A))
    if P is None:
        P = np.eye(len(A))
    x = x0.copy()
    r = b - A @ x
    r0 = r
    niter = 0   
    res = [1]
    inc = []
    while True:
        niter += 1
        d = np.linalg.solve(P, b - A @ x)
        x_new = x + alpha * d
        res.append(np.linalg.norm(b - A @ x_new)/np.linalg.norm(r0))
        inc.append(np.linalg.norm(x_new - x))
        if res[-1] < tol:
            break
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter, inc
