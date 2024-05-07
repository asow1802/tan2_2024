import numpy as np

def eigpower(A, tol=1e-6, nmax=100, x0=None):
    """
    Evalue numériquement une valeur propre d’une matrice par la méthode de la puissance.

    Paramètres :
    - A : Matrice carrée dont on cherche la valeur propre de module maximal.
    - tol : Tolerance pour l'erreur absolue (1.e-6 par défaut).
    - nmax : Nombre maximal d'itérations (100 par défaut).
    - x0 : Vecteur initial (vecteur de 1 par défaut).

    Retourne :
    - lambda : Valeur propre de module maximal de la matrice A.
    - x : Vecteur propre associé à la valeur propre lambda.
    - iter : Nombre d'itérations effectuées.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrice doit être carrée")

    if x0 is None:
        x0 = np.ones(n)

    x0 = x0 / np.linalg.norm(x0)
    pro = A @ x0
    lambda_ = x0.T @ pro
    err = tol * abs(lambda_) + 1
    iter_ = 0
    while err > tol * abs(lambda_) and abs(lambda_) != 0 and iter_ <= nmax:
        x = pro
        x = x / np.linalg.norm(x)
        pro = A @ x
        lambda_new = x.T @ pro
        err = abs(lambda_new - lambda_)
        lambda_ = lambda_new
        iter_ += 1

    return lambda_, x, iter_

# Exemple d'utilisation
if __name__ == "__main__":
    A = np.array([[2, 1], [1, 3]])
    lambda_, x, iter_ = eigpower(A)
    print(f"Valeur propre de module maximal : {lambda_}")
    print(f"Vecteur propre associé : {x}")
    print(f"Nombre d'itérations : {iter_}")
    # Sortie attendue avec tol=1.e-6 et nmax=100 et scipy
    
    
