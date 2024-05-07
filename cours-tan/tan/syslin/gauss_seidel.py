import numpy as np
import scipy.linalg # <0>

def gauss_seidel1(A, b, x0, tol=1e-6, maxiter=100):
    """Solve the linear system Ax = b using the Gauss-Seidel method.

    Parameters
    ----------
    A : array_like
        The matrix of the linear system.
    b : array_like
        The right-hand side vector of the linear system.
    x0 : array_like
        The initial guess for the solution.
    tol : float, optional
        The tolerance for the stopping criterion.
    maxiter : int, optional
        The maximum number of iterations.

    Returns
    -------
    x : array_like
        The solution of the linear system.
    niter : int
        The number of iterations performed.
    inc : array_like
        The increment at each iteration.
    """
    x = x0.copy()
    niter = 0
    inc = []
    while True:
        niter += 1
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter,inc

def gauss_seidel2(A, b, x0, tol=1e-6, maxiter=100):
    """Solve the linear system Ax = b using the Gauss-Seidel method.

    Parameters
    ----------
    A : array_like
        The matrix of the linear system.
    b : array_like
        The right-hand side vector of the linear system.
    x0 : array_like
        The initial guess for the solution.
    tol : float, optional
        The tolerance for the stopping criterion.
    maxiter : int, optional
        The maximum number of iterations.

    Returns
    -------
    x : array_like
        The solution of the linear system.
    niter : int
        The number of iterations performed.
    inc : array_like
        The increment at each iteration.
    """
    x = x0.copy()
    niter = 0
    inc = []
    D=np.diag(np.diag(A))
    E=-np.tril(A,-1)
    F=-np.triu(A,1)
    invB=np.linalg.inv(D-E)
    B=scipy.linalg.solve_triangular(D-E,F,lower=True)
    g=scipy.linalg.solve_triangular(D-E,b,lower=True)
    x_new = np.zeros_like(x)
    while True:
        niter += 1
        x_new=np.dot(B,x)+g
        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter,inc