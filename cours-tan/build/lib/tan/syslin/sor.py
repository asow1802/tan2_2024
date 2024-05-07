import numpy as np
import scipy.linalg # <0>

def sor1( A, b, x0, omega=1.0, tol=1e-6, maxiter=100):
    """Solve the linear system Ax = b using the SOR method.

    Parameters
    ----------
    A : array_like
        The matrix of the linear system.
    b : array_like
        The right-hand side vector of the linear system.
    x0 : array_like
        The initial guess for the solution.
    omega : float, optional
        The relaxation parameter.        
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
            x_new[i] = (1-omega)*x[i] + omega*(b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i] # <1>

        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter,inc

def sor2( A, b, x0, omega=1.0, tol=1e-6, maxiter=100):
    """Solve the linear system Ax = b using the SOR method.

    Parameters
    ----------
    A : array_like
        The matrix of the linear system.
    b : array_like
        The right-hand side vector of the linear system.
    x0 : array_like
        The initial guess for the solution.
    omega : float, optional
        The relaxation parameter.        
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
    invD=np.diag(1/np.diag(A))
    E=-np.tril(A,-1)
    F=-np.triu(A,1)
    B=scipy.linalg.solve_triangular(np.eye(len(A))-omega*invD@E,(1-omega)*np.eye(len(A))+omega*invD@F, lower=True) # <2>
    g=omega*invD@b
    while True:
        niter += 1
        x_new = np.zeros_like(x)
        x_new=np.dot(B,x)+g # <4>
        inc.append(np.linalg.norm(x_new - x))
        if inc[-1] < tol:
            break
        if niter == maxiter:
            break
        x = x_new
    return x, niter,inc    