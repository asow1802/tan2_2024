import numpy as np

def matrix(n, epsi):
    """
    Generate a matrix A and vector b based on given parameters.
    
    Parameters:
    -----------
    n : int
        The size of the square matrix A.
    epsi : float
        A parameter that influences off-diagonal values in matrix A.
        
    Returns:
    --------
    A : ndarray
        A n x n matrix defined as:
        A = diag(ones(n)) + 
            epsi * (diag(ones(n-1), -1) + diag(ones(n-1), 1)) + 
            epsi**2 * (diag(ones(n-2), -2) + diag(ones(n-2), 2))
    b : ndarray
        A vector defined as:
        b = A @ ones(n)
    
    Example:
    --------
    >>> A, b = matrix(5, 0.1)
    >>> print(A)
    ...
    >>> print(b)
    ...
    """
    
    A = np.diag(np.ones(n)) + \
        epsi * (np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1)) + \
        epsi**2 * (np.diag(np.ones(n-2), -2) + np.diag(np.ones(n-2), 2))
    
    b = A @ np.ones(n)
    
    return A, b
