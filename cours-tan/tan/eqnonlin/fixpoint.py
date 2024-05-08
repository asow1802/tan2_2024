def fixpoint(phi, x0, tol, nmax, *args):
    """
    FIXPOINT Fixed point iterations.
    P=FIXPOINT(PHI,X0,TOL,NMAX) tries to find the fixed point P of the 
    continuous and differentiable function PHI using fixed point iterations 
    starting from X0. PHI is a function which accepts real scalar input x 
    and returns a real scalar value. If the search fails, an error message is displayed.
    
    [P,RES,NITER]= FIXPOINT(PHI,...) returns the value of the residual in ZERO
    and the iteration number at which ZERO was computed.
    
    [P,RES,NITER,INC]= FIXPOINT(PHI,...) returns a list INC with the absolute values of the
    differences between successive approximations (increments).
    """
    x = x0
    phix = phi(x, *args)
    niter = 0
    diff = tol + 1
    inc = [x0]
    
    while diff >= tol and niter < nmax:
        niter += 1
        diff = abs(phix - x)
        x = phix
        phix = phi(x, *args)
        inc.append(diff)
        
    if niter >= nmax:
        print('fixpoint stopped without converging to the desired tolerance',
              'because the maximum number of iterations was reached')
        
    p = x
    res = phix - x
    
    return p, res, niter, inc
