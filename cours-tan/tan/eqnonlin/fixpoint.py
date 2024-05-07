def fixpoint(phi,s, x0, tol, nmax, *args):
    x = x0
    phix = phi(x,s, *args)
    niter = 0
    sol = []
    diff = tol + 1
    inc = [x0]
    
    while diff >= tol and niter < nmax:
        niter += 1
        diff = abs(phix - x)
        x = phix
        phix = phi(x,s, *args)
        inc.append(diff) 
    p = x
    res = phix - x
    
    return p, niter, inc 

