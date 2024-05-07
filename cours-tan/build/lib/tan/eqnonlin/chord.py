def chord(fun, a, b, x0, tol, nmax, *args):
    """
    CHORD Chord method.
    ZERO=CHORD(FUN,A,B,X0,TOL,NMAX) tries to find the zero ZERO of the 
    continuous function FUN in the interval [A,B] using the chord method 
    starting from X0, A < X0 < B. FUN is a function which accepts 
    real scalar input x and returns a real scalar value. 
    If the search fails an error message is displayed.
    
    [ZERO,RES,NITER]= CHORD(FUN,...) returns the value of the residual in ZERO
    and the iteration number at which ZERO was computed.
    """
    x = a
    fa = fun(x, *args)
    x = b
    fb = fun(x, *args)
    r = (fb - fa) / (b - a)
    err = tol + 1
    niter = 0
    x = x0
    fx = fun(x, *args)
    inc=[x0]
    while niter < nmax and err > tol:
        niter += 1
        xn = x - fx / r
        err = abs(xn - x)
        inc.append(err)
        x = xn
        fx = fun(x, *args)
        
    if niter >= nmax:
        print('Chord method stopped without converging to the desired tolerance',
              'because the maximum number of iterations was reached')
        
    zero = x
    res = fx
    
    return zero, res, niter,inc
