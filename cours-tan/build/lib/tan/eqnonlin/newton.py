def newton(f, df, x0, tol, nmax, *args):
    """
    NEWTON Find function zeros.
    ZERO=NEWTON(FUN,DFUN,X0,TOL,NMAX) tries to find the zero ZERO of the 
    continuous and differentiable function FUN nearest to X0 using the Newton 
    method. FUN and its derivative DFUN accept real scalar input x and returns 
    a real scalar value. If the search fails an error message is displayed.
    FUN and DFUN can also be inline objects.
    
    [ZERO,RES,NITER]= NEWTON(FUN,...) returns the value of the residual in ZERO
    and the iteration number at which ZERO was computed.
    
    [ZERO,RES,NITER,INC]= NEWTON(FUN,...) returns a list INC with the absolute values of the
    differences between successive approximations (increments).
    """
    x = x0
    fx = f(x, *args)
    dfx = df(x, *args)
    niter = 0
    inc = [x0]
    diff = tol + 1
    
    while diff >= tol and niter < nmax:
        niter += 1
        diff = - fx / dfx
        x = x + diff
        diff = abs(diff)
        fx = f(x, *args)
        dfx = df(x, *args)
        inc.append(diff)
        
    if niter >= nmax:
        print('Newton method stopped without converging to the desired tolerance because the maximum number of iterations was reached')
    
    zero = x
    res = fx
    
    return zero, res, niter, inc


