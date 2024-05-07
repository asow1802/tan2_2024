def bisection(fun, a, b, tol, nmax, *args):
    """
    BISECTION Find function zeros.
    ZERO=BISECTION(FUN,A,B,TOL,NMAX) tries to find a zero ZERO of the continuous 
    function FUN in the interval [A,B] using the bisection method. FUN accepts 
    real scalar input x and returns a real scalar value. If the search fails 
    an error message is displayed. 
    ZERO=BISECTION(FUN,A,B,TOL,NMAX,P1,P2,...) passes parameters P1,P2,...
    to function: FUN(X,P1,P2,...).
    [ZERO,RES,NITER]= BISECTION(FUN,...) returns the value of the residual in ZERO
    and the iteration number at which ZERO was computed.
    [ZERO,RES,NITER,INC]= BISECTION(FUN,...) returns a vector INC with the absolute value of the
    differences between successive approximations (increments).
    """
    x = [a, (a+b)/2.0, b]
    inc = [x[1]]
    fx = [fun(x[0], *args), fun(x[1], *args), fun(x[2], *args)]

    if fx[0] * fx[2] > 0:
        raise ValueError(
            'The sign of FUN at the extrema of the interval must be different')
    elif fx[0] == 0:
        return a, 0, 0, inc
    elif fx[2] == 0:
        return b, 0, 0, inc

    niter = 0
    I = (b - a) / 2.0
    while I >= tol and niter < nmax:
        niter += 1
        if fx[0] * fx[1] < 0:
            x[2] = x[1]
            x[1] = x[0] + (x[2] - x[0]) / 2.0
            inc.append(abs(x[2] - x[1]))
        elif fx[1] * fx[2] < 0:
            x[0] = x[1]
            x[1] = x[0] + (x[2] - x[0]) / 2.0
            inc.append(abs(x[0] - x[1]))
        else:
            x[1] = x[fx.index(0)]
            I = 0
            inc.append(0)

        fx = [fun(x[0], *args), fun(x[1], *args), fun(x[2], *args)]
        I = (x[2] - x[0]) / 2.0
        

    inc.pop()
    if niter >= nmax:
        print('Bisection stopped without converging to the desired tolerance because the maximum number of iterations was reached')

    zero = x[1]
    res = fun(x[1], *args)
    return zero, res, niter, inc


