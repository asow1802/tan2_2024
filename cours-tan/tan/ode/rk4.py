import numpy as np

def rk4(odefun, tspan, y0, Nh, *args):
    """
    Solve differential equations using the RK4 method.
    
    Parameters:
    odefun : callable
        The function computing the derivative of y at t.
    tspan : tuple
        The time span (t0, tfinal) of integration.
    y0 : array_like
        Initial condition on y (can be a vector).
    Nh : int
        Number of equally spaced intervals.
    *args : tuple, optional
        Additional arguments to pass to the derivative function.
        
    Returns:
    t : ndarray
        A 1D array of time points where the solution was computed.
    y : ndarray
        A 2D array of values representing the solution.
    """
    h = (tspan[1] - tspan[0]) / Nh
    t = np.linspace(tspan[0], tspan[1], Nh+1)
    y = np.empty((Nh+1, len(y0)))
    y[0, :] = y0
    
    for i in range(Nh):
        k1 = h * odefun(t[i], y[i, :], *args)
        k2 = h * odefun(t[i] + 0.5*h, y[i, :] + 0.5*k1, *args)
        k3 = h * odefun(t[i] + 0.5*h, y[i, :] + 0.5*k2, *args)
        k4 = h * odefun(t[i] + h, y[i, :] + k3, *args)
        y[i+1, :] = y[i, :] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
    return t, y