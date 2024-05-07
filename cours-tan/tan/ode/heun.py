import numpy as np

def heun(odefun, tspan, y0, Nh, *args):
    """
    Solve differential equations using the Heun method.
    
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
        # forward euler step
        y_end = y[i, :] + h * odefun(t[i], y[i, :], *args)
        # f(t_{n+1},y_n+1 )
        pred_y = odefun(t[i] + h, y_end, *args)
        y[i+1, :] = y[i, :] + 0.5 * h * (odefun(t[i], y[i, :], *args) + pred_y)
        
    return t, y

# Exemple d'utilisation avec une EDO simple
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Définir la fonction dérivée pour l'EDO y' = 1 - y^2
    def dydt(t, y): return 1 - y**2

    
    # Conditions initiales et paramètres de la simulation
    tspan = (0, 5)
    y0 = [0]
    Nh = 40
    
    # Résoudre l'EDO
    t, y = heun(dydt, tspan, y0, Nh)
    
    # Tracer la solution
    plt.plot(t, y[:, 0], '-o', label='Heun method')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()
