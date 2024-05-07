import numpy as np

def feuler(odefun, tspan, y0, Nh, *args):
    """
    Résout une équation différentielle en utilisant la méthode d'Euler explicite.

    Paramètres :
    - odefun : La fonction de l'équation différentielle y' = f(t, y, *args)
    - tspan : Tuple (T0, TF) définissant l'intervalle de temps de l'intégration
    - y0 : Condition initiale
    - Nh : Nombre d'intervalles de temps
    - *args : Arguments supplémentaires à passer à odefun

    Retour :
    - t : Vecteur des temps où la solution est calculée
    - u : Solution de l'équation différentielle à chaque instant t
    """
    h = (tspan[1] - tspan[0]) / Nh  # Taille de chaque intervalle
    t = np.linspace(tspan[0], tspan[1], Nh+1)  # Points de temps
    u = np.zeros((Nh+1, len(y0)))  # Initialisation du tableau de solutions
    u[0, :] = y0  # Définir la condition initiale

    for i in range(Nh):
        u[i+1, :] = u[i, :] + h * np.asarray(odefun(t[i], u[i, :], *args))
    
    return t, u

# Exemple d'utilisation de feuler
if __name__ == "__main__":
    # Définition d'une fonction d'équation différentielle simple : y' = -2y
    def odefun(t, y):
        return -2 * y
    
    # Paramètres
    tspan = (0, 5)  # De t=0 à t=5
    y0 = [1]  # Condition initiale y(0) = 1
    Nh = 50  # Nombre d'intervalles

    # Appel de feuler
    t, u = feuler(odefun, tspan, y0, Nh)

    # Affichage ou tracé de la solution
    import matplotlib.pyplot as plt
    plt.plot(t, u)
    plt.xlabel('Temps')
    plt.ylabel('Solution y(t)')
    plt.title('Solution de l\'EDO par la méthode d\'Euler')
    plt.show()
