import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gershgorin_circles(A):
    """
    Trace les cercles de Gershgorin pour une matrice A.

    Paramètres :
    - A : Matrice carrée dont on veut tracer les cercles de Gershgorin.

    Retourne :
    - fig : Objet figure de Plotly contenant les cercles de Gershgorin.
    """

    n = A.shape[0]
    center = np.diag(A)
    radiir = np.sum(np.abs(A - np.diag(center)), axis=1)
    radiic = np.sum(np.abs(A.T - np.diag(center)), axis=1)

    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Créer une figure avec une ligne et deux colonnes
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Cercles de lignes', 'Cercles de colonnes'))

    # Ajouter les cercles pour les lignes
    for k in range(n):
        x_circle_r = np.real(center[k]) + radiir[k] * np.cos(theta)
        y_circle_r = np.imag(center[k]) + radiir[k] * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_circle_r, y=y_circle_r, mode='lines', fill='toself', fillcolor='rgba(255,0,0,0.2)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[np.real(center[k])], y=[np.imag(center[k])], mode='markers', marker=dict(color='Black', size=8)), row=1, col=1)

    # Ajouter les cercles pour les colonnes
    for k in range(n):
        x_circle_c = np.real(center[k]) + radiic[k] * np.cos(theta)
        y_circle_c = np.imag(center[k]) + radiic[k] * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_circle_c, y=y_circle_c, mode='lines', fill='toself', fillcolor='rgba(0,255,0,0.2)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[np.real(center[k])], y=[np.imag(center[k])], mode='markers', marker=dict(color='Black', size=8)), row=1, col=2)

    fig.update_layout(title_text='Cercles de Gershgorin pour les lignes et colonnes', showlegend=False)
    fig.update_xaxes(title_text='Re', row=1, col=1)
    fig.update_yaxes(title_text='Im', row=1, col=1)
    fig.update_xaxes(title_text='Re', row=1, col=2)
    fig.update_yaxes(title_text='Im', row=1, col=2)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
    return fig
#    fig.show()


