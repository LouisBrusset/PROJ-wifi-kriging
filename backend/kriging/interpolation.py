"""Interpolation par krigeage ordinaire (pykrige)."""

import numpy as np
from pykrige.ok import OrdinaryKriging
from schemas import CelluleHeatmap, ResultatKrigeage, ParametresKrigeage


def interpoler_krigeage(
    points_x: list[float],
    points_y: list[float],
    valeurs_rssi: list[float],
    parametres: ParametresKrigeage,
) -> ResultatKrigeage:
    """
    Effectue un krigeage ordinaire sur les mesures WiFi.

    Args:
        points_x: Coordonnées X des mesures (mètres)
        points_y: Coordonnées Y des mesures (mètres)
        valeurs_rssi: Valeurs RSSI en dBm
        parametres: Paramètres du krigeage (résolution, variogramme)

    Returns:
        Grille interpolée avec valeurs et variances
    """
    x = np.array(points_x)
    y = np.array(points_y)
    z = np.array(valeurs_rssi)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Vérifier que les points couvrent une zone 2D suffisante
    if x_max - x_min < 0.1 and y_max - y_min < 0.1:
        raise ValueError(
            "Tous les points de mesure sont au même endroit. "
            "Déplacez-vous davantage avant de mesurer."
        )
    if x_max - x_min < 0.1:
        raise ValueError(
            "Tous les points sont alignés sur l'axe Y (pas de déplacement horizontal). "
            "Déplacez-vous latéralement pour couvrir une surface 2D."
        )
    if y_max - y_min < 0.1:
        raise ValueError(
            "Tous les points sont alignés sur l'axe X (pas de déplacement vertical). "
            "Déplacez-vous en profondeur pour couvrir une surface 2D."
        )

    # Marge de 10% autour des mesures
    marge_x = (x_max - x_min) * 0.1
    marge_y = (y_max - y_min) * 0.1
    x_min -= marge_x
    x_max += marge_x
    y_min -= marge_y
    y_max += marge_y

    grille_x = np.linspace(x_min, x_max, parametres.resolution)
    grille_y = np.linspace(y_min, y_max, parametres.resolution)

    ok = OrdinaryKriging(
        x, y, z,
        variogram_model=parametres.variogramme,
        verbose=False,
        enable_plotting=False,
    )

    z_pred, z_var = ok.execute("grid", grille_x, grille_y)

    cellules: list[CelluleHeatmap] = []
    for i, cy in enumerate(grille_y):
        for j, cx in enumerate(grille_x):
            cellules.append(CelluleHeatmap(
                x=float(cx),
                y=float(cy),
                valeur=float(z_pred[i, j]),
                variance=float(z_var[i, j]),
            ))

    return ResultatKrigeage(
        cellules=cellules,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        rssi_min=float(z_pred.min()),
        rssi_max=float(z_pred.max()),
        resolution=parametres.resolution,
    )
