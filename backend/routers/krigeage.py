"""Routeur pour le krigeage et la génération de heatmap."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import obtenir_session
from kriging.interpolation import interpoler_krigeage
import modeles
import schemas

routeur = APIRouter(prefix="/batiments", tags=["Krigeage"])

MESURES_MINIMUM = 4  # Krigeage impossible avec moins de 4 points


@routeur.post("/{batiment_id}/krigeage", response_model=schemas.ResultatKrigeage)
def calculer_heatmap(
    batiment_id: int,
    parametres: schemas.ParametresKrigeage = schemas.ParametresKrigeage(),
    session: Session = Depends(obtenir_session),
):
    """
    Lance le krigeage sur les mesures WiFi du bâtiment et retourne
    une grille interpolée pour générer la heatmap.
    """
    if not session.get(modeles.Batiment, batiment_id):
        raise HTTPException(status_code=404, detail="Bâtiment introuvable")

    mesures = session.query(modeles.MesureWifi).filter(
        modeles.MesureWifi.batiment_id == batiment_id
    ).all()

    if len(mesures) < MESURES_MINIMUM:
        raise HTTPException(
            status_code=422,
            detail=f"Le krigeage nécessite au moins {MESURES_MINIMUM} mesures "
                   f"(actuellement : {len(mesures)})",
        )

    try:
        resultat = interpoler_krigeage(
            points_x=[m.x for m in mesures],
            points_y=[m.y for m in mesures],
            valeurs_rssi=[m.rssi for m in mesures],
            parametres=parametres,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de krigeage : {str(e)}")

    return resultat
