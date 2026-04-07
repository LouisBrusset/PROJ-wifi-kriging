"""Routeur pour la gestion des mesures WiFi."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import obtenir_session
import modeles
import schemas

routeur = APIRouter(prefix="/batiments", tags=["Mesures WiFi"])


@routeur.get("/{batiment_id}/mesures", response_model=list[schemas.MesureReponse])
def lister_mesures(batiment_id: int, session: Session = Depends(obtenir_session)):
    _verifier_batiment(batiment_id, session)
    return session.query(modeles.MesureWifi).filter(
        modeles.MesureWifi.batiment_id == batiment_id
    ).all()


@routeur.post("/{batiment_id}/mesures", response_model=schemas.MesureReponse, status_code=201)
def ajouter_mesure(
    batiment_id: int,
    donnees: schemas.MesureCreation,
    session: Session = Depends(obtenir_session),
):
    _verifier_batiment(batiment_id, session)
    mesure = modeles.MesureWifi(batiment_id=batiment_id, **donnees.model_dump())
    session.add(mesure)
    session.commit()
    session.refresh(mesure)
    return mesure


@routeur.post("/{batiment_id}/mesures/lot", response_model=list[schemas.MesureReponse], status_code=201)
def ajouter_mesures_lot(
    batiment_id: int,
    donnees: list[schemas.MesureCreation],
    session: Session = Depends(obtenir_session),
):
    """Ajouter plusieurs mesures en une seule requête."""
    _verifier_batiment(batiment_id, session)
    mesures = [modeles.MesureWifi(batiment_id=batiment_id, **m.model_dump()) for m in donnees]
    session.add_all(mesures)
    session.commit()
    for m in mesures:
        session.refresh(m)
    return mesures


@routeur.delete("/{batiment_id}/mesures", status_code=204)
def effacer_mesures(batiment_id: int, session: Session = Depends(obtenir_session)):
    """Supprimer toutes les mesures d'un bâtiment."""
    _verifier_batiment(batiment_id, session)
    session.query(modeles.MesureWifi).filter(
        modeles.MesureWifi.batiment_id == batiment_id
    ).delete()
    session.commit()


@routeur.delete("/{batiment_id}/mesures/{mesure_id}", status_code=204)
def supprimer_mesure(batiment_id: int, mesure_id: int, session: Session = Depends(obtenir_session)):
    mesure = session.get(modeles.MesureWifi, mesure_id)
    if not mesure or mesure.batiment_id != batiment_id:
        raise HTTPException(status_code=404, detail="Mesure introuvable")
    session.delete(mesure)
    session.commit()


def _verifier_batiment(batiment_id: int, session: Session):
    if not session.get(modeles.Batiment, batiment_id):
        raise HTTPException(status_code=404, detail="Bâtiment introuvable")
