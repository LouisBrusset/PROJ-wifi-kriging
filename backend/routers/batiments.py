"""Routeur pour la gestion des bâtiments."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import obtenir_session
import modeles
import schemas

routeur = APIRouter(prefix="/batiments", tags=["Bâtiments"])


@routeur.get("/", response_model=list[schemas.BatimentReponse])
def lister_batiments(session: Session = Depends(obtenir_session)):
    return session.query(modeles.Batiment).all()


@routeur.post("/", response_model=schemas.BatimentReponse, status_code=201)
def creer_batiment(donnees: schemas.BatimentCreation, session: Session = Depends(obtenir_session)):
    batiment = modeles.Batiment(**donnees.model_dump())
    session.add(batiment)
    session.commit()
    session.refresh(batiment)
    return batiment


@routeur.get("/{batiment_id}", response_model=schemas.BatimentReponse)
def obtenir_batiment(batiment_id: int, session: Session = Depends(obtenir_session)):
    batiment = session.get(modeles.Batiment, batiment_id)
    if not batiment:
        raise HTTPException(status_code=404, detail="Bâtiment introuvable")
    return batiment


@routeur.delete("/{batiment_id}", status_code=204)
def supprimer_batiment(batiment_id: int, session: Session = Depends(obtenir_session)):
    batiment = session.get(modeles.Batiment, batiment_id)
    if not batiment:
        raise HTTPException(status_code=404, detail="Bâtiment introuvable")
    session.delete(batiment)
    session.commit()
