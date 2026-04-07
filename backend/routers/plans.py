"""Routeur pour la gestion des plans (pièces et images uploadées)."""

import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from PIL import Image
from database import obtenir_session
import modeles
import schemas

routeur = APIRouter(prefix="/batiments", tags=["Plans"])

DOSSIER_UPLOADS = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(DOSSIER_UPLOADS, exist_ok=True)


# ── Pièces ────────────────────────────────────────────────────────────────────

@routeur.get("/{batiment_id}/pieces", response_model=list[schemas.PieceReponse])
def lister_pieces(batiment_id: int, session: Session = Depends(obtenir_session)):
    _verifier_batiment(batiment_id, session)
    return session.query(modeles.Piece).filter(modeles.Piece.batiment_id == batiment_id).all()


@routeur.post("/{batiment_id}/pieces", response_model=schemas.PieceReponse, status_code=201)
def creer_piece(
    batiment_id: int,
    donnees: schemas.PieceCreation,
    session: Session = Depends(obtenir_session),
):
    _verifier_batiment(batiment_id, session)
    piece = modeles.Piece(
        batiment_id=batiment_id,
        nom=donnees.nom,
        points=[p.model_dump() for p in donnees.points],
    )
    session.add(piece)
    session.commit()
    session.refresh(piece)
    return piece


@routeur.put("/{batiment_id}/pieces/{piece_id}", response_model=schemas.PieceReponse)
def mettre_a_jour_piece(
    batiment_id: int,
    piece_id: int,
    donnees: schemas.PieceCreation,
    session: Session = Depends(obtenir_session),
):
    piece = session.get(modeles.Piece, piece_id)
    if not piece or piece.batiment_id != batiment_id:
        raise HTTPException(status_code=404, detail="Pièce introuvable")
    piece.nom = donnees.nom
    piece.points = [p.model_dump() for p in donnees.points]
    session.commit()
    session.refresh(piece)
    return piece


@routeur.delete("/{batiment_id}/pieces/{piece_id}", status_code=204)
def supprimer_piece(batiment_id: int, piece_id: int, session: Session = Depends(obtenir_session)):
    piece = session.get(modeles.Piece, piece_id)
    if not piece or piece.batiment_id != batiment_id:
        raise HTTPException(status_code=404, detail="Pièce introuvable")
    session.delete(piece)
    session.commit()


# ── Image plan ────────────────────────────────────────────────────────────────

@routeur.post("/{batiment_id}/plan-image", response_model=schemas.ImagePlanReponse, status_code=201)
async def uploader_plan(
    batiment_id: int,
    fichier: UploadFile = File(...),
    session: Session = Depends(obtenir_session),
):
    _verifier_batiment(batiment_id, session)

    chemin = os.path.join(DOSSIER_UPLOADS, f"batiment_{batiment_id}_{fichier.filename}")
    with open(chemin, "wb") as f:
        shutil.copyfileobj(fichier.file, f)

    img = Image.open(chemin)
    largeur_px, hauteur_px = img.size

    # Supprimer l'ancien plan si existant
    existant = session.query(modeles.ImagePlan).filter(
        modeles.ImagePlan.batiment_id == batiment_id
    ).first()
    if existant:
        try:
            os.remove(existant.chemin_fichier)
        except FileNotFoundError:
            pass
        session.delete(existant)

    image_plan = modeles.ImagePlan(
        batiment_id=batiment_id,
        chemin_fichier=chemin,
        largeur_px=largeur_px,
        hauteur_px=hauteur_px,
    )
    session.add(image_plan)
    session.commit()
    session.refresh(image_plan)
    return image_plan


@routeur.patch("/{batiment_id}/plan-image/echelle", response_model=schemas.ImagePlanReponse)
def definir_echelle(
    batiment_id: int,
    donnees: schemas.EchelleUpdate,
    session: Session = Depends(obtenir_session),
):
    image_plan = session.query(modeles.ImagePlan).filter(
        modeles.ImagePlan.batiment_id == batiment_id
    ).first()
    if not image_plan:
        raise HTTPException(status_code=404, detail="Aucun plan image pour ce bâtiment")
    image_plan.metres_par_pixel = donnees.metres_par_pixel
    session.commit()
    session.refresh(image_plan)
    return image_plan


def _verifier_batiment(batiment_id: int, session: Session):
    if not session.get(modeles.Batiment, batiment_id):
        raise HTTPException(status_code=404, detail="Bâtiment introuvable")
