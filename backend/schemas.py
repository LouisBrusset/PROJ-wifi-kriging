"""Schémas Pydantic pour la validation des données."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ── Bâtiment ──────────────────────────────────────────────────────────────────

class BatimentCreation(BaseModel):
    nom: str
    description: Optional[str] = None


class BatimentReponse(BaseModel):
    id: int
    nom: str
    description: Optional[str]
    cree_le: datetime

    model_config = {"from_attributes": True}


# ── Pièce ─────────────────────────────────────────────────────────────────────

class Point(BaseModel):
    x: float
    y: float


class PieceCreation(BaseModel):
    nom: str
    points: list[Point]


class PieceReponse(BaseModel):
    id: int
    batiment_id: int
    nom: str
    points: list[Point]
    cree_le: datetime

    model_config = {"from_attributes": True}


# ── Image plan ────────────────────────────────────────────────────────────────

class EchelleUpdate(BaseModel):
    metres_par_pixel: float


class ImagePlanReponse(BaseModel):
    id: int
    batiment_id: int
    chemin_fichier: str
    metres_par_pixel: Optional[float]
    largeur_px: Optional[int]
    hauteur_px: Optional[int]

    model_config = {"from_attributes": True}


# ── Mesure WiFi ───────────────────────────────────────────────────────────────

class MesureCreation(BaseModel):
    x: float
    y: float
    rssi: float
    ssid: Optional[str] = None
    type_reseau: str = "wifi"


class MesureReponse(BaseModel):
    id: int
    batiment_id: int
    x: float
    y: float
    rssi: float
    ssid: Optional[str]
    type_reseau: str
    mesure_le: datetime

    model_config = {"from_attributes": True}


# ── Krigeage ──────────────────────────────────────────────────────────────────

class ParametresKrigeage(BaseModel):
    resolution: int = 50          # Nombre de cellules par axe (50x50)
    variogramme: str = "spherical"  # Modèle : spherical, gaussian, exponential


class CelluleHeatmap(BaseModel):
    x: float
    y: float
    valeur: float
    variance: float


class ResultatKrigeage(BaseModel):
    cellules: list[CelluleHeatmap]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    rssi_min: float
    rssi_max: float
    resolution: int
