"""Modèles SQLAlchemy pour la base de données."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from database import Base


class Batiment(Base):
    """Un bâtiment ou lieu dont on cartographie le WiFi."""
    __tablename__ = "batiments"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    cree_le = Column(DateTime, default=datetime.utcnow)

    pieces = relationship("Piece", back_populates="batiment", cascade="all, delete-orphan")
    mesures = relationship("MesureWifi", back_populates="batiment", cascade="all, delete-orphan")
    image_plan = relationship("ImagePlan", back_populates="batiment", uselist=False, cascade="all, delete-orphan")


class Piece(Base):
    """Une pièce ou zone dans un bâtiment, définie par ses coins (polygone)."""
    __tablename__ = "pieces"

    id = Column(Integer, primary_key=True, index=True)
    batiment_id = Column(Integer, ForeignKey("batiments.id"), nullable=False)
    nom = Column(String(200), nullable=False)
    # Liste de points {x: float, y: float} en mètres depuis l'origine du bâtiment
    points = Column(JSON, nullable=False, default=list)
    cree_le = Column(DateTime, default=datetime.utcnow)

    batiment = relationship("Batiment", back_populates="pieces")


class ImagePlan(Base):
    """Plan uploadé par l'utilisateur (image raster)."""
    __tablename__ = "images_plan"

    id = Column(Integer, primary_key=True, index=True)
    batiment_id = Column(Integer, ForeignKey("batiments.id"), nullable=False, unique=True)
    chemin_fichier = Column(String(500), nullable=False)
    # Facteur d'échelle : mètres par pixel
    metres_par_pixel = Column(Float, nullable=True)
    largeur_px = Column(Integer, nullable=True)
    hauteur_px = Column(Integer, nullable=True)
    cree_le = Column(DateTime, default=datetime.utcnow)

    batiment = relationship("Batiment", back_populates="image_plan")


class MesureWifi(Base):
    """Une mesure de signal WiFi à une position donnée."""
    __tablename__ = "mesures_wifi"

    id = Column(Integer, primary_key=True, index=True)
    batiment_id = Column(Integer, ForeignKey("batiments.id"), nullable=False)
    # Position en mètres dans le repère du bâtiment
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    # Signal en dBm (valeur négative, ex: -65)
    rssi = Column(Float, nullable=False)
    ssid = Column(String(200), nullable=True)
    type_reseau = Column(String(20), default="wifi")  # "wifi" ou "mobile"
    mesure_le = Column(DateTime, default=datetime.utcnow)

    batiment = relationship("Batiment", back_populates="mesures")
