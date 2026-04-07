"""Point d'entrée de l'API FastAPI KrigiFi."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import moteur, Base
from routers import batiments, plans, mesures, krigeage

# Création des tables si elles n'existent pas
Base.metadata.create_all(bind=moteur)

app = FastAPI(
    title="KrigiFi API",
    description="API pour la cartographie WiFi par krigeage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, restreindre aux origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les images uploadées
DOSSIER_UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(DOSSIER_UPLOADS, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=DOSSIER_UPLOADS), name="uploads")

# Enregistrement des routeurs
app.include_router(batiments.routeur)
app.include_router(plans.routeur)
app.include_router(mesures.routeur)
app.include_router(krigeage.routeur)


@app.get("/")
def accueil():
    return {"message": "KrigiFi API opérationnelle", "docs": "/docs"}


@app.get("/sante")
def sante():
    return {"statut": "ok"}
