"""Configuration de la base de données SQLAlchemy."""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://kriging_user:motdepasse@localhost:5432/kriging_wifi")

moteur = create_engine(DATABASE_URL)
SessionLocale = sessionmaker(autocommit=False, autoflush=False, bind=moteur)


class Base(DeclarativeBase):
    pass


def obtenir_session():
    """Générateur de session pour les dépendances FastAPI."""
    session = SessionLocale()
    try:
        yield session
    finally:
        session.close()
