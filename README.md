# KrigiFi – Cartographie WiFi par Krigeage

Application mobile Android (React Native + TypeScript) permettant de mesurer et visualiser la couverture WiFi d'un bâtiment sous forme de heatmap interpolée par krigeage ordinaire.

---

## Fonctionnalités

### 1. Gestion des bâtiments
- Créer plusieurs bâtiments / lieux
- Nommer et décrire chaque espace
- Accès rapide au plan, aux mesures et à la heatmap depuis l'écran d'accueil

### 2. Construction du plan au sol

**Mode PDR (Pedestrian Dead Reckoning)** — tracer la pièce en marchant :
- L'utilisateur marche le long des murs en tenant le téléphone à la verticale
- L'**accéléromètre** détecte les pas (comptage par pics de magnitude)
- Le **gyroscope** intègre les changements de direction (cap en degrés)
- Les virages supérieurs à 60° créent automatiquement un coin
- Le plan se trace en temps réel sur un canvas SVG

**Mode image** — importer un plan existant :
- Sélectionner une photo depuis la galerie
- Uploader l'image sur le serveur
- Calibrer l'échelle (mètres par pixel) pour des mesures à l'échelle réelle

### 3. Mesure du signal WiFi

- Démarrer la capture : l'app enregistre la position (PDR) et le RSSI WiFi
- **Mesure automatique toutes les 2 secondes** pendant le déplacement
- Médiane sur 3 mesures pour réduire le bruit
- Compatible WiFi (dBm via `WifiManager`) et données mobiles (4G estimé)
- Indicateur visuel de qualité du signal (Excellent / Bon / Moyen / Faible)
- Envoi des mesures au backend par lot pour optimiser les requêtes

### 4. Heatmap par krigeage

- **Krigeage ordinaire** (`pykrige`) sur les points de mesure enregistrés
- Choix du modèle de variogramme : `spherical`, `gaussian`, `exponential`
- Choix de la résolution de la grille : 30×30, 50×50, 80×80
- Visualisation par gradient de couleurs : bleu (faible) → rouge (fort)
- Superposition des points de mesure bruts en option
- Statistiques : RSSI min/max, modèle utilisé

---

## Architecture technique

```
PROJ-wifi-kriging/
├── backend/                    # API FastAPI + PostgreSQL
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── database.py             # Connexion SQLAlchemy
│   ├── modeles.py              # Modèles ORM
│   ├── schemas.py              # Schémas Pydantic
│   ├── routers/
│   │   ├── batiments.py        # CRUD bâtiments
│   │   ├── plans.py            # Pièces + upload image
│   │   ├── mesures.py          # Mesures WiFi
│   │   └── krigeage.py         # Calcul heatmap
│   ├── kriging/
│   │   └── interpolation.py   # Krigeage ordinaire (pykrige)
│   └── requirements.txt
│
├── KrigiFi/                    # Application React Native (TSX)
│   ├── src/
│   │   ├── types/index.ts      # Interfaces TypeScript
│   │   ├── services/
│   │   │   ├── api.ts          # Appels HTTP (axios)
│   │   │   ├── capteurs.ts     # Abstraction IMU (react-native-sensors)
│   │   │   └── wifi.ts         # Mesure WiFi (wifi-reborn + netinfo)
│   │   ├── hooks/
│   │   │   ├── useImu.ts       # Hook PDR (position par inertie)
│   │   │   └── useWifi.ts      # Hook mesure WiFi temporisée
│   │   ├── components/
│   │   │   ├── DessinateurPlan.tsx   # Canvas SVG du plan
│   │   │   └── CarteThermique.tsx    # Heatmap SVG colorée
│   │   ├── screens/
│   │   │   ├── AccueilScreen.tsx
│   │   │   ├── PlanScreen.tsx
│   │   │   ├── MesureScreen.tsx
│   │   │   └── CarteScreen.tsx
│   │   └── navigation/
│   │       └── AppNavigateur.tsx
│   └── android/                # Build Android natif
│
└── Makefile                    # Commandes de développement
```

### Base de données PostgreSQL

| Table            | Contenu                                           |
|------------------|---------------------------------------------------|
| `batiments`      | Bâtiments (nom, description)                     |
| `pieces`         | Pièces polygonales (JSON de points en mètres)    |
| `images_plan`    | Plans uploadés (chemin, échelle mètres/pixel)    |
| `mesures_wifi`   | Mesures RSSI (x, y, dBm, SSID, horodatage)       |

---

## Installation

### Prérequis

| Outil       | Version     |
|-------------|-------------|
| Node.js     | ≥ 22.11.0   |
| Python      | ≥ 3.11      |
| Java (JDK)  | 17          |
| Android SDK | API 34+     |
| ADB         | ≥ 1.0.41    |
| PostgreSQL  | ≥ 14        |

### Installation complète

```bash
# Clone du dépôt
git clone <url>
cd PROJ-wifi-kriging

# Setup complet (venv Python + npm install)
make setup

# Configurer la base de données
cp backend/.env.example backend/.env
# Éditer backend/.env avec votre DATABASE_URL

# Créer la base PostgreSQL
make db-init

# (Alternative manuelle PostgreSQL)
psql -U postgres -c "CREATE USER kriging_user WITH PASSWORD 'motdepasse';"
psql -U postgres -c "CREATE DATABASE kriging_wifi OWNER kriging_user;"
```

---

## Lancement

Dans **3 terminaux** séparés :

```bash
# Terminal 1 – Backend API
make backend

# Terminal 2 – Metro Bundler
make metro

# Terminal 3 – Application Android (émulateur ou appareil connecté)
make android
```

> **WSL2** : Le backend écoute sur `0.0.0.0:8000`. L'app React Native utilise `10.0.2.2:8000` pour atteindre l'hôte Windows depuis l'émulateur Android.

---

## Utilisation pas à pas

1. **Créer un bâtiment** : appuyer sur `+` depuis l'accueil
2. **Tracer le plan** : choisir "Marcher" → tenir le téléphone vertical → marcher le long des murs → appuyer sur "Enregistrer la pièce"
3. **Mesurer le WiFi** : appuyer sur "Mesure" → "Démarrer" → marcher librement dans l'espace → "Envoyer"
4. **Générer la heatmap** : appuyer sur "Carte" → choisir le variogramme → "Calculer la heatmap"

---

## Commandes Makefile

```bash
make setup          # Installation complète
make backend        # Lancer le backend (hot-reload)
make metro          # Lancer Metro
make android        # Lancer sur émulateur/appareil
make clean          # Nettoyer builds et caches
make clean-all      # + node_modules et .venv
make db-init        # Créer la base PostgreSQL
make help           # Afficher toutes les commandes
```

---

## Références

- [Krigeage (Wikipedia)](https://fr.wikipedia.org/wiki/Krigeage)
- [Variogramme (Wikipedia)](https://fr.wikipedia.org/wiki/Variogramme)
- [pykrige – Python Kriging Toolkit](https://github.com/GeoStat-Framework/PyKrige)
- [react-native-sensors](https://github.com/react-native-sensors/react-native-sensors)
- [react-native-wifi-reborn](https://github.com/JuanSeBestia/react-native-wifi-reborn)
