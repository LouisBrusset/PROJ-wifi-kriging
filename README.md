# KrigiFi – Cartographie WiFi par Krigeage

Application mobile Android (React Native + TypeScript) permettant de cartographier la couverture WiFi d'un bâtiment. L'utilisateur trace le plan de son espace à l'aide des capteurs inertiels du téléphone, relève le signal WiFi en se déplaçant, puis génère une heatmap interpolée par **krigeage ordinaire**.

> Documentation technique détaillée → [TECHNIQUE.md](TECHNIQUE.md)

---

## Fonctionnalités

| Écran | Description |
|---|---|
| **Accueil** | Gestion des bâtiments (créer, lister, supprimer) |
| **Plan** | Tracer les pièces par marche (PDR) ou importer une image |
| **Mesure** | Capturer le signal WiFi/4G en se déplaçant |
| **Carte** | Générer et visualiser la heatmap par krigeage |

---

## Prérequis

| Outil | Version minimale |
|---|---|
| Node.js | ≥ 22.11.0 |
| Python | ≥ 3.11 |
| Java JDK | 17 |
| Android SDK | API 31+ |
| ADB | ≥ 1.0.41 |
| PostgreSQL | ≥ 14 |

Environnement cible : **WSL2 sous Windows 11**, avec Android SDK installé côté Windows.

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url>
cd PROJ-wifi-kriging

# 2. Installer backend Python + dépendances npm
make setup

# 3. Configurer la base de données
cp backend/.env.example backend/.env
# Éditer backend/.env → ajuster DATABASE_URL si besoin

# 4. Créer l'utilisateur et la base PostgreSQL
make db-init
# (demande le mot de passe sudo)
```

---

## Lancer l'application

Ouvrir **3 terminaux WSL2** :

```bash
# Terminal 1 — API backend (hot-reload)
make backend

# Terminal 2 — Metro bundler React Native
make metro

# Terminal 3 — Compiler et installer sur l'appareil
make android
```

Après le premier build (~5-10 min), les suivants prennent 1-2 min grâce au cache Gradle.

---

## Connexion téléphone → backend (WSL2)

Le backend tourne dans WSL2 ; le téléphone doit pouvoir l'atteindre. Il existe deux cas selon la connexion USB ou WiFi.

### Cas 1 – Téléphone branché en USB (recommandé)

Le pont USB est géré par ADB côté **Windows**. Il faut créer un tunnel en deux étapes :

**Étape A – Port forwarding Windows → WSL2** (PowerShell admin) :
```powershell
# Récupérer l'IP WSL2 (change à chaque redémarrage)
wsl hostname -I

# Créer le forwarding
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=127.0.0.1
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=127.0.0.1 connectport=8000 connectaddress=<IP_WSL2>

# Optionnel : autoriser le port dans le pare-feu Windows
New-NetFirewallRule -DisplayName "KrigiFi" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

**Étape B – Tunnel ADB reverse** (terminal Windows ou WSL2) :
```bash
adb reverse tcp:8000 tcp:8000
```

**Résultat** :
```
Téléphone:localhost:8000
    → [adb reverse] → Windows:127.0.0.1:8000
    → [netsh portproxy] → WSL2:<IP_WSL2>:8000
    → FastAPI ✓
```

> `make tunnel` dans WSL2 affiche les commandes exactes avec l'IP WSL2 du moment.

**À répéter après chaque :**
- Redémarrage du PC (l'IP WSL2 change)
- Reconnexion du câble USB (`adb reverse` se réinitialise)

---

### Cas 2 – Téléphone connecté en WiFi (ADB over WiFi)

Activer le débogage WiFi sur le téléphone (`Paramètres → Options développeur → Débogage sans fil`), puis depuis Windows :

```powershell
adb connect <IP_TELEPHONE>:5555
adb reverse tcp:8000 tcp:8000
```

Le reste est identique au cas USB. L'IP du téléphone doit être sur le même réseau local que le PC.

---

### Cas 3 – Mirroring Windows (Scrcpy, Phone Link…)

Si l'affichage est mirroré via un outil Windows (ex: scrcpy, Samsung DeX, Phone Link), ADB tourne côté **Windows**, pas WSL2. Il faut donc :

1. Lancer `adb reverse` depuis un **terminal PowerShell Windows** (pas WSL2)
2. Vérifier que le port forwarding `netsh` est bien actif (étape A ci-dessus)
3. Depuis WSL2, tester la connectivité :

```bash
# Depuis WSL2 — doit répondre {"statut":"ok"}
curl http://localhost:8000/sante
```

---

### Diagnostic rapide

```bash
# Backend accessible depuis WSL2 ?
curl http://localhost:8000/sante

# Appareil détecté par ADB ?
adb devices

# Tunnel reverse actif ?
adb reverse --list

# IP WSL2 actuelle
hostname -I

# Forwarding Windows actif ? (PowerShell)
netsh interface portproxy show all
```

---

### Erreurs courantes

| Erreur | Cause | Solution |
|---|---|---|
| `Impossible de contacter le serveur` | `adb reverse` absent ou expiré | Relancer `adb reverse tcp:8000 tcp:8000` |
| `adb: protocol fault` | Serveur ADB WSL2 et Windows en conflit | `adb kill-server && adb start-server` |
| `Address already in use` (port 8000) | Processus Windows sur ce port | `netsh interface portproxy delete...` puis relancer |
| `FATAL: Peer authentication failed` | PostgreSQL nécessite `sudo -u postgres` | Utiliser `make db-init` (pas `psql -U postgres` direct) |
| `Each lower bound must be strictly less` | Points de mesure tous alignés | Marcher en 2D (zigzag), pas en ligne droite |
| Build Gradle bloqué à 99% | ADB non disponible pendant l'install | `adb start-server` puis `make android` à nouveau |

---

## Utilisation pas à pas

1. **Créer un bâtiment** → bouton `+` sur l'écran d'accueil
2. **Tracer le plan** → écran Plan → "Marcher dans la pièce" → marcher le long des murs → "Enregistrer la pièce"
3. **Mesurer le WiFi** → écran Mesure → "Démarrer" → se déplacer librement en zigzag → "Envoyer"
4. **Heatmap** → écran Carte → choisir le variogramme → "Calculer la heatmap"

> Pour le krigeage, les points de mesure doivent couvrir une **surface 2D** (pas une simple ligne). Une vingtaine de points répartis en zigzag donne un bon résultat.

---

## Commandes Makefile

```bash
make setup          # Installation complète (Python venv + npm)
make backend        # Lancer le backend FastAPI (hot-reload)
make metro          # Lancer Metro bundler
make android        # Compiler et installer sur appareil/émulateur
make tunnel         # Afficher les commandes de tunnel WSL2↔Windows
make db-init        # Créer la base PostgreSQL
make clean          # Nettoyer builds Android et cache Metro
make clean-all      # + node_modules et .venv
make help           # Lister toutes les commandes
```

---

## Références

- [Krigeage — Wikipedia](https://fr.wikipedia.org/wiki/Krigeage)
- [Variogramme — Wikipedia](https://fr.wikipedia.org/wiki/Variogramme)
- [PyKrige — Python Kriging Toolkit](https://github.com/GeoStat-Framework/PyKrige)
- [react-native-sensors](https://github.com/react-native-sensors/react-native-sensors)
- [react-native-wifi-reborn](https://github.com/JuanSeBestia/react-native-wifi-reborn)
- [ADB over WiFi — Android Developers](https://developer.android.com/tools/adb#wireless)
