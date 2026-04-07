# KrigiFi — Documentation Technique

Ce document détaille l'implémentation de chaque fonctionnalité de l'application : fondements théoriques, choix techniques, fichiers concernés et permissions Android associées.

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [Permissions Android](#2-permissions-android)
3. [Traçage du plan par PDR (Pedestrian Dead Reckoning)](#3-traçage-du-plan-par-pdr)
4. [Mesure du signal WiFi](#4-mesure-du-signal-wifi)
5. [Krigeage ordinaire et heatmap](#5-krigeage-ordinaire-et-heatmap)
6. [Communication frontend ↔ backend](#6-communication-frontend--backend)
7. [Sources et références](#7-sources-et-références)

---

## 1. Architecture générale

```
┌─────────────────────────────────────────────────┐
│                Téléphone Android                 │
│  ┌──────────────────────────────────────────┐   │
│  │         Application React Native          │   │
│  │  AccueilScreen → PlanScreen               │   │
│  │               → MesureScreen              │   │
│  │               → CarteScreen               │   │
│  │                                           │   │
│  │  useImu (PDR)   useWifi (RSSI)            │   │
│  │  ↑                ↑                       │   │
│  │  Accéléromètre   NetInfo + WifiManager    │   │
│  │  Gyroscope                                │   │
│  └──────────────────────────────────────────┘   │
│         ↕ HTTP via adb reverse :8000             │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│              Backend WSL2 (FastAPI)              │
│  POST /batiments/{id}/krigeage                   │
│         ↓                                        │
│  kriging/interpolation.py (pykrige)              │
│         ↓                                        │
│  PostgreSQL (batiments, pieces, mesures_wifi)    │
└─────────────────────────────────────────────────┘
```

### Fichiers principaux

| Couche | Fichier | Rôle |
|---|---|---|
| Android natif | `MainActivity.kt` | Point d'entrée Android, monte le composant React `KrigiFi` |
| Android natif | `MainApplication.kt` | Initialise React Native, charge les packages via autolinking |
| Navigation | `src/navigation/AppNavigateur.tsx` | Stack Navigator entre les 4 écrans |
| Types | `src/types/index.ts` | Interfaces TypeScript partagées |
| API | `src/services/api.ts` | Toutes les requêtes HTTP vers le backend (axios) |
| Backend | `backend/main.py` | Application FastAPI, CORS, montage des routeurs |
| ORM | `backend/modeles.py` | Modèles SQLAlchemy (Batiment, Piece, MesureWifi…) |

---

## 2. Permissions Android

Les permissions sont déclarées dans `android/app/src/main/AndroidManifest.xml` et demandées dynamiquement au runtime dans `src/hooks/usePermissions.ts`.

### Déclaration statique (AndroidManifest.xml)

```xml
<!-- Réseau (HTTP vers le backend) -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

<!-- WiFi — obligatoire pour lire le SSID et le RSSI sur Android 10+ -->
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.CHANGE_WIFI_STATE" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />

<!-- Capteurs inertiels -->
<uses-permission android:name="android.permission.BODY_SENSORS" />
<uses-permission android:name="android.permission.HIGH_SAMPLING_RATE_SENSORS" />

<!-- Galerie photos (import de plan) -->
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
```

### Permissions runtime (Android 6+)

Sur Android 6+, certaines permissions sont **dangereuses** et doivent être accordées par l'utilisateur à l'exécution. Elles sont demandées dans `src/hooks/usePermissions.ts` via `PermissionsAndroid.requestMultiple()` :

| Permission | Pourquoi | Fichier qui l'utilise |
|---|---|---|
| `ACCESS_FINE_LOCATION` | Obligatoire depuis Android 10 pour lire le SSID WiFi | `src/services/wifi.ts`, `src/hooks/useWifi.ts` |
| `BODY_SENSORS` | Accès à l'accéléromètre et au gyroscope | `src/services/capteurs.ts`, `src/hooks/useImu.ts` |

> **Pourquoi la localisation pour le WiFi ?** Depuis Android 10, Google considère que la liste des réseaux WiFi à portée révèle la position physique de l'utilisateur. L'API `WifiManager` exige donc `ACCESS_FINE_LOCATION` même pour lire le RSSI du réseau connecté.

### Fichiers Android natifs

**`MainActivity.kt`** — Activité principale Android. Elle hérite de `ReactActivity` et retourne `"KrigiFi"` comme nom du composant JS racine. Elle ne contient aucune logique métier ; tout est délégué à React Native via le bridge JS/natif.

**`MainApplication.kt`** — Application Android. Elle initialise le runtime React Native (`loadReactNative`) et charge automatiquement tous les modules natifs déclarés (`PackageList(this).packages`) grâce au mécanisme d'**autolinking** de React Native 0.60+. C'est ainsi que `react-native-sensors`, `react-native-wifi-reborn`, etc. sont enregistrés sans code natif manuel.

---

## 3. Traçage du plan par PDR

### Fondements théoriques

Le **PDR (Pedestrian Dead Reckoning)** est une méthode d'estimation de position basée sur l'intégration des données inertielles. En l'absence de GPS intérieur, on reconstitue la trajectoire à partir de deux mesures :

- **Détection de pas** via l'accéléromètre
- **Intégration du cap** via le gyroscope

La position à l'instant *t* est :

```
x(t) = x(t-1) + L × sin(θ)
y(t) = y(t-1) + L × cos(θ)
```

où *L* est la longueur de pas (≈ 0,75 m) et *θ* le cap courant.

Le cap est intégré à partir du gyroscope (axe Z) :

```
θ(t) = θ(t-1) + ωz × Δt
```

Les erreurs s'accumulent avec le temps (dérive), ce qui rend le PDR imprécis pour de longues trajectoires. Pour le tracé d'une pièce (quelques mètres), la précision est suffisante.

### Implémentation

**Fichier principal :** `src/hooks/useImu.ts`

**Accès aux capteurs :** `src/services/capteurs.ts` → `react-native-sensors`

```
Accéléromètre (50 Hz)
    → calculerMagnitude(x, y, z) = √(x² + y² + z²)
    → Détection de pic > 11,5 m/s² avec délai réfractaire (300 ms)
    → Un pic = 1 pas → avancer de 0,75 m dans la direction du cap

Gyroscope (50 Hz)
    → Intégration : Δθ = ωz × 0,05 s (en degrés)
    → Cap courant = somme des Δθ

Détection de virage
    → Δcap accumulé > 60° depuis le dernier sommet → nouveau sommet enregistré
    → Les sommets forment le polygone de la pièce
```

**Rendu :** `src/components/DessinateurPlan.tsx`

Le plan est affiché via `react-native-svg`. Les coordonnées métriques sont projetées en pixels SVG avec un facteur d'échelle calculé dynamiquement à partir des bornes de la trajectoire.

### Limites connues

- La dérive gyroscopique fausse le cap sur de longues trajectoires
- La longueur de pas est fixe (0,75 m) ; elle varie selon la morphologie et l'allure
- Les surfaces glissantes ou les escaliers perturbent la détection de pas

---

## 4. Mesure du signal WiFi

### Fondements théoriques

Le **RSSI (Received Signal Strength Indicator)** mesure la puissance du signal électromagnétique reçu, exprimée en **dBm** (décibels-milliwatts). C'est une échelle logarithmique :

```
P(dBm) = 10 × log₁₀(P / 1 mW)
```

Valeurs typiques pour le WiFi :

| RSSI (dBm) | Qualité |
|---|---|
| ≥ -50 | Excellent |
| -50 à -60 | Bon |
| -60 à -70 | Moyen |
| -70 à -80 | Faible |
| < -80 | Très faible |

La puissance reçue décroît avec la distance selon le **modèle de propagation en espace libre** :

```
PL(d) = PL(d₀) + 10n × log₁₀(d/d₀)
```

où *n* est le coefficient de propagation (2 en espace libre, 2-4 en intérieur selon les matériaux). C'est cette variation spatiale que le krigeage va interpoler.

### Implémentation

**Fichiers :** `src/services/wifi.ts`, `src/hooks/useWifi.ts`

```
NetInfo.fetch()
    → Détecter type réseau (wifi / cellular)
    → Si WiFi : lire details.strength (0-100, Android)
    → Conversion approx : RSSI ≈ -100 + strength × 0,7 (dBm)

WifiManager.getCurrentWifiSSID()
    → Lire le nom du réseau connecté (nécessite ACCESS_FINE_LOCATION)

Médiane sur 3 mesures espacées de 250 ms
    → Réduire le bruit de mesure instantané

Intervalle automatique : 1 mesure toutes les 2 secondes
    → Associée à la position PDR courante (x, y) en mètres
```

**Envoi en lot :** `POST /batiments/{id}/mesures/lot` — toutes les mesures de la session sont envoyées en une seule requête HTTP.

### Note sur Android 10+

Depuis Android 10, l'API `WifiManager.getRssi()` n'est plus accessible directement depuis une app tierce sans permission `ACCESS_FINE_LOCATION`. Le RSSI est donc lu indirectement via `@react-native-community/netinfo` qui expose `details.strength` (0-100), converti en dBm.

---

## 5. Krigeage ordinaire et heatmap

### Fondements théoriques

Le **krigeage** (du nom du géologue D.G. Krige) est une méthode d'interpolation spatiale géostatistique. Contrairement à l'interpolation bilinéaire ou IDW (inverse distance weighting), le krigeage est un **estimateur optimal non biaisé** au sens des moindres carrés.

#### Variogramme

Le variogramme γ(h) mesure la dissimilarité entre deux mesures séparées par une distance *h* :

```
γ(h) = ½ × E[(Z(x) - Z(x+h))²]
```

Trois modèles sont proposés dans l'interface :

| Modèle | Formule | Caractéristique |
|---|---|---|
| **Sphérique** | γ(h) = c₀ + c × [1.5(h/a) - 0.5(h/a)³] si h ≤ a | Palier net, adapté aux zones de transition |
| **Gaussien** | γ(h) = c₀ + c × [1 - exp(-(h/a)²)] | Transitions douces, sur-régularise |
| **Exponentiel** | γ(h) = c₀ + c × [1 - exp(-h/a)] | Variation rapide, adapté aux milieux hétérogènes |

*c₀ = pépite (bruit), c = palier, a = portée*

#### Système krigeant

Pour estimer la valeur en un point x₀ non mesuré, on cherche les poids λᵢ minimisant la variance d'estimation :

```
Z*(x₀) = Σ λᵢ × Z(xᵢ)     sous contrainte Σ λᵢ = 1
```

Les poids λᵢ sont solutions du système linéaire :

```
[ Γ  1 ] [ λ ]   [ γ₀ ]
[ 1ᵀ 0 ] [ μ ] = [  1  ]
```

où Γᵢⱼ = γ(xᵢ - xⱼ) et γ₀ᵢ = γ(x₀ - xᵢ).

### Implémentation

**Backend :** `backend/kriging/interpolation.py`
**Bibliothèque :** `pykrige` (OrdinaryKriging)
**Déclenchement :** `POST /batiments/{id}/krigeage` → `backend/routers/krigeage.py`

```python
ok = OrdinaryKriging(x, y, z, variogram_model='spherical')
z_pred, z_var = ok.execute("grid", grille_x, grille_y)
```

La grille retournée (50×50 par défaut) contient pour chaque cellule :
- `valeur` : RSSI interpolé (dBm)
- `variance` : incertitude du krigeage (plus élevée loin des points de mesure)

**Validation en entrée :**
- Minimum 4 points de mesure
- Les points doivent couvrir une surface 2D (pas tous alignés)
- Plage spatiale > 0,1 m dans chaque dimension

**Rendu :** `src/components/CarteThermique.tsx`

Chaque cellule de la grille est rendue comme un rectangle SVG coloré selon ce gradient :

```
-90 dBm → Bleu     #0000FF  (signal très faible)
-70 dBm → Cyan     #00FFFF
-60 dBm → Vert     #00FF00
-50 dBm → Jaune    #FFFF00
-30 dBm → Rouge    #FF0000  (signal excellent)
```

---

## 6. Communication frontend ↔ backend

### API REST

**Fichier :** `src/services/api.ts` (axios, base URL `http://localhost:8000`)

| Méthode | Endpoint | Usage |
|---|---|---|
| GET | `/batiments/` | Lister les bâtiments |
| POST | `/batiments/` | Créer un bâtiment |
| DELETE | `/batiments/{id}` | Supprimer un bâtiment |
| GET | `/batiments/{id}/pieces` | Lister les pièces |
| POST | `/batiments/{id}/pieces` | Sauvegarder une pièce (polygone) |
| POST | `/batiments/{id}/plan-image` | Uploader un plan image |
| POST | `/batiments/{id}/mesures/lot` | Envoyer les mesures WiFi par lot |
| GET | `/batiments/{id}/mesures` | Récupérer les mesures |
| POST | `/batiments/{id}/krigeage` | Lancer le krigeage → retourne la grille |

### Tunnel réseau WSL2 → téléphone

```
FastAPI (WSL2:0.0.0.0:8000)
    → [netsh portproxy] → Windows:127.0.0.1:8000
    → [adb reverse tcp:8000 tcp:8000] → Téléphone:localhost:8000
```

L'URL dans le frontend (`http://localhost:8000`) est résolue par Android vers le socket ADB reverse, qui remonte jusqu'au backend WSL2. Ce mécanisme ne dépend pas du réseau WiFi et fonctionne en USB ou en WiFi dès lors qu'ADB est actif.

---

## 7. Sources et références

### Krigeage et géostatistique
- Krige, D.G. (1951). *A statistical approach to some basic mine valuation problems on the Witwatersrand.* J. Chem. Metall. Min. Soc. South Africa, 52(6), 119-139.
- Matheron, G. (1963). *Principles of geostatistics.* Economic Geology, 58(8), 1246-1266.
- Cressie, N. (1993). *Statistics for Spatial Data.* Wiley.
- [Krigeage — Wikipedia FR](https://fr.wikipedia.org/wiki/Krigeage)
- [Variogramme — Wikipedia FR](https://fr.wikipedia.org/wiki/Variogramme)

### Propagation du signal WiFi
- Rappaport, T.S. (2002). *Wireless Communications: Principles and Practice* (2nd ed.). Prentice Hall.
- Friis, H.T. (1946). *A Note on a Simple Transmission Formula.* Proceedings of the IRE, 34(5), 254-256.
- [Path loss models — Wikipedia EN](https://en.wikipedia.org/wiki/Log-distance_path_loss_model)

### PDR (Pedestrian Dead Reckoning)
- Foxlin, E. (2005). *Pedestrian tracking with shoe-mounted inertial sensors.* IEEE Computer Graphics and Applications, 25(6), 38-46.
- Jimenez, A.R. et al. (2009). *A comparison of pedestrian dead-reckoning algorithms using a low-cost MEMS IMU.* IEEE WISP 2009.
- [Dead reckoning — Wikipedia EN](https://en.wikipedia.org/wiki/Dead_reckoning)

### Bibliothèques utilisées
- [PyKrige — GeoStat-Framework](https://github.com/GeoStat-Framework/PyKrige) — krigeage ordinaire en Python
- [react-native-sensors](https://github.com/react-native-sensors/react-native-sensors) — accéléromètre et gyroscope
- [react-native-wifi-reborn](https://github.com/JuanSeBestia/react-native-wifi-reborn) — lecture SSID WiFi
- [@react-native-community/netinfo](https://github.com/react-native-community/react-native-netinfo) — état réseau et force du signal
- [FastAPI](https://fastapi.tiangolo.com/) — framework API Python
- [SQLAlchemy](https://www.sqlalchemy.org/) — ORM Python
- [react-native-svg](https://github.com/software-mansion/react-native-svg) — rendu vectoriel

### Android
- [Android Developers — WifiManager](https://developer.android.com/reference/android/net/wifi/WifiManager)
- [Android Developers — SensorManager](https://developer.android.com/reference/android/hardware/SensorManager)
- [Android Developers — Permissions](https://developer.android.com/guide/topics/permissions/overview)
- [Android Developers — ADB](https://developer.android.com/tools/adb)
