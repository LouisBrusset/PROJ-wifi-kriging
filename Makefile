# ============================================================
#  KrigiFi – Makefile
#  Commandes de développement sous WSL2 / Windows 11
# ============================================================

.PHONY: backend metro android install setup clean db-init tunnel help

BACKEND_DIR := backend
FRONTEND_DIR := KrigiFi
VENV         := $(BACKEND_DIR)/.venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
UVICORN      := $(VENV)/bin/uvicorn

# ── Aide ──────────────────────────────────────────────────────────────────────

help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Backend FastAPI ───────────────────────────────────────────────────────────

backend: ## Lancer le backend FastAPI (hot-reload)
	cd $(BACKEND_DIR) && .venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

backend-setup: ## Créer l'environnement virtuel et installer les dépendances Python
	cd $(BACKEND_DIR) && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && \
	  .venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "  Copiez .env.example → .env et ajustez DATABASE_URL"
	@echo "  cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env"

tunnel: ## Configurer le tunnel phone→Windows→WSL2 (PowerShell admin requis)
	$(eval WSL_IP := $(shell hostname -I | awk '{print $$1}'))
	@echo "IP WSL2 : $(WSL_IP)"
	@echo "Copiez-collez dans PowerShell (admin) :"
	@echo ""
	@echo "  netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=127.0.0.1"
	@echo "  netsh interface portproxy add v4tov4 listenport=8000 listenaddress=127.0.0.1 connectport=8000 connectaddress=$(WSL_IP)"
	@echo "  adb reverse tcp:8000 tcp:8000"
	@echo ""

db-init: ## Créer la base de données PostgreSQL et l'utilisateur
	@echo "Création de la base de données PostgreSQL…"
	sudo -u postgres psql -c "CREATE USER kriging_user WITH PASSWORD 'motdepasse';" || true
	sudo -u postgres psql -c "CREATE DATABASE kriging_wifi OWNER kriging_user;" || true
	@echo "Base créée. Démarrez le backend pour créer les tables automatiquement."

# ── Frontend React Native ─────────────────────────────────────────────────────

install: ## Installer les dépendances npm
	cd $(FRONTEND_DIR) && npm install

metro: ## Lancer le serveur Metro (bundler React Native)
	cd $(FRONTEND_DIR) && npx react-native start --reset-cache

android: ## Compiler et lancer l'app sur l'émulateur / appareil Android
	cd $(FRONTEND_DIR) && npx react-native run-android

android-release: ## Compiler une APK de release
	cd $(FRONTEND_DIR)/android && ./gradlew assembleRelease

# ── Setup complet ─────────────────────────────────────────────────────────────

setup: backend-setup install ## Installation complète (backend + frontend)
	@echo ""
	@echo "  Installation terminée !"
	@echo "  1. Configurez $(BACKEND_DIR)/.env (DATABASE_URL)"
	@echo "  2. make db-init   → créer la base PostgreSQL"
	@echo "  3. make backend   → lancer l'API"
	@echo "  4. make metro     → lancer Metro (nouveau terminal)"
	@echo "  5. make android   → lancer l'app"

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean: ## Nettoyer les artefacts de build Android et le cache Metro
	@echo "Nettoyage Android…"
	rm -rf $(FRONTEND_DIR)/android/build
	rm -rf $(FRONTEND_DIR)/android/app/build
	rm -rf $(FRONTEND_DIR)/android/.gradle
	@echo "Nettoyage cache Metro…"
	rm -rf /tmp/metro-*
	rm -rf $(FRONTEND_DIR)/.metro
	@echo "Nettoyage Python…"
	find $(BACKEND_DIR) -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Nettoyage terminé."

clean-all: clean ## Nettoyage complet (y compris node_modules et venv)
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf $(BACKEND_DIR)/.venv
