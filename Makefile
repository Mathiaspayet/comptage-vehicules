.PHONY: test test-cov lint build logs shell clean

# Lancer les tests unitaires
test:
	python -m pytest --tb=short -q

# Tests avec rapport de couverture
test-cov:
	python -m pytest --tb=short --cov=src --cov-report=term-missing -q

# Linter (si ruff est disponible)
lint:
	ruff check src/ tests/ || true

# Build de l'image Docker locale
build:
	docker build -t comptage-vehicules:local .

# Logs en direct du conteneur
logs:
	docker logs -f comptage-vehicules

# Shell dans le conteneur en cours d'exécution
shell:
	docker exec -it comptage-vehicules /bin/bash

# Redémarrer le conteneur
restart:
	docker compose restart

# Arrêter et supprimer le conteneur (les données /app/data sont conservées)
down:
	docker compose down

# Mettre à jour l'image et redémarrer
update:
	docker compose pull && docker compose up -d
