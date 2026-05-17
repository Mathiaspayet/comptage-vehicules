# Comptage de véhicules — Sonnette Reolink + NAS Synology

Système autonome de comptage des véhicules passant devant une sonnette vidéo. Tourne 24h/24 sur un NAS Synology DS218+, se met à jour automatiquement à chaque modification du code.

---

## Ce que fait ce système

- **Surveille** le dossier où la sonnette dépose ses vidéos.
- **Filtre** rapidement les images vides (rue calme) pour ne pas gaspiller de ressources.
- **Détecte et compte** les véhicules (voiture, camion, bus, moto) qui franchissent une ligne virtuelle configurée par vos soins.
- **Stocke** les résultats dans une base de données locale.
- **Affiche** un tableau de bord web : statistiques horaires, vue quotidienne, graphiques.

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Création du dépôt GitHub](#2-création-du-dépôt-github)
3. [Rendre l'image publique sur GHCR](#3-rendre-limage-publique-sur-ghcr)
4. [Préparer le NAS](#4-préparer-le-nas)
5. [Configurer l'application](#5-configurer-lapplication)
6. [Calibration : définir la ligne de comptage](#6-calibration--définir-la-ligne-de-comptage)
7. [Démarrer l'application](#7-démarrer-lapplication)
8. [Accéder au tableau de bord](#8-accéder-au-tableau-de-bord)
9. [Mises à jour automatiques](#9-mises-à-jour-automatiques)
10. [Revenir à une version précédente](#10-revenir-à-une-version-précédente)
11. [Dépannage](#11-dépannage)

---

## 1. Prérequis

- Un compte [GitHub](https://github.com) (gratuit).
- Le logiciel **Git** installé sur votre ordinateur ([télécharger](https://git-scm.com)).
- **Docker Desktop** (optionnel, pour tester en local avant de déployer sur le NAS).
- Sur le NAS : **Container Manager** installé depuis le Centre de paquets DSM.

---

## 2. Création du dépôt GitHub

> Cette étape est à faire une seule fois, depuis votre ordinateur.

1. Connectez-vous à [github.com](https://github.com).
2. Cliquez sur **New repository** (bouton vert en haut à droite).
3. Donnez-lui un nom, par exemple `comptage-vehicules`.
4. Laissez-le **Public** (plus simple pour commencer — l'image ne contient aucune donnée personnelle).
5. Cliquez **Create repository**.

Depuis votre ordinateur, dans le dossier du projet :

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/VOTRE_COMPTE/comptage-vehicules.git
git push -u origin main
```

GitHub Actions va immédiatement démarrer la construction de l'image. Vous pouvez suivre la progression dans l'onglet **Actions** de votre dépôt. La première construction dure environ 5–10 minutes.

---

## 3. Rendre l'image publique sur GHCR

Après le premier push, une image Docker a été publiée sur GitHub Container Registry.

1. Sur GitHub, cliquez sur votre avatar → **Your profile**.
2. Cliquez sur l'onglet **Packages**.
3. Cliquez sur `comptage-vehicules`.
4. À droite, cliquez **Package settings**.
5. Faites défiler jusqu'à **Danger Zone** → **Change visibility** → choisissez **Public**.

> **Pourquoi public ?** Cela permet à Watchtower de télécharger l'image sur le NAS sans avoir à gérer un token d'authentification. L'image ne contient que le code de l'application, aucune donnée personnelle.

---

## 4. Préparer le NAS

### 4.1 Créer les dossiers nécessaires

Dans **File Station** (interface web du NAS), créez :

```
/volume1/docker/comptage/
/volume1/docker/comptage/data/
```

### 4.2 Déposer les fichiers sur le NAS

Transférez depuis votre ordinateur vers `/volume1/docker/comptage/` :
- `docker-compose.yml`
- `config.example.yaml` (vous le renommerez en `config.yaml` à l'étape suivante)

Dans `docker-compose.yml`, remplacez `VOTRE_COMPTE_GITHUB` par votre nom d'utilisateur GitHub **en minuscules**.

---

## 5. Configurer l'application

1. Renommez `/volume1/docker/comptage/data/config.example.yaml` en `config.yaml`.
   (Ou copiez `config.example.yaml` dans `/volume1/docker/comptage/data/config.yaml`.)

2. Ouvrez `config.yaml` avec un éditeur de texte et remplissez les paramètres :

| Paramètre | Ce qu'il faut mettre |
|---|---|
| `video_folder` | Le chemin vers le dossier des vidéos de la sonnette (ex. `/volume1/surveillance/sonnette`) |
| `filename_datetime_format` | Le format du nom de vos fichiers vidéo (voir exemples dans le fichier) |
| `timezone` | Votre fuseau horaire (ex. `Europe/Paris`) |

> Les paramètres `counting.line_p1`, `counting.line_p2` et `counting.roi_polygon` seront définis à l'étape 6 (calibration). Laissez les valeurs par défaut pour l'instant.

---

## 6. Calibration : définir la ligne de comptage

La ligne de comptage est la ligne virtuelle sur l'image que les véhicules doivent franchir pour être comptés. Elle doit traverser la rue de gauche à droite (ou en diagonale selon l'angle de la caméra).

### 6.1 Démarrer l'application une première fois

Démarrez l'application (voir étape 7). Attendez 1–2 minutes, puis ouvrez dans votre navigateur :

```
http://[IP-de-votre-NAS]:8081
```

### 6.2 Utiliser l'outil de calibration

1. **Choisissez une vidéo** dans la liste déroulante (les vidéos les plus récentes apparaissent en premier).
2. **Extrayez une image** en cliquant sur "Extraire l'image". Ajustez le curseur "Seconde" pour choisir un moment où la rue est visible.
3. **Tracez la ligne de comptage** (mode "Ligne de comptage", actif par défaut) :
   - Cliquez sur le premier point (ex. bord gauche de la rue).
   - Cliquez sur le second point (ex. bord droit de la rue).
   - La ligne jaune apparaît sur l'image.
4. **Tracez la zone d'intérêt** (mode "Zone d'intérêt / ROI") — *optionnel mais recommandé* :
   - Cliquez plusieurs points pour entourer la rue (là où passent les voitures).
   - Double-cliquez pour fermer le polygone.
   - Cela évite que des mouvements hors de la rue (feuilles, passants sur le trottoir) ne déclenchent la détection.
5. Vérifiez les coordonnées affichées, puis cliquez **Enregistrer dans config.yaml**.

> **Conseil :** placez la ligne à un endroit où les véhicules passent clairement et où la rue n'est pas obstruée. Évitez les zones avec des ombres fixes ou des objets statiques.

### 6.3 Appliquer la calibration

Après l'enregistrement, redémarrez l'application pour appliquer la nouvelle configuration :

```bash
# Via SSH sur le NAS
cd /volume1/docker/comptage/
docker compose restart comptage
```

Ou depuis Container Manager : sélectionnez le conteneur → Action → Redémarrer.

---

## 7. Démarrer l'application

### Via Container Manager (interface graphique DSM)

1. Ouvrez **Container Manager** dans DSM.
2. Cliquez sur **Projet** → **Créer**.
3. Donnez un nom au projet (ex. `comptage`).
4. Dans "Chemin", pointez vers `/volume1/docker/comptage/`.
5. Cliquez **Suivant** → **Créer**.

### Via SSH (ligne de commande)

```bash
cd /volume1/docker/comptage/
docker compose up -d
```

Pour voir les logs en direct :

```bash
docker compose logs -f comptage
```

---

## 8. Accéder au tableau de bord

Ouvrez dans votre navigateur (depuis n'importe quel appareil du réseau local) :

```
http://[IP-de-votre-NAS]:8080
```

Le tableau de bord affiche :
- **Vue horaire** : nombre de passages par heure pour un jour donné, avec graphique et tableau détaillé.
- **Vue quotidienne** : comparaison des jours sur les 7, 14, 30 ou 90 derniers jours.
- **Filtres** : par type de véhicule (voiture, camion, bus, moto).
- **Indicateurs** : total du jour, heure de pointe, moyenne par heure active.

> Note : les données apparaissent au fur et à mesure que les vidéos sont traitées. La première fois, si beaucoup de vidéos sont en attente, le traitement peut prendre plusieurs heures.

---

## 9. Mises à jour automatiques

Le mécanisme est entièrement automatique :

1. Vous modifiez le code sur votre ordinateur.
2. Vous faites `git push`.
3. GitHub Actions construit la nouvelle image (5–10 min).
4. **Watchtower** (qui tourne sur le NAS) détecte la nouvelle image et redémarre l'application automatiquement — en 5 à 15 minutes.

**Rien à faire sur le NAS.** Votre configuration (`config.yaml`) et votre base de données ne sont jamais touchées par les mises à jour.

> **Exception :** si le fichier `docker-compose.yml` lui-même change (nouveau port, nouveau volume…), vous devrez le mettre à jour manuellement sur le NAS et relancer `docker compose up -d`.

---

## 10. Revenir à une version précédente

Si une mise à jour pose problème, vous pouvez revenir à une version antérieure.

### 10.1 Trouver le tag de la version précédente

Sur GitHub, allez dans votre dépôt → **Packages** → `comptage-vehicules`. Vous verrez la liste des tags publiés, par exemple :
- `latest` (version actuelle)
- `sha-a1b2c3d` (versions précédentes identifiées par le SHA du commit)

### 10.2 Revenir à une version précédente

Sur le NAS (via SSH) :

```bash
cd /volume1/docker/comptage/

# Modifier temporairement l'image dans docker-compose.yml
# Remplacez :latest par le tag souhaité, ex. :sha-a1b2c3d
docker compose down
docker compose pull
# Éditez docker-compose.yml pour mettre le bon tag
docker compose up -d
```

Pour éviter que Watchtower ne remette à jour automatiquement, stoppez-le temporairement :

```bash
docker compose stop watchtower
```

Relancez-le quand vous voulez reprendre les mises à jour automatiques :

```bash
docker compose start watchtower
```

---

## 11. Dépannage

### L'application ne démarre pas

```bash
docker compose logs comptage
```

Vérifiez que `config.yaml` existe dans `/volume1/docker/comptage/data/` et est correctement rempli.

### Aucune vidéo n'est traitée

- Vérifiez que le chemin `video_folder` dans `config.yaml` correspond exactement au chemin des vidéos sur le NAS.
- Vérifiez que le volume est bien monté : dans `docker-compose.yml`, le chemin à gauche du `:` doit être le bon chemin sur le NAS.
- Consultez les logs : `docker compose logs -f comptage`

### Le format du nom de fichier n'est pas reconnu

Le paramètre `filename_datetime_format` utilise les codes Python `strptime`. Copiez un vrai nom de fichier de votre sonnette et adaptez le format :

| Exemple de nom | Format correspondant |
|---|---|
| `Doorbell_20260515_140000.mp4` | `Doorbell_%Y%m%d_%H%M%S.mp4` |
| `record_2026-05-15_14-00-00.mp4` | `record_%Y-%m-%d_%H-%M-%S.mp4` |
| `cam1_20260515140000.mp4` | `cam1_%Y%m%d%H%M%S.mp4` |

### Les comptages semblent incorrects

- Ajustez la position de la ligne de comptage via l'outil de calibration (port 8081).
- Ajustez `detector.confidence_threshold` (augmenter pour moins de faux positifs, diminuer pour ne pas rater de véhicules).
- Ajustez `motion_filter.motion_threshold` si trop de segments vides sont analysés (ou pas assez).

### Watchtower ne met pas à jour l'application

- Vérifiez que l'image GHCR est bien **publique** (étape 3).
- Consultez les logs Watchtower : `docker compose logs watchtower`

---

## Structure du projet

```
comptage-vehicules/
├── src/
│   ├── main.py            # Orchestration principale
│   ├── config.py          # Chargement de la configuration
│   ├── database.py        # Base de données SQLite
│   ├── ingestion.py       # Surveillance du dossier vidéo
│   ├── motion_filter.py   # Filtre de mouvement (rapide)
│   ├── detector.py        # Détection IA + comptage
│   ├── dashboard.py       # Tableau de bord web
│   ├── calibration.py     # Outil de calibration
│   └── templates/         # Pages HTML
├── Dockerfile
├── docker-compose.yml
├── config.example.yaml
├── requirements.txt
├── .github/workflows/
│   └── build-and-publish.yml
└── .gitignore
```

---

## Précision attendue

Ce système fournit une **estimation fiable** des flux de véhicules, pas un comptage parfait. Des erreurs sont attendues dans certains cas :
- Deux véhicules qui se croisent en masquant l'un l'autre.
- Conditions de nuit ou de pluie forte (détection moins fiable).
- Véhicules très rapides ou très éloignés.

L'objectif est d'obtenir des **statistiques exploitables** : tendances horaires, comparaisons de jours, identification des heures de pointe. La précision peut être améliorée en ajustant la position de la ligne et les seuils de détection.
