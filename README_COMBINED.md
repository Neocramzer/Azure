# 🌍 Application Combinée d'Analyse de la Qualité de l'Air

Cette application combine deux modèles d'intelligence artificielle pour analyser la qualité de l'air :

1. **Computer Vision** : Analyse la qualité de l'air à partir d'images du ciel
2. **Classification de Métriques** : Prédit l'indice AQI à partir des mesures de polluants

## 🚀 Démarrage Rapide avec Docker

### 1. Construire l'image Docker

```bash
docker build -t air-quality-app -f Dockerfile.combined .
```

### 2. Lancer le container

```bash
docker run -p 8080:8080 air-quality-app
```

### 3. Accéder à l'application

Ouvrez votre navigateur et allez sur : **http://localhost:8080**

## 📋 Installation Locale (sans Docker)

### Prérequis
- Python 3.9+
- pip

### Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app_combined.py
```

## 🎯 Utilisation

### Onglet 1 : Analyse d'Image 📷

1. Cliquez sur "Analyse d'Image" 
2. Sélectionnez ou glissez-déposez une image du ciel
3. Cliquez sur "Analyser l'Image"
4. Obtenez les probabilités pour chaque niveau de qualité d'air :
   - **GOOD** : Bonne qualité
   - **MODERATE** : Qualité modérée
   - **UNHEALTHY_SENSITIVE_GROUP** : Mauvais pour les groupes sensibles
   - **UNHEALTHY** : Mauvais
   - **SEVERE** : Sévère
   - **VERY_UNHEALTHY** : Très mauvais

### Onglet 2 : Analyse de Métriques 📊

1. Cliquez sur "Analyse de Métriques"
2. Entrez les valeurs des polluants (en μg/m³) :
   - **CO** : Monoxyde de carbone
   - **NO** : Monoxyde d'azote
   - **NO₂** : Dioxyde d'azote
   - **O₃** : Ozone
   - **SO₂** : Dioxyde de soufre
   - **PM2.5** : Particules fines
   - **PM10** : Particules
   - **NH₃** : Ammoniac
3. Cliquez sur "Prédire l'AQI"
4. Obtenez l'indice AQI prédit (1-5) avec sa signification

## 📊 Interprétation des Résultats

### Indices AQI
- **1** : GOOD - Bonne qualité d'air
- **2** : MODERATE - Qualité modérée
- **3** : UNHEALTHY FOR SENSITIVE GROUPS - Mauvais pour les groupes sensibles
- **4** : UNHEALTHY - Mauvais pour la santé
- **5** : VERY UNHEALTHY - Très mauvais pour la santé

## 🔧 API Endpoints

L'application expose également des APIs REST :

### Prédiction par Image
```bash
curl -X POST \
  -F "imageData=@image.jpg" \
  http://localhost:8080/api/predict-image
```

### Prédiction par Métriques
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "co": 520.71,
    "no": 2.38,
    "no2": 16.28,
    "o3": 130.18,
    "so2": 47.68,
    "pm2_5": 65.96,
    "pm10": 72.13,
    "nh3": 8.36
  }' \
  http://localhost:8080/api/predict-aqi
```

## 📁 Structure du Projet

```
.
├── app_combined.py          # Application Flask combinée
├── Dockerfile.combined      # Dockerfile pour l'app combinée
├── requirements.txt         # Dépendances Python
├── app/                     # Modèle Computer Vision
│   ├── model.tflite
│   ├── labels.txt
│   └── ...
├── model/                   # Modèle de classification AQI
│   ├── model.pkl
│   └── ...
└── image/                   # Images de test
    └── image.jpg
```

## 🛠️ Technologies Utilisées

- **Backend** : Flask, Python 3.9
- **Computer Vision** : TensorFlow Lite
- **Machine Learning** : scikit-learn, joblib
- **Frontend** : HTML5, CSS3, JavaScript (Vanilla)
- **Containerisation** : Docker

## 🐛 Dépannage

### Erreur "Modèle non trouvé"
Vérifiez que les fichiers suivants existent :
- `app/model.tflite`
- `app/labels.txt`
- `model/model.pkl`

### Erreur de port
Si le port 8080 est occupé, changez le port :
```bash
docker run -p 8081:8080 air-quality-app
```

### Problème de mémoire
Si vous manquez de mémoire, ajustez les limites Docker :
```bash
docker run --memory="2g" -p 8080:8080 air-quality-app
```

## 📝 Notes

- Taille maximale des images : 4MB
- Formats d'images supportés : JPG, PNG, GIF
- L'application utilise les modèles pré-entraînés fournis
- Interface responsive compatible mobile et desktop

## 🔄 Arrêter l'Application

```bash
# Arrêter tous les containers en cours
docker stop $(docker ps -q)

# Ou arrêter un container spécifique
docker stop <container_id>
```