import io
import json
import logging
import os
import pickle
import pathlib
import urllib.request
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Imports pour l'API REST
from flask import Flask, request, jsonify, render_template_string

# Imports pour le traitement d'images
from PIL import Image

# Imports pour les prédictions Computer Vision
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# 4MB Max image size limit
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

# Variables globales pour les modèles
vision_predictor = None
aqi_model = None

# Chemins des modèles
VISION_MODEL_PATH = pathlib.Path('app/model.tflite')
VISION_LABELS_PATH = pathlib.Path('app/labels.txt')
AQI_MODEL_PATH = pathlib.Path('model/model.pkl')

# Classes pour le modèle Computer Vision (reprises de predict.py)
class VisionPredictor:
    def __init__(self, model_path, labels_path):
        logging.info(f"Chargement du modèle Computer Vision depuis {model_path}")
        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        assert len(input_details) == 1
        assert len(output_details) == 1
        self._input_index = input_details[0]['index']
        self._output_index = output_details[0]['index']

        input_size = int(input_details[0]['shape'][1])
        self._preprocessor = ImagePreprocessor(input_size, is_bgr=False)

        self._labels = [label.strip() for label in labels_path.read_text().splitlines()]
        logging.info(f"Labels du modèle: {self._labels}")

    @property
    def labels(self):
        return self._labels

    def predict(self, image: Image.Image):
        input_array = self._preprocessor.preprocess(image)
        input_array = input_array[np.newaxis, :, :, :]

        self._interpreter.set_tensor(self._input_index, input_array)
        self._interpreter.invoke()

        outputs = self._interpreter.get_tensor(self._output_index)
        assert len(outputs) == 1
        return outputs[0].tolist()

class ImagePreprocessor:
    def __init__(self, input_size: int, is_bgr: bool):
        self._input_size = input_size
        self._is_bgr = is_bgr

    def preprocess(self, image: Image.Image):
        image = self._update_orientation(image)
        image = self._resize_keep_aspect_ratio(image)
        image = self._crop_center(image)

        image = image.convert('RGB') if image.mode != 'RGB' else image
        np_array = np.array(image, dtype=np.float32)
        if self._is_bgr:
            np_array = np_array[:, :, (2, 1, 0)]
        return np_array

    def _update_orientation(self, image: Image.Image):
        exif_orientation_tag = 0x0112
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None and exif_orientation_tag in exif:
                orientation = exif.get(exif_orientation_tag, 1)
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation in [2, 3, 6, 7]:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation in [1, 2, 5, 6]:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _resize_keep_aspect_ratio(self, image: Image.Image):
        width, height = image.size
        aspect_ratio = width / height
        if width < height:
            new_width = self._input_size
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = self._input_size
            new_width = round(new_height * aspect_ratio)
        return image.resize((new_width, new_height), Image.BILINEAR)

    def _crop_center(self, image: Image.Image):
        width, height = image.size
        left = (width - self._input_size) // 2
        top = (height - self._input_size) // 2
        right = left + self._input_size
        bottom = top + self._input_size
        return image.crop((left, top, right, bottom))

def initialize_models():
    """Initialise les deux modèles"""
    global vision_predictor, aqi_model
    
    # Chargement du modèle Computer Vision
    if VISION_MODEL_PATH.exists() and VISION_LABELS_PATH.exists():
        vision_predictor = VisionPredictor(VISION_MODEL_PATH, VISION_LABELS_PATH)
        logging.info("Modèle Computer Vision chargé avec succès")
    else:
        logging.error("Fichiers du modèle Computer Vision non trouvés")
    
    # Chargement du modèle AQI
    if AQI_MODEL_PATH.exists():
        aqi_model = joblib.load(AQI_MODEL_PATH)
        logging.info("Modèle AQI chargé avec succès")
    else:
        logging.error("Modèle AQI non trouvé")

# Template HTML pour l'interface web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse de la Qualité de l'Air</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .tab-button {
            flex: 1;
            padding: 15px 20px;
            background: #fff;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            border-right: 1px solid #ddd;
        }
        
        .tab-button:last-child {
            border-right: none;
        }
        
        .tab-button.active {
            background: #4CAF50;
            color: white;
        }
        
        .tab-button:hover:not(.active) {
            background: #f5f5f5;
        }
        
        .tab-content {
            display: none;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #4CAF50;
            background: #f9f9f9;
        }
        
        .upload-area.dragover {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .form-col {
            flex: 1;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        
        input[type="number"], input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        input[type="number"]:focus, input[type="file"]:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .btn {
            background: #4CAF50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }
        
        .btn:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .result.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
        }
        
        .probability-bar {
            width: 200px;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin-left: 10px;
        }
        
        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #81C784);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .image-preview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            display: block;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .aqi-result {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            margin: 20px 0;
            border-radius: 15px;
            color: white;
        }
        
        .aqi-1 { background: linear-gradient(135deg, #4CAF50, #81C784); }
        .aqi-2 { background: linear-gradient(135deg, #FFC107, #FFD54F); }
        .aqi-3 { background: linear-gradient(135deg, #FF9800, #FFB74D); }
        .aqi-4 { background: linear-gradient(135deg, #F44336, #EF5350); }
        .aqi-5 { background: linear-gradient(135deg, #9C27B0, #BA68C8); }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Analyse de la Qualité de l'Air</h1>
        
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('vision')">📷 Analyse d'Image</button>
            <button class="tab-button" onclick="switchTab('metrics')">📊 Analyse de Métriques</button>
        </div>
        
        <!-- Onglet Computer Vision -->
        <div id="vision-tab" class="tab-content active">
            <h2>Analyse de la Qualité de l'Air par Image</h2>
            <p>Téléchargez une image du ciel pour analyser la qualité de l'air visuellement.</p>
            
            <form id="vision-form" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('imageFile').click()">
                    <div>
                        <h3>📸 Cliquez pour sélectionner une image</h3>
                        <p>ou glissez-déposez votre image ici</p>
                        <small>Formats supportés: JPG, PNG, GIF (max 4MB)</small>
                    </div>
                </div>
                <input type="file" id="imageFile" name="imageData" accept="image/*" style="display: none;" onchange="previewImage(this)">
                
                <img id="imagePreview" class="image-preview" style="display: none;">
                
                <button type="submit" class="btn" id="vision-btn">Analyser l'Image</button>
            </form>
            
            <div class="loading" id="vision-loading">
                <div class="spinner"></div>
                <p>Analyse en cours...</p>
            </div>
            
            <div id="vision-result" class="result"></div>
        </div>
        
        <!-- Onglet Métriques -->
        <div id="metrics-tab" class="tab-content">
            <h2>Prédiction AQI par Métriques</h2>
            <p>Entrez les valeurs des polluants pour obtenir une prédiction de l'indice de qualité de l'air (AQI).</p>
            
            <form id="metrics-form">
                <div class="form-row">
                    <div class="form-col">
                        <label for="co">CO (Monoxyde de carbone) μg/m³:</label>
                        <input type="number" id="co" name="co" step="0.01" required>
                    </div>
                    <div class="form-col">
                        <label for="no">NO (Monoxyde d'azote) μg/m³:</label>
                        <input type="number" id="no" name="no" step="0.01" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-col">
                        <label for="no2">NO₂ (Dioxyde d'azote) μg/m³:</label>
                        <input type="number" id="no2" name="no2" step="0.01" required>
                    </div>
                    <div class="form-col">
                        <label for="o3">O₃ (Ozone) μg/m³:</label>
                        <input type="number" id="o3" name="o3" step="0.01" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-col">
                        <label for="so2">SO₂ (Dioxyde de soufre) μg/m³:</label>
                        <input type="number" id="so2" name="so2" step="0.01" required>
                    </div>
                    <div class="form-col">
                        <label for="pm2_5">PM2.5 (Particules fines) μg/m³:</label>
                        <input type="number" id="pm2_5" name="pm2_5" step="0.01" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-col">
                        <label for="pm10">PM10 (Particules) μg/m³:</label>
                        <input type="number" id="pm10" name="pm10" step="0.01" required>
                    </div>
                    <div class="form-col">
                        <label for="nh3">NH₃ (Ammoniac) μg/m³:</label>
                        <input type="number" id="nh3" name="nh3" step="0.01" required>
                    </div>
                </div>
                
                <button type="submit" class="btn" id="metrics-btn">Prédire l'AQI</button>
            </form>
            
            <div class="loading" id="metrics-loading">
                <div class="spinner"></div>
                <p>Calcul en cours...</p>
            </div>
            
            <div id="metrics-result" class="result"></div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            // Cacher tous les onglets
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Afficher l'onglet sélectionné
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        function previewImage(input) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('imagePreview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        // Gestion du drag & drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('imageFile').files = files;
                previewImage(document.getElementById('imageFile'));
            }
        });
        
        // Soumission du formulaire Computer Vision
        document.getElementById('vision-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const imageFile = document.getElementById('imageFile').files[0];
            
            if (!imageFile) {
                showResult('vision-result', 'Veuillez sélectionner une image.', 'error');
                return;
            }
            
            formData.append('imageData', imageFile);
            
            showLoading('vision-loading', true);
            document.getElementById('vision-btn').disabled = true;
            
            try {
                const response = await fetch('/api/predict-image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayVisionResult(result);
                } else {
                    showResult('vision-result', result.error || 'Erreur lors de l\'analyse', 'error');
                }
            } catch (error) {
                showResult('vision-result', 'Erreur de connexion: ' + error.message, 'error');
            } finally {
                showLoading('vision-loading', false);
                document.getElementById('vision-btn').disabled = false;
            }
        });
        
        // Soumission du formulaire Métriques
        document.getElementById('metrics-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const metrics = Object.fromEntries(formData.entries());
            
            // Convertir en nombres
            Object.keys(metrics).forEach(key => {
                metrics[key] = parseFloat(metrics[key]);
            });
            
            showLoading('metrics-loading', true);
            document.getElementById('metrics-btn').disabled = true;
            
            try {
                const response = await fetch('/api/predict-aqi', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(metrics)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayAQIResult(result);
                } else {
                    showResult('metrics-result', result.error || 'Erreur lors de la prédiction', 'error');
                }
            } catch (error) {
                showResult('metrics-result', 'Erreur de connexion: ' + error.message, 'error');
            } finally {
                showLoading('metrics-loading', false);
                document.getElementById('metrics-btn').disabled = false;
            }
        });
        
        function showLoading(elementId, show) {
            document.getElementById(elementId).style.display = show ? 'block' : 'none';
        }
        
        function showResult(elementId, message, type) {
            const resultDiv = document.getElementById(elementId);
            resultDiv.innerHTML = message;
            resultDiv.className = `result ${type}`;
            resultDiv.style.display = 'block';
        }
        
        function displayVisionResult(result) {
            const predictions = result.predictions;
            let html = '<h3>Résultats de l\'analyse:</h3>';
            
            // Trier par probabilité décroissante
            predictions.sort((a, b) => b.probability - a.probability);
            
            predictions.forEach(pred => {
                const percentage = (pred.probability * 100).toFixed(2);
                html += `
                    <div class="prediction-item">
                        <span><strong>${pred.tagName}</strong>: ${percentage}%</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            
            showResult('vision-result', html, 'success');
        }
        
        function displayAQIResult(result) {
            const aqi = Math.round(result.aqi);
            const labels = {
                1: 'GOOD - Bonne qualité',
                2: 'MODERATE - Qualité modérée', 
                3: 'UNHEALTHY FOR SENSITIVE GROUPS - Mauvais pour groupes sensibles',
                4: 'UNHEALTHY - Mauvais',
                5: 'VERY UNHEALTHY - Très mauvais'
            };
            
            const label = labels[aqi] || 'Inconnu';
            
            const html = `
                <h3>Prédiction AQI:</h3>
                <div class="aqi-result aqi-${aqi}">
                    AQI: ${aqi}
                    <br>
                    <small>${label}</small>
                </div>
                <p><strong>Probabilité:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
            `;
            
            showResult('metrics-result', html, 'success');
        }
    </script>
</body>
</html>
"""

# Routes de l'application
@app.route('/')
def index():
    """Page principale avec l'interface web"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict-image', methods=['POST'])
def predict_image_api():
    """API pour la prédiction d'image"""
    try:
        if vision_predictor is None:
            return jsonify({'error': 'Modèle Computer Vision non disponible'}), 500
            
        imageData = None
        if 'imageData' in request.files:
            imageData = request.files['imageData']
        elif 'imageData' in request.form:
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())

        img = Image.open(imageData)
        outputs = vision_predictor.predict(img)
        
        predictions = [
            {
                'tagName': label, 
                'probability': round(p, 8), 
                'tagId': '', 
                'boundingBox': None
            } 
            for label, p in zip(vision_predictor.labels, outputs)
        ]
        
        response = {
            'id': '',
            'project': '',
            'iteration': '',
            'created': datetime.utcnow().isoformat(),
            'predictions': predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f'Erreur prédiction image: {str(e)}')
        return jsonify({'error': f'Erreur lors de l\'analyse: {str(e)}'}), 500

@app.route('/api/predict-aqi', methods=['POST'])
def predict_aqi_api():
    """API pour la prédiction AQI"""
    try:
        if aqi_model is None:
            return jsonify({'error': 'Modèle AQI non disponible'}), 500
            
        data = request.get_json()
        
        # Vérifier que toutes les métriques requises sont présentes
        required_metrics = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        for metric in required_metrics:
            if metric not in data:
                return jsonify({'error': f'Métrique manquante: {metric}'}), 400
        
        # Créer un DataFrame avec les données
        input_data = pd.DataFrame([data])
        
        # Faire la prédiction
        prediction = aqi_model.predict(input_data)[0]
        
        # Calculer la probabilité/confiance (si le modèle le supporte)
        try:
            probabilities = aqi_model.predict_proba(input_data)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.85  # Valeur par défaut
        
        response = {
            'aqi': int(prediction),
            'confidence': float(confidence),
            'metrics_used': data
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f'Erreur prédiction AQI: {str(e)}')
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialiser les modèles
    initialize_models()
    
    # Lancer le serveur
    app.run(host='0.0.0.0', port=8080, debug=True)