import io
import json
import logging

# Imports for the REST API
from flask import Flask, request, jsonify

# Imports for image procesing
from PIL import Image

# Imports for prediction
from predict import initialize, predict_image, predict_url
from predict_aqi import initialize_aqi_model, predict_aqi, validate_pollution_data

app = Flask(__name__)

# 4MB Max image size limit
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

# CORS headers pour permettre les requêtes depuis le navigateur
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Default route just shows simple text
@app.route('/')
def index():
    return '''
    <h1>CustomVision.ai model host harness</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><strong>POST /image</strong> - Classify air quality from sky images</li>
        <li><strong>POST /predict-aqi</strong> - Predict AQI from pollution data</li>
        <li><strong>POST /url</strong> - Classify air quality from image URL</li>
    </ul>
    '''

# NOUVELLE ROUTE: Prédiction AQI à partir de données de pollution
@app.route('/predict-aqi', methods=['POST'])
@app.route('/predict-aqi/', methods=['POST'])
def predict_aqi_handler():
    """
    Endpoint pour prédire l'AQI à partir de données de pollution
    Attend un JSON avec les champs: co, no, no2, o3, so2, pm2_5, pm10, nh3
    """
    try:
        # Vérifier le Content-Type
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'expected_format': {
                    'co': 'number',
                    'no': 'number', 
                    'no2': 'number',
                    'o3': 'number',
                    'so2': 'number',
                    'pm2_5': 'number',
                    'pm10': 'number',
                    'nh3': 'number'
                }
            }), 400

        pollution_data = request.get_json()
        
        if not pollution_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validation des données
        validation_errors = validate_pollution_data(pollution_data)
        if validation_errors:
            return jsonify({
                'error': 'Invalid input data',
                'validation_errors': validation_errors
            }), 400

        # Prédiction
        results = predict_aqi(pollution_data)
        
        # Log de la prédiction réussie
        app.logger.info(f"AQI prediction successful: AQI={results['predicted_aqi']}, Category={results['aqi_category']}")
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f'Error in AQI prediction: {str(e)}')
        return jsonify({
            'error': 'Error processing pollution data',
            'message': str(e)
        }), 500

# Route pour obtenir des informations sur les modèles
@app.route('/models/info', methods=['GET'])
def models_info():
    """Retourne des informations sur les modèles disponibles"""
    return jsonify({
        'models': {
            'image_classifier': {
                'description': 'Classifies air quality from sky images',
                'input': 'Image file (jpg, png, etc.)',
                'output': 'Air quality classification (GOOD, MODERATE, etc.)',
                'endpoint': '/image'
            },
            'aqi_predictor': {
                'description': 'Predicts AQI from pollution measurements',
                'input': 'JSON with pollution data (co, no, no2, o3, so2, pm2_5, pm10, nh3)',
                'output': 'Numerical AQI value and category',
                'endpoint': '/predict-aqi'
            }
        },
        'pollution_parameters': {
            'co': 'Carbon monoxide (mg/m³)',
            'no': 'Nitric oxide (µg/m³)', 
            'no2': 'Nitrogen dioxide (µg/m³)',
            'o3': 'Ozone (µg/m³)',
            'so2': 'Sulfur dioxide (µg/m³)',
            'pm2_5': 'Particulate matter ≤ 2.5µm (µg/m³)',
            'pm10': 'Particulate matter ≤ 10µm (µg/m³)',
            'nh3': 'Ammonia (µg/m³)'
        }
    })

# Like the CustomVision.ai Prediction service /image route handles either
#     - octet-stream image file
#     - a multipart/form-data with files in the imageData parameter
@app.route('/image', methods=['POST'])
@app.route('/<project>/image', methods=['POST'])
@app.route('/<project>/image/nostore', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/image', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/image/nostore', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/image', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/image/nostore', methods=['POST'])
def predict_image_handler(project=None, publishedName=None):
    try:
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        elif ('imageData' in request.form):
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())

        img = Image.open(imageData)
        results = predict_image(img)
        return jsonify(results)
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


# Like the CustomVision.ai Prediction service /url route handles url's
# in the body of hte request of the form:
#     { 'Url': '<http url>'}
@app.route('/url', methods=['POST'])
@app.route('/<project>/url', methods=['POST'])
@app.route('/<project>/url/nostore', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/url', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/url/nostore', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/url', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/url/nostore', methods=['POST'])
def predict_url_handler(project=None, publishedName=None):
    try:
        image_url = json.loads(request.get_data().decode('utf-8'))['url']
        results = predict_url(image_url)
        return jsonify(results)
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Load and intialize the image classification model
    initialize()
    
    # Load and initialize the AQI prediction model
    aqi_model_loaded = initialize_aqi_model()
    if not aqi_model_loaded:
        logging.warning("AQI model could not be loaded. AQI prediction will not be available.")
    else:
        logging.info("AQI model loaded successfully")

    # Run the server
    app.run(host='0.0.0.0', port=80)