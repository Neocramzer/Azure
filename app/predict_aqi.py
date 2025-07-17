import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

logger = logging.getLogger(__name__)
aqi_model = None
scaler = None
MODEL_PATH = Path('aqi_model/simple_model.pkl')

# Mapping des valeurs AQI vers les catégories
AQI_CATEGORIES = {
    (0, 50): "GOOD",
    (51, 100): "MODERATE", 
    (101, 150): "UNHEALTHY_SENSITIVE_GROUP",
    (151, 200): "UNHEALTHY",
    (201, 300): "VERY_UNHEALTHY",
    (301, float('inf')): "SEVERE"
}

def get_aqi_category(aqi_value):
    """Convertit une valeur AQI numérique en catégorie"""
    for (min_val, max_val), category in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return category
    return "UNKNOWN"

def train_simple_model():
    """Entraîne un modèle simple avec le dataset CSV"""
    logger.info("Training simple AQI model from dataset...")
    
    # Lire le dataset depuis le fichier CSV
    csv_path = Path('air_pollution_data.csv')
    if not csv_path.exists():
        logger.error("Dataset CSV not found")
        return False
    
    try:
        # Charger les données
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Préparer les features et target
        feature_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        X = df[feature_cols].copy()
        
        # Pour ce dataset, l'AQI semble être dans l'intervalle 1-5
        # Convertissons-le vers une échelle 0-300 plus réaliste
        # En analysant les données PM2.5, on peut faire une estimation
        def estimate_realistic_aqi(row):
            pm25 = row['pm2_5']
            pm10 = row['pm10']
            no2 = row['no2']
            o3 = row['o3']
            
            # Formule simplifiée basée sur les principaux polluants
            aqi_pm25 = min(300, max(0, pm25 * 2))  # PM2.5 contribue beaucoup
            aqi_pm10 = min(300, max(0, pm10 * 1.5))
            aqi_no2 = min(300, max(0, no2 * 2))
            aqi_o3 = min(300, max(0, o3 * 1.2))
            
            # Prendre le maximum (comme dans le vrai calcul AQI)
            return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3)
        
        y = df.apply(estimate_realistic_aqi, axis=1)
        
        # Nettoyer les données
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Training with {len(X)} clean samples")
        logger.info(f"AQI range: {y.min():.1f} - {y.max():.1f}")
        
        # Diviser en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normaliser les features
        global scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraîner le modèle
        global aqi_model
        aqi_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5
        )
        
        aqi_model.fit(X_train_scaled, y_train)
        
        # Évaluer le modèle
        train_score = aqi_model.score(X_train_scaled, y_train)
        test_score = aqi_model.score(X_test_scaled, y_test)
        
        logger.info(f"Model training completed")
        logger.info(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Sauvegarder le modèle
        model_data = {
            'model': aqi_model,
            'scaler': scaler,
            'feature_names': feature_cols
        }
        
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def initialize_aqi_model():
    """Initialise le modèle AQI"""
    global aqi_model, scaler
    
    try:
        # Essayer de charger le modèle simple d'abord
        if MODEL_PATH.exists():
            logger.info(f"Loading simple AQI model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            aqi_model = model_data['model']
            scaler = model_data['scaler']
            
            logger.info("Simple AQI model loaded successfully")
            return True
        
        # Si pas de modèle simple, essayer d'entraîner
        logger.info("No simple model found, training new model...")
        if train_simple_model():
            logger.info("Model training successful")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error initializing AQI model: {str(e)}")
        return False

def predict_aqi(pollution_data):
    """
    Prédit l'AQI à partir des données de pollution
    """
    global aqi_model, scaler
    
    if aqi_model is None:
        raise Exception("AQI model not initialized. Call initialize_aqi_model() first.")
    
    try:
        # Validation des données d'entrée
        required_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        
        for feature in required_features:
            if feature not in pollution_data:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Créer DataFrame avec l'ordre correct des colonnes
        data_dict = {feature: [float(pollution_data[feature])] for feature in required_features}
        df = pd.DataFrame(data_dict)
        
        # Prédiction avec normalisation
        X_scaled = scaler.transform(df)
        prediction = aqi_model.predict(X_scaled)
        
        aqi_value = float(prediction[0])
        
        # S'assurer que l'AQI est dans une plage raisonnable
        aqi_value = max(0, min(500, aqi_value))
        
        # Obtenir la catégorie AQI
        aqi_category = get_aqi_category(aqi_value)
        
        result = {
            'predicted_aqi': round(aqi_value, 2),
            'aqi_category': aqi_category,
            'aqi_rounded': int(round(aqi_value)),
            'input_values': pollution_data,
            'confidence': None,
            'model_type': 'Simple Random Forest Model'
        }
        
        logger.info(f"AQI prediction successful: {aqi_value} ({aqi_category})")
        return result
        
    except Exception as e:
        logger.error(f"Error in AQI prediction: {str(e)}")
        raise

def validate_pollution_data(data):
    """Valide les données de pollution"""
    required_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    errors = []
    
    for feature in required_features:
        if feature not in data:
            errors.append(f"Missing feature: {feature}")
        else:
            try:
                value = float(data[feature])
                if value < 0:
                    errors.append(f"Negative value for {feature}: {value}")
                # Validation des plages raisonnables
                if feature == 'co' and value > 10000:
                    errors.append(f"CO value seems too high: {value}")
                elif feature in ['pm2_5', 'pm10'] and value > 1000:
                    errors.append(f"{feature} value seems too high: {value}")
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {feature}: {data[feature]}")
    
    return errors
