#!/usr/bin/env python3
"""
Script de test pour valider le bon fonctionnement des deux modèles
Usage: python test_api.py
"""

import requests
import json
import sys
import time

API_BASE_URL = "http://localhost:8080"

def test_server_health():
    """Test si le serveur répond"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ Serveur accessible")
            return True
        else:
            print(f"❌ Serveur non accessible (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au serveur")
        print("   Vérifiez que le container Docker est en cours d'exécution sur le port 8080")
        return False

def test_models_info():
    """Test de l'endpoint d'informations sur les modèles"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint /models/info fonctionne")
            print(f"   Modèles disponibles: {', '.join(data['models'].keys())}")
            return True
        else:
            print(f"❌ Erreur /models/info (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du test /models/info: {e}")
        return False

def test_aqi_prediction():
    """Test de l'endpoint de prédiction AQI"""
    # Données de test réalistes
    test_data = {
        "co": 1.2,
        "no": 15.5,
        "no2": 25.3,
        "o3": 45.2,
        "so2": 8.1,
        "pm2_5": 35.7,
        "pm10": 55.2,
        "nh3": 12.4
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-aqi",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prédiction AQI fonctionne")
            print(f"   AQI prédit: {data['predicted_aqi']}")
            print(f"   Catégorie: {data['aqi_category']}")
            if 'confidence' in data and data['confidence']:
                print(f"   Confiance: {data['confidence']:.2%}")
            return True
        else:
            print(f"❌ Erreur prédiction AQI (status: {response.status_code})")
            try:
                error_data = response.json()
                print(f"   Détails: {error_data}")
            except:
                print(f"   Réponse: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du test AQI: {e}")
        return False

def test_aqi_validation():
    """Test de la validation des données AQI"""
    # Test avec données manquantes
    invalid_data = {
        "co": 1.2,
        "no": 15.5
        # Données manquantes volontairement
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-aqi",
            headers={"Content-Type": "application/json"},
            json=invalid_data
        )
        
        if response.status_code == 400:
            print("✅ Validation des données fonctionne (rejet données invalides)")
            return True
        else:
            print(f"❌ Validation échoue - devrait rejeter les données invalides")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du test de validation: {e}")
        return False

def test_image_endpoint():
    """Test basique de l'endpoint image (sans fichier réel)"""
    try:
        # Test avec données vides pour vérifier que l'endpoint existe
        response = requests.post(f"{API_BASE_URL}/image")
        
        # On s'attend à une erreur mais pas une 404
        if response.status_code != 404:
            print("✅ Endpoint /image existe et répond")
            return True
        else:
            print("❌ Endpoint /image non trouvé")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du test image: {e}")
        return False

def run_all_tests():
    """Lance tous les tests"""
    print("🔍 Test du système dual de prédiction AQI")
    print("=" * 50)
    
    tests = [
        ("Santé du serveur", test_server_health),
        ("Informations modèles", test_models_info),
        ("Prédiction AQI", test_aqi_prediction),
        ("Validation AQI", test_aqi_validation),
        ("Endpoint Images", test_image_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)  # Pause entre les tests
    
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"{emoji} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Le système est opérationnel.")
        return True
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez la configuration.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)