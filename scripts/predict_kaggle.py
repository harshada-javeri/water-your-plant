from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import re

app = Flask(__name__)

# Load models
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'model_kaggle.pkl')
encoder_path = os.path.join(script_dir, '..', 'label_encoder.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

def extract_features_from_input(data):
    """Extract features from input data similar to training"""
    
    # Default values if not provided
    defaults = {
        'temperature': 22,
        'humidity': 'medium',
        'light_exposure': 'medium',
        'watering_frequency': 7,
        'plant_type': 'Unknown Plant'
    }
    
    # Fill missing values
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    
    # Extract temperature
    if isinstance(data['temperature'], str):
        numbers = re.findall(r'\d+', data['temperature'])
        if numbers:
            temp_avg = np.mean([int(n) for n in numbers])
            temp_range = max([int(n) for n in numbers]) - min([int(n) for n in numbers]) if len(numbers) > 1 else 0
        else:
            temp_avg, temp_range = 22, 0
    else:
        temp_avg = float(data['temperature'])
        temp_range = 0
    
    # Extract humidity level
    humidity_str = str(data['humidity']).lower()
    if 'high' in humidity_str or humidity_str == '3':
        humidity_level = 3
    elif 'medium' in humidity_str or humidity_str == '2':
        humidity_level = 2
    elif 'low' in humidity_str or humidity_str == '1':
        humidity_level = 1
    else:
        try:
            humidity_level = min(3, max(1, int(float(humidity_str) / 33) + 1))
        except:
            humidity_level = 2
    
    # Extract light level
    light_str = str(data['light_exposure']).lower()
    if 'bright' in light_str and 'direct' in light_str:
        light_level = 4
    elif 'bright' in light_str:
        light_level = 3
    elif 'medium' in light_str:
        light_level = 2
    elif 'low' in light_str:
        light_level = 1
    else:
        try:
            light_level = min(4, max(1, int(float(light_str))))
        except:
            light_level = 2
    
    # Watering frequency
    if isinstance(data['watering_frequency'], str):
        numbers = re.findall(r'\d+', data['watering_frequency'])
        watering_days = int(numbers[0]) if numbers else 7
    else:
        watering_days = int(data['watering_frequency'])
    
    # Moisture requirement (simplified)
    moisture_requirement = 2  # default medium
    if 'soil_moisture' in data:
        moisture_str = str(data['soil_moisture']).lower()
        if 'dry' in moisture_str:
            moisture_requirement = 1
        elif 'moist' in moisture_str:
            moisture_requirement = 3
    
    # Plant type encoding
    plant_type = data['plant_type']
    try:
        plant_type_encoded = label_encoder.transform([plant_type])[0]
    except:
        # If plant type not in training data, use most common encoding
        plant_type_encoded = 0
    
    # Create feature vector
    features = {
        'temp_avg': temp_avg,
        'temp_range': temp_range,
        'humidity_level': humidity_level,
        'watering_frequency_days': watering_days,
        'light_level': light_level,
        'moisture_requirement': moisture_requirement,
        'plant_type_encoded': plant_type_encoded,
        'water_light_ratio': moisture_requirement / (light_level + 1),
        'temp_humidity_interaction': temp_avg * humidity_level,
        'watering_temp_ratio': watering_days / (temp_avg + 1)
    }
    
    return pd.DataFrame([features])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        feature_df = extract_features_from_input(data)
        
        # Make prediction
        if isinstance(model, tuple):  # Model with scaler (LR or SVM)
            model_obj, scaler = model
            features_scaled = scaler.transform(feature_df)
            prediction = model_obj.predict(features_scaled)[0]
            probability = model_obj.predict_proba(features_scaled)[0][1]
        else:  # Tree-based model
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
        
        return jsonify({
            "needs_watering": int(prediction),
            "watering_probability": float(probability),
            "features_used": feature_df.to_dict('records')[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Simplified prediction endpoint compatible with original format"""
    try:
        data = request.get_json()
        
        # Convert to new format
        converted_data = {
            'plant_type': data.get('plant_type', 'Unknown Plant'),
            'temperature': data.get('temperature', 22),
            'humidity': data.get('humidity', 50),
            'light_exposure': 'medium',
            'watering_frequency': 7
        }
        
        # Add soil moisture if provided
        if 'soil_moisture' in data:
            converted_data['soil_moisture'] = data['soil_moisture']
        
        feature_df = extract_features_from_input(converted_data)
        
        # Make prediction
        if isinstance(model, tuple):
            model_obj, scaler = model
            features_scaled = scaler.transform(feature_df)
            prediction = model_obj.predict(features_scaled)[0]
        else:
            prediction = model.predict(feature_df)[0]
        
        return jsonify({"needs_watering": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)