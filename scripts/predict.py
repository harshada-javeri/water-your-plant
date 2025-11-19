from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400
        
        # Validate required fields
        required_fields = ['plant_type', 'soil_moisture', 'temperature', 'humidity', 'time_since_last_watering']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Validate data types
        try:
            data['soil_moisture'] = float(data['soil_moisture'])
            data['temperature'] = float(data['temperature'])
            data['humidity'] = float(data['humidity'])
            data['time_since_last_watering'] = float(data['time_since_last_watering'])
        except (ValueError, TypeError):
            return jsonify({"error": "Numeric fields must be valid numbers"}), 400
        
        # Validate ranges
        if not (0 <= data['soil_moisture'] <= 100):
            return jsonify({"error": "Soil moisture must be between 0-100%"}), 400
        if not (-10 <= data['temperature'] <= 50):
            return jsonify({"error": "Temperature must be between -10 and 50°C"}), 400
        if not (0 <= data['humidity'] <= 100):
            return jsonify({"error": "Humidity must be between 0-100%"}), 400
        if data['time_since_last_watering'] < 0:
            return jsonify({"error": "Time since last watering cannot be negative"}), 400
        
        df = pd.DataFrame([data])
        # Ensure all plant_type columns exist
        for pt in ['Aloe', 'Bonsai', 'Cactus', 'Fern', 'Lavender', 'Mint', 'Orchid', 'Rose', 'Succulent', 'Sunflower']:
            col = f'plant_type_{pt}'
            if col not in df.columns:
                df[col] = 0
        
        # Set the correct plant type to 1
        if 'plant_type' in data:
            plant_col = f'plant_type_{data["plant_type"]}'
            if plant_col in df.columns:
                df[plant_col] = 1
            else:
                return jsonify({"warning": f"Unknown plant type: {data['plant_type']}. Using default prediction."}), 200
        
        df = df.drop('plant_type', axis=1, errors='ignore')
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None
        
        response = {"needs_watering": int(prediction[0])}
        if probability is not None:
            response["watering_probability"] = float(probability)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    return jsonify({
        "model_type": type(model).__name__,
        "supported_plants": ['Aloe', 'Bonsai', 'Cactus', 'Fern', 'Lavender', 'Mint', 'Orchid', 'Rose', 'Succulent', 'Sunflower'],
        "input_features": ['plant_type', 'soil_moisture', 'temperature', 'humidity', 'time_since_last_watering'],
        "output_format": {"needs_watering": "0 or 1", "watering_probability": "0.0 to 1.0"}
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found. Available endpoints: /predict, /health, /info"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed. Use POST for /predict, GET for /health and /info"}), 405

if __name__ == '__main__':
    print("🌱 Plant Watering Prediction Service Starting...")
    print(f"Model loaded: {type(model).__name__}")
    print("Available endpoints:")
    print("  POST /predict - Make watering predictions")
    print("  GET  /health  - Health check")
    print("  GET  /info    - Model information")
    print("Server running on http://localhost:9696")
    app.run(host='0.0.0.0', port=9696)
