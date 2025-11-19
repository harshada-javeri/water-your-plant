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
    data = request.get_json()
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
    df = df.drop('plant_type', axis=1, errors='ignore')
    prediction = model.predict(df)
    return jsonify({"needs_watering": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)
