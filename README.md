# water-your-plant
The Plant Watering Reminder is a machine learning project designed to help gardeners and plant enthusiasts remember when to water their plants.# Plant Watering Reminder

## Project Description
Many gardeners and plant lovers struggle to keep track of when their plants need watering, resulting in wilting or dead plants. This project aims to build a machine learning model that predicts when a plant needs watering based on environmental factors such as soil moisture, temperature, humidity, plant type, and time since last watering. The model is then deployed as a simple web service to allow easy integration with reminder systems.

## Data
The dataset used includes features:
- `plant_type`: Type of plant (e.g., Succulent, Fern, Rose)
- `soil_moisture`: Soil moisture percentage
- `temperature`: Ambient temperature in Celsius
- `humidity`: Ambient humidity percentage
- `time_since_last_watering`: Hours since the plant was last watered
- `needs_watering`: Target label (0 = no, 1 = yes)

Sample dataset can be downloaded from [Kaggle Plant Water Prediction](https://www.kaggle.com/datasets/siddharthss/plant-water-prediction) or use the provided sample CSV `plant_watering.csv`.

## Installation
1. Clone the repository:
git clone <repo-url>
cd plant-watering-reminder

2. Install dependencies:

pip install -r requirements.txt

3. Download or place the dataset `plant_watering.csv` in the project directory.

## Usage

### Training the model
Run the training script to train and save the model:

python train.py

### Starting the prediction service
Run the Flask app for serving predictions:

python predict.py


The service will be available locally at `http://localhost:9696/predict`.

### Making a prediction
Send a POST request with JSON payload specifying plant data. Example using `curl`:

curl -X POST http://localhost:9696/predict
-H "Content-Type: application/json"
-d '{"plant_type": "Rose", "soil_moisture": 25, "temperature": 30, "humidity": 40, "time_since_last_watering": 48}'


The response will be:
{"needs_watering": 1}


## Deployment

A `Dockerfile` is included for containerizing the prediction service:

docker build -t plant-water-service .
docker run -p 9696:9696 plant-water-service


You can deploy this container to cloud platforms like Heroku, Render, or any Kubernetes cluster.

## Files

- `notebook.ipynb`: Jupyter notebook with EDA, training, and model evaluation
- `train.py`: Script to train and save the ML model
- `predict.py`: Flask-based web server for model inference
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container definition for the app
- `plant_watering.csv`: Sample dataset (or instructions to download)

## License
This project is licensed under the MIT License.

## Acknowledgements
ML Zoomcamp project guidelines.


