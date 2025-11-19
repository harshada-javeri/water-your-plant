# Plant Watering Reminder 🌱

A comprehensive machine learning project that predicts when indoor plants need watering based on environmental factors and plant characteristics. This project uses multiple ML models trained on real plant data to help gardeners maintain healthy plants.

## Project Description

Many gardeners and plant enthusiasts struggle to keep track of when their plants need watering, resulting in wilting or dead plants. This project aims to build a machine learning model that predicts when a plant needs watering based on environmental factors such as soil moisture, temperature, humidity, plant type, and time since last watering. The model is then deployed as a simple web service to allow easy integration with reminder systems.

## Features

- 🤖 **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM with hyperparameter tuning
- 📊 **Comprehensive EDA**: Correlation analysis, feature importance, multiple visualizations
- 🌿 **Real Plant Data**: 98+ indoor plant species with detailed care requirements
- 🔧 **Feature Engineering**: Advanced text processing and feature extraction
- 🐳 **Containerized**: Docker support for easy deployment
- ☁️ **Cloud Ready**: Heroku and Render deployment configurations
- 📈 **Model Comparison**: Automatic selection of best performing model

## Datasets

### Primary Dataset (Kaggle)
- **Source**: [Synthetic 100 Indoor Plant Dataset](https://www.kaggle.com/datasets/imalshap/synthetic-100-indoor-plant-dataset)
- **Size**: 98 plant species with 10+ features
- **Features**: Plant type, soil moisture requirements, watering frequency, temperature ranges, humidity levels, light exposure, pot size, soil type, fertilizer type

### Secondary Dataset (Original)
- **Size**: 10 plants with basic features
- **Features**: `plant_type`, `soil_moisture`, `temperature`, `humidity`, `time_since_last_watering`, `needs_watering`

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
1. Clone the repository:
```bash
git clone <repo-url>
cd water-your-plant
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. The datasets are already included in the `data/` directory.

## Usage

### Data Processing
Process the Kaggle dataset and create features:
```bash
python scripts/process_kaggle_data.py
```

### Model Training

#### Train with Original Dataset (Quick Test)
```bash
python scripts/train.py
```

#### Train with Kaggle Dataset (Full Training)
```bash
python scripts/train_kaggle.py
```

### Starting the Prediction Service

#### Original Model Service
```bash
python scripts/predict.py
```

#### Enhanced Kaggle Model Service
```bash
python scripts/predict_kaggle.py
```

The service will be available at `http://localhost:9696/predict`.

### Making Predictions

#### Simple Prediction (Original Format)
```bash
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{"plant_type": "Rose", "soil_moisture": 25, "temperature": 30, "humidity": 40, "time_since_last_watering": 48}'
```

#### Enhanced Prediction (Kaggle Model)
```bash
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{"plant_type": "Snake Plant", "temperature": "18-32°C", "humidity": "low", "light_exposure": "bright indirect", "watering_frequency": "2-4 weeks"}'
```

Response format:
```json
{
  "needs_watering": 1,
  "watering_probability": 0.85,
  "features_used": {...}
}
```

## Model Performance

The project trains multiple models and automatically selects the best performer:

- **Random Forest**: Tree-based ensemble with feature importance
- **Gradient Boosting**: Advanced boosting with hyperparameter tuning
- **Logistic Regression**: Linear model with regularization
- **Support Vector Machine**: Kernel-based classification

Models are evaluated using ROC-AUC score and cross-validation.

## Containerization

### Build and run with Docker:
```bash
# Build the Docker image
docker build -t plant-water-service .

# Run the container
docker run -p 9696:9696 plant-water-service
```

The service will be available at `http://localhost:9696/predict`

## Cloud Deployment

### Deploy to Render
1. Fork this repository
2. Connect your GitHub account to [Render](https://render.com)
3. Create a new Web Service
4. Select your forked repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python scripts/predict.py`
   - **Environment**: `Python 3`

### Deploy to Heroku
1. Install Heroku CLI
2. The `Procfile` is already included
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Project Structure

```
water-your-plant/
├── data/
│   ├── plant_watering.csv              # Original small dataset
│   ├── processed_plant_data.csv        # Processed Kaggle dataset
│   ├── 100 indoor plant dataset.xlsx   # Raw Kaggle dataset
│   └── LICENSE.txt                     # Dataset license
├── scripts/
│   ├── train.py                        # Original model training
│   ├── train_kaggle.py                 # Enhanced model training
│   ├── predict.py                      # Original prediction service
│   ├── predict_kaggle.py               # Enhanced prediction service
│   └── process_kaggle_data.py          # Data processing script
├── notebooks/
│   └── notebook.ipynb                  # EDA and model development
├── requirements.txt                    # Python dependencies
├── dockerfile                          # Container definition
├── Procfile                           # Heroku deployment config
├── model.pkl                          # Trained original model
├── model_kaggle.pkl                   # Trained Kaggle model
├── label_encoder.pkl                  # Plant type encoder
└── README.md                          # This file
```

## Testing

Run comprehensive project tests:
```bash
python test_complete_project.py
```

This will verify:
- ✅ Data processing and model training
- ✅ Model loading and predictions
- ✅ API endpoints functionality
- ✅ Docker configuration
- ✅ Requirements and dependencies
- ✅ Project structure completeness

## API Endpoints

### `/predict` (POST)
Main prediction endpoint supporting both simple and enhanced formats.

### `/predict/simple` (POST)
Simplified endpoint for backward compatibility.

### `/health` (GET)
Health check endpoint.

## Development

### Adding New Plant Types
1. Add plant data to the Kaggle dataset
2. Retrain the model: `python scripts/train_kaggle.py`
3. Update the prediction service

### Model Improvements
- Add more features (seasonal data, plant age, etc.)
- Experiment with deep learning models
- Implement ensemble methods
- Add time series forecasting

## License
This project is licensed under the MIT License.

## Acknowledgements
- ML Zoomcamp project guidelines
- [Kaggle Indoor Plant Dataset](https://www.kaggle.com/datasets/imalshap/synthetic-100-indoor-plant-dataset)
- Plant care community for domain expertise

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**🌱 Happy Gardening! Keep your plants healthy with ML! 🌱**