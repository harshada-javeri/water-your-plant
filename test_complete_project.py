#!/usr/bin/env python3
"""
Complete project test script for the enhanced water-your-plant project
"""

import os
import pandas as pd
import pickle
import requests
import time
import subprocess
import signal
import sys

def test_data_processing():
    """Test data processing and model training"""
    print("=== Testing Data Processing ===")
    
    # Check if processed data exists
    processed_data_path = 'data/processed_plant_data.csv'
    if os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
        print(f"✅ Processed Kaggle dataset: {df.shape[0]} plants, {df.shape[1]} features")
        print(f"✅ Target distribution: {df['needs_watering'].value_counts().to_dict()}")
    else:
        print("❌ Processed dataset not found")
    
    # Check original small dataset
    original_data_path = 'data/plant_watering.csv'
    if os.path.exists(original_data_path):
        df_orig = pd.read_csv(original_data_path)
        print(f"✅ Original dataset: {df_orig.shape[0]} plants, {df_orig.shape[1]} features")
    else:
        print("❌ Original dataset not found")

def test_models():
    """Test trained models"""
    print("\n=== Testing Models ===")
    
    # Test original model
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Original model loaded: {type(model).__name__}")
    else:
        print("❌ Original model not found")
    
    # Test Kaggle model
    if os.path.exists('model_kaggle.pkl'):
        with open('model_kaggle.pkl', 'rb') as f:
            model_kaggle = pickle.load(f)
        print(f"✅ Kaggle model loaded: {type(model_kaggle).__name__}")
    else:
        print("❌ Kaggle model not found")
    
    # Test label encoder
    if os.path.exists('label_encoder.pkl'):
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ Label encoder loaded: {len(encoder.classes_)} plant types")
    else:
        print("❌ Label encoder not found")

def test_api(port=9696):
    """Test API endpoints"""
    print(f"\n=== Testing API on port {port} ===")
    
    base_url = f"http://localhost:{port}"
    
    # Test original prediction endpoint
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={
                "plant_type": "Rose",
                "soil_moisture": 25,
                "temperature": 30,
                "humidity": 40,
                "time_since_last_watering": 48
            },
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Original API: {result}")
        else:
            print(f"❌ Original API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Original API error: {e}")

def test_docker():
    """Test Docker setup"""
    print("\n=== Testing Docker Setup ===")
    
    if os.path.exists('dockerfile'):
        print("✅ Dockerfile exists")
        with open('dockerfile', 'r') as f:
            content = f.read()
            if 'python scripts/predict.py' in content:
                print("✅ Dockerfile has correct CMD")
            else:
                print("❌ Dockerfile CMD might be incorrect")
    else:
        print("❌ Dockerfile not found")
    
    if os.path.exists('Procfile'):
        print("✅ Procfile exists for Heroku deployment")
    else:
        print("❌ Procfile not found")

def test_requirements():
    """Test requirements and dependencies"""
    print("\n=== Testing Requirements ===")
    
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'flask', 'pandas', 'scikit-learn', 
            'matplotlib', 'seaborn', 'numpy', 'openpyxl'
        ]
        
        missing = []
        for package in required_packages:
            if package not in requirements:
                missing.append(package)
        
        if not missing:
            print("✅ All required packages in requirements.txt")
        else:
            print(f"❌ Missing packages: {missing}")
    else:
        print("❌ requirements.txt not found")

def test_project_structure():
    """Test overall project structure"""
    print("\n=== Testing Project Structure ===")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'dockerfile',
        'Procfile',
        'data/plant_watering.csv',
        'scripts/train.py',
        'scripts/predict.py',
        'notebooks/notebook.ipynb'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def calculate_score():
    """Calculate evaluation score based on criteria"""
    print("\n=== Calculating Evaluation Score ===")
    
    score = 0
    max_score = 16
    
    # Problem description (2 points)
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme = f.read()
        if len(readme) > 500 and 'plant watering' in readme.lower():
            score += 2
            print("✅ Problem description: 2/2 points")
        else:
            score += 1
            print("⚠️ Problem description: 1/2 points")
    
    # EDA (2 points)
    if os.path.exists('notebooks/notebook.ipynb'):
        with open('notebooks/notebook.ipynb', 'r') as f:
            notebook = f.read()
        if all(term in notebook for term in ['correlation', 'GridSearchCV', 'feature_importance']):
            score += 2
            print("✅ EDA: 2/2 points")
        else:
            score += 1
            print("⚠️ EDA: 1/2 points")
    
    # Model training (3 points)
    if os.path.exists('scripts/train.py'):
        with open('scripts/train.py', 'r') as f:
            train_script = f.read()
        if all(term in train_script for term in ['RandomForestClassifier', 'LogisticRegression', 'GridSearchCV']):
            score += 3
            print("✅ Model training: 3/3 points")
        else:
            score += 1
            print("⚠️ Model training: 1/3 points")
    
    # Exporting notebook to script (1 point)
    if os.path.exists('scripts/train.py'):
        score += 1
        print("✅ Exporting notebook to script: 1/1 point")
    
    # Reproducibility (1 point)
    if os.path.exists('data/plant_watering.csv') and os.path.exists('scripts/train.py'):
        score += 1
        print("✅ Reproducibility: 1/1 point")
    
    # Model deployment (1 point)
    if os.path.exists('scripts/predict.py'):
        score += 1
        print("✅ Model deployment: 1/1 point")
    
    # Dependency management (2 points)
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            req = f.read()
        if 'flask' in req and 'pandas' in req and 'scikit-learn' in req:
            if os.path.exists('README.md'):
                with open('README.md', 'r') as f:
                    readme = f.read()
                if 'venv' in readme or 'virtual environment' in readme:
                    score += 2
                    print("✅ Dependency management: 2/2 points")
                else:
                    score += 1
                    print("⚠️ Dependency management: 1/2 points")
    
    # Containerization (2 points)
    if os.path.exists('dockerfile'):
        score += 1
        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                readme = f.read()
            if 'docker build' in readme and 'docker run' in readme:
                score += 1
                print("✅ Containerization: 2/2 points")
            else:
                print("⚠️ Containerization: 1/2 points")
    
    # Cloud deployment (2 points)
    if os.path.exists('README.md') and os.path.exists('Procfile'):
        with open('README.md', 'r') as f:
            readme = f.read()
        if 'heroku' in readme.lower() or 'render' in readme.lower():
            score += 2
            print("✅ Cloud deployment: 2/2 points")
    
    print(f"\n🎯 TOTAL SCORE: {score}/{max_score} points")
    
    if score >= 14:
        print("🏆 EXCELLENT! Your project meets all criteria for maximum score!")
    elif score >= 12:
        print("🥈 GREAT! Your project is very strong with minor improvements needed.")
    elif score >= 10:
        print("🥉 GOOD! Your project meets most criteria with some improvements needed.")
    else:
        print("📚 Your project needs significant improvements to meet the evaluation criteria.")
    
    return score

def main():
    print("🌱 WATER-YOUR-PLANT PROJECT COMPREHENSIVE TEST 🌱")
    print("=" * 60)
    
    # Change to project directory
    if not os.path.exists('data'):
        print("❌ Please run this script from the project root directory")
        return
    
    # Run all tests
    test_project_structure()
    test_data_processing()
    test_models()
    test_requirements()
    test_docker()
    
    # Calculate final score
    final_score = calculate_score()
    
    print("\n" + "=" * 60)
    print("🎉 TEST COMPLETE!")
    print(f"📊 Final Score: {final_score}/16 points")
    
    if final_score >= 14:
        print("\n✨ Your project is ready for submission! ✨")
        print("\nNext steps:")
        print("1. git add . && git commit -m 'Complete ML project'")
        print("2. git push origin main")
        print("3. Deploy to cloud platform of choice")
    else:
        print(f"\n🔧 Consider implementing the missing features to improve your score.")

if __name__ == "__main__":
    main()