import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import re

def extract_numeric_features(df):
    """Extract numeric features from text-based columns"""
    
    # Extract temperature ranges
    def extract_temp_range(temp_str):
        if pd.isna(temp_str):
            return 22, 22  # default
        # Extract numbers from temperature string
        numbers = re.findall(r'\d+', str(temp_str))
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            temp = int(numbers[0])
            return temp, temp
        return 22, 22  # default
    
    # Extract humidity levels
    def extract_humidity_level(humidity_str):
        if pd.isna(humidity_str):
            return 2  # medium
        humidity_str = str(humidity_str).lower()
        if 'high' in humidity_str:
            return 3
        elif 'medium' in humidity_str:
            return 2
        elif 'low' in humidity_str:
            return 1
        return 2  # default
    
    # Extract watering frequency (days)
    def extract_watering_days(watering_str):
        if pd.isna(watering_str):
            return 7  # default weekly
        watering_str = str(watering_str).lower()
        numbers = re.findall(r'\d+', watering_str)
        if numbers:
            if 'week' in watering_str:
                return int(numbers[0]) * 7
            elif 'day' in watering_str:
                return int(numbers[0])
        return 7  # default
    
    # Extract light requirements
    def extract_light_level(light_str):
        if pd.isna(light_str):
            return 2  # medium
        light_str = str(light_str).lower()
        if 'bright' in light_str and 'direct' in light_str:
            return 4
        elif 'bright' in light_str:
            return 3
        elif 'medium' in light_str:
            return 2
        elif 'low' in light_str:
            return 1
        return 2  # default
    
    # Apply feature extraction
    df['temp_min'], df['temp_max'] = zip(*df['Temperature'].apply(extract_temp_range))
    df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    df['humidity_level'] = df['Humidity'].apply(extract_humidity_level)
    df['watering_frequency_days'] = df['Watering Date Range'].apply(extract_watering_days)
    df['light_level'] = df['Light Exposure'].apply(extract_light_level)
    
    # Create moisture requirement categories
    def categorize_moisture(moisture_str):
        if pd.isna(moisture_str):
            return 2
        moisture_str = str(moisture_str).lower()
        if 'dry' in moisture_str and 'completely' in moisture_str:
            return 1  # low water needs
        elif 'moist' in moisture_str and 'consistently' in moisture_str:
            return 3  # high water needs
        elif 'moist' in moisture_str:
            return 2  # medium water needs
        return 2
    
    df['moisture_requirement'] = df['Soil Moisture'].apply(categorize_moisture)
    
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    
    # Extract numeric features
    df = extract_numeric_features(df)
    
    # Select relevant features
    feature_columns = [
        'temp_avg', 'temp_range', 'humidity_level', 
        'watering_frequency_days', 'light_level', 'moisture_requirement'
    ]
    
    # Add plant type encoding
    le = LabelEncoder()
    df['plant_type_encoded'] = le.fit_transform(df['Name of Indoor Plant'])
    feature_columns.append('plant_type_encoded')
    
    # Create additional engineered features
    df['water_light_ratio'] = df['moisture_requirement'] / (df['light_level'] + 1)
    df['temp_humidity_interaction'] = df['temp_avg'] * df['humidity_level']
    df['watering_temp_ratio'] = df['watering_frequency_days'] / (df['temp_avg'] + 1)
    
    feature_columns.extend(['water_light_ratio', 'temp_humidity_interaction', 'watering_temp_ratio'])
    
    return df[feature_columns + ['needs_watering']], le

def train_models(X, y):
    """Train multiple models and return the best one"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    rf_pred = rf_grid.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf_grid.predict_proba(X_test)[:, 1])
    models['Random Forest'] = rf_grid.best_estimator_
    results['Random Forest'] = rf_auc
    
    print(f"Random Forest - Best params: {rf_grid.best_params_}")
    print(f"Random Forest AUC: {rf_auc:.3f}")
    
    # Gradient Boosting
    print("\\nTraining Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    
    gb_pred = gb_grid.predict(X_test)
    gb_auc = roc_auc_score(y_test, gb_grid.predict_proba(X_test)[:, 1])
    models['Gradient Boosting'] = gb_grid.best_estimator_
    results['Gradient Boosting'] = gb_auc
    
    print(f"Gradient Boosting - Best params: {gb_grid.best_params_}")
    print(f"Gradient Boosting AUC: {gb_auc:.3f}")
    
    # Logistic Regression
    print("\\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_params = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    
    lr_pred = lr_grid.predict(X_test_scaled)
    lr_auc = roc_auc_score(y_test, lr_grid.predict_proba(X_test_scaled)[:, 1])
    models['Logistic Regression'] = (lr_grid.best_estimator_, scaler)
    results['Logistic Regression'] = lr_auc
    
    print(f"Logistic Regression - Best params: {lr_grid.best_params_}")
    print(f"Logistic Regression AUC: {lr_auc:.3f}")
    
    # SVM
    print("\\nTraining SVM...")
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc', n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    
    svm_pred = svm_grid.predict(X_test_scaled)
    svm_auc = roc_auc_score(y_test, svm_grid.predict_proba(X_test_scaled)[:, 1])
    models['SVM'] = (svm_grid.best_estimator_, scaler)
    results['SVM'] = svm_auc
    
    print(f"SVM - Best params: {svm_grid.best_params_}")
    print(f"SVM AUC: {svm_auc:.3f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_auc = results[best_model_name]
    
    print(f"\\n=== BEST MODEL: {best_model_name} (AUC: {best_auc:.3f}) ===")
    
    # Print detailed results for best model
    if best_model_name in ['Logistic Regression', 'SVM']:
        best_pred = best_model[0].predict(X_test_scaled)
    else:
        best_pred = best_model.predict(X_test)
    
    print("\\nClassification Report:")
    print(classification_report(y_test, best_pred))
    
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    return best_model, best_model_name, best_auc

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed_plant_data.csv')
    
    print(f"Loading processed Kaggle dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features
    print("\\nPreparing features...")
    feature_df, label_encoder = prepare_features(df)
    
    X = feature_df.drop('needs_watering', axis=1)
    y = feature_df['needs_watering']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\\n{y.value_counts()}")
    
    # Train models
    print("\\n" + "="*50)
    print("TRAINING MULTIPLE MODELS")
    print("="*50)
    
    best_model, best_model_name, best_auc = train_models(X, y)
    
    # Save models
    model_path = os.path.join(script_dir, '..', 'model_kaggle.pkl')
    encoder_path = os.path.join(script_dir, '..', 'label_encoder.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\\nBest model ({best_model_name}) saved as: {model_path}")
    print(f"Label encoder saved as: {encoder_path}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print("\\nFeature Importance:")
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance_df)
    elif hasattr(best_model, '__len__') and hasattr(best_model[0], 'feature_importances_'):
        print("\\nFeature Importance:")
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model[0].feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance_df)

if __name__ == "__main__":
    main()