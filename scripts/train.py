import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'plant_watering.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded: {len(df)} rows")

# Encode categorical data
df_encoded = pd.get_dummies(df, columns=['plant_type'])
X = df_encoded.drop('needs_watering', axis=1)
y = df_encoded['needs_watering']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Random Forest with hyperparameter tuning
print("\nTraining Random Forest with hyperparameter tuning...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_grid.predict_proba(X_test)[:, 1])
print(f"Best RF params: {rf_grid.best_params_}")
print(f"RF AUC: {rf_auc:.3f}")

# Model 2: Logistic Regression with hyperparameter tuning
print("\nTraining Logistic Regression with hyperparameter tuning...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_params = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
lr = LogisticRegression(random_state=42, max_iter=1000)
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)
lr_pred = lr_grid.predict(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr_grid.predict_proba(X_test_scaled)[:, 1])
print(f"Best LR params: {lr_grid.best_params_}")
print(f"LR AUC: {lr_auc:.3f}")

# Select best model
if rf_auc >= lr_auc:
    best_model = rf_grid.best_estimator_
    best_model_name = "Random Forest"
    best_auc = rf_auc
else:
    best_model = lr_grid.best_estimator_
    best_model_name = "Logistic Regression"
    best_auc = lr_auc

print(f"\nBest model: {best_model_name} (AUC: {best_auc:.3f})")

# Save best model
model_path = os.path.join(script_dir, '..', 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Model saved as {model_path}")

# Save scaler if LR was selected
if best_model_name == "Logistic Regression":
    scaler_path = os.path.join(script_dir, '..', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved as {scaler_path}")