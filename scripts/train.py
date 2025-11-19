import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'plant_watering.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded: {len(df)} rows")
df = pd.get_dummies(df, columns=['plant_type'])
X = df.drop('needs_watering', axis=1)
y = df['needs_watering']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training model...")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model saved as model.pkl")
