import pandas as pd
import numpy as np
import os

def process_kaggle_dataset():
    """Process the Kaggle indoor plant dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, '..', 'data', '100 indoor plant dataset.xlsx')
    
    print(f"Loading Kaggle dataset from: {excel_path}")
    
    # Read all sheets from Excel file
    excel_file = pd.ExcelFile(excel_path)
    print(f"Available sheets: {excel_file.sheet_names}")
    
    # Read the first sheet (assuming it contains the main data)
    df = pd.read_excel(excel_path, sheet_name=0)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Create a synthetic 'needs_watering' target based on plant characteristics
    # This is a heuristic approach since the original dataset might not have this target
    if 'needs_watering' not in df.columns:
        print("\nCreating synthetic 'needs_watering' target...")
        
        # Create needs_watering based on multiple factors
        df['needs_watering'] = 0
        
        # Check if we have relevant columns for watering prediction
        watering_factors = []
        
        # Look for moisture-related columns
        moisture_cols = [col for col in df.columns if 'moisture' in col.lower() or 'water' in col.lower()]
        if moisture_cols:
            print(f"Found moisture columns: {moisture_cols}")
            # Low moisture = needs watering
            for col in moisture_cols:
                if df[col].dtype in ['int64', 'float64']:
                    df.loc[df[col] < df[col].quantile(0.3), 'needs_watering'] = 1
                    watering_factors.append(f"low_{col}")
        
        # Look for time-related columns
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'day' in col.lower() or 'hour' in col.lower()]
        if time_cols:
            print(f"Found time columns: {time_cols}")
            for col in time_cols:
                if df[col].dtype in ['int64', 'float64']:
                    df.loc[df[col] > df[col].quantile(0.7), 'needs_watering'] = 1
                    watering_factors.append(f"high_{col}")
        
        # Look for temperature/humidity columns
        env_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'humid', 'light'])]
        if env_cols:
            print(f"Found environmental columns: {env_cols}")
        
        # If no specific watering logic can be applied, create random but realistic distribution
        if df['needs_watering'].sum() == 0:
            print("Creating random but realistic watering needs distribution...")
            np.random.seed(42)
            df['needs_watering'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        
        print(f"Target distribution:\n{df['needs_watering'].value_counts()}")
    
    # Save processed dataset
    output_path = os.path.join(script_dir, '..', 'data', 'processed_plant_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    df = process_kaggle_dataset()