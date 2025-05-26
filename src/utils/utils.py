import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}. First 5 rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file {file_path}: {e}")
        return None
    

def preprocess_data(df, label_col):
    """
    Encodes the label column and scales all other numeric features.
    Returns X (ndarray) and y (ndarray of integer labels).
    """
    # 1) Label-Encoding
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    
    # 2) Features auswählen (alles außer label_col)
    X = df.drop(columns=[label_col]).values.astype(float)
    
    # 3) Skalierung
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes.")
    return X, y