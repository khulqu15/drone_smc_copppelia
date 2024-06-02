import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def preprocess_data(data):
    numeric_columns = data.columns.drop(['EKF Yaw', 'EKF Pitch', 'EKF Roll'])
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.fillna(method='ffill', inplace=True)
    return data