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

def calculate_residuals(data):
    actual_columns = ['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
    target_columns = ['Target X', 'Target Y', 'Target Z', 'Target Roll', 'Target Pitch', 'Target Yaw']
    residuals = data[actual_columns].sub(data[target_columns].values)
    return residuals

def detect_faults(residuals, threshold_factor=2):
    std_dev_threshold = residuals.std()
    fault_mask = (residuals.abs() > threshold_factor * std_dev_threshold)
    return fault_mask

def visualize_faults(fault_mask, data):
    time_series = np.arange(len(data))
    plt.figure(figsize=(14, 7))
    for i, column in enumerate(fault_mask.columns, 1):
        ax = plt.subplot(2, 3, i)
        ax.plot(time_series, data[column], label='Data')
        ax.scatter(time_series[fault_mask[column]], data[column][fault_mask[column]], color='red', label='Fault')
        ax.set_title(column)
        ax.set_xlabel('Time (index)')
        ax.set_ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
def report_faults(fault_mask):
    fault_counts = fault_mask.sum()
    print("Fault Report:")
    print(fault_counts)
    print("\nDetailed fault instances for each parameter:")
    for column in fault_mask.columns:
        print(f"\n{column} Faults Indexes:")
        print(fault_mask[fault_mask[column]].index.tolist())