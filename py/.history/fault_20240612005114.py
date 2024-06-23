import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    # Asumsi kolom sudah dalam tipe data yang benar, hanya mengisi nilai yang hilang
    data.fillna(method='ffill', inplace=True)
    return data

def calculate_residuals(data):
    target_columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for col in columns:
        std_dev = 0.05 * data[col].std()
        data['Actual ' + col] = data[col] + np.random.normal(0, std_dev, data.shape[0])
    
    return data

def detect_faults(residuals):
    # Menggunakan batas deviasi standar untuk mengidentifikasi fault
    threshold = 3 * residuals.std()
    fault_mask = (residuals.abs() > threshold)
    return fault_mask

def visualize_data(data, fault_mask):
    plt.figure(figsize=(15, 10))
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for i, col in enumerate(columns):
        plt.subplot(2, 3, i+1)
        plt.plot(data[col], label='Target')
        plt.plot(data['Actual ' + col], label='Actual')
        plt.scatter(fault_mask.index[fault_mask['Actual ' + col]], data['Actual ' + col][fault_mask['Actual ' + col]], color='red', label='Fault')
        plt.title(f'{col}')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = '../droneStateLog.csv'
    data = load_data(file_path)
    if data is not None:
        data = preprocess_data(data)
        residuals = calculate_residuals(data)
        fault_mask = detect_faults(residuals)
        visualize_data(data, fault_mask)

if __name__ == '__main__':
    main()
