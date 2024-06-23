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
    data.fillna(method='ffill', inplace=True)
    
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for column in columns:
        noise = np.random.normal(0, 0.05 * data[column].std(), data[column].shape)
        window = np.ones(10) / 10
        smooth = np.convolve(noise, window, mode='same')
        data['Actual ' + column] = data[column] + smooth
        
    return data

def calculate_residuals(data):
    target_columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    actual_columns = ['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
    residuals = data[actual_columns] - data[target_columns]
    return residuals

def detect_faults(residuals):
    threshold = 3 * residuals.std()
    fault_mask = (residuals.abs() > threshold)
    return fault_mask

def visualize_data(data, fault_mask):
    plt.figure(figsize=(15, 10))
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for i, col in enumerate(columns):
        plt.subplot(2, 3, i+1)
        plt.plot(data['Actual ' + col], label='Actual')
        plt.scatter(fault_mask.index[fault_mask['Actual ' + col]], data['Actual ' + col][fault_mask['Actual ' + col]], color='red', label='Fault')
        plt.plot(data[col], label='Target')
        plt.title(f'{col}')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()
    
def visualize_motor_health(data, residuals):
    plt.figure(figsize=(18, 6))
    angles = ['Roll', 'Pitch', 'Yaw']
    residuals = ['Residual Roll', 'Residual Pitch', 'Residual Yaw']

    for i, (angle, residual) in enumerate(zip(angles, residuals)):
        plt.subplot(1, 3, i+1)
        plt.scatter(data[angle], data[residual], alpha=0.6, label='Residual')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residual vs {angle}')
        plt.xlabel(f'{angle} Angle')
        plt.ylabel('Residual')
        plt.grid(True)
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
        visualize_motor_health(data, residuals)

if __name__ == '__main__':
    main()
