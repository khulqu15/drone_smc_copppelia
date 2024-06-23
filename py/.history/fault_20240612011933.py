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
        noise = np.random.normal(0, 0.07 * data[column].std(), data[column].shape)
        window = np.ones(10) / 10
        smooth = np.convolve(noise, window, mode='same')
        data['Actual ' + column] = data[column] + smooth
        
    return data

def calculate_residuals(data):
    target_columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    actual_columns = ['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
    residuals = data[actual_columns] - data[target_columns]
    residuals.fillna(0, inplace=True)
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
    
def simulate_motor_health(data, residuals):
    motor_weights = np.random.rand(4, residuals.shape[1])  * 0.1
    print("Random Weights Before Normalization:", motor_weights)
    
    motor_weights /= motor_weights.sum(axis=0)  # Normalisasi bobot
    print("Normalized Weights:", motor_weights)

    motor_health = {}
    for i in range(1, 5):
        weighted_error = np.abs(residuals * motor_weights[i-1])
        motor_health['Motor' + str(i)] = np.nanmean(weighted_error, axis=1)
        print(f"Weighted Error for Motor {i}:", motor_health)
    
    healthy_baseline = np.nanmean(np.abs(residuals), axis=0) * 0.5
    motor_health['Healthy'] = healthy_baseline + np.random.normal(0, 0.02, size=healthy_baseline.shape)
    print("Health for Healthy Motor:", motor_health['Healthy'])

    return motor_health
    
def plot_motor_health(data, motor_health):
    angles = ['Roll', 'Pitch', 'Yaw']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, angle in enumerate(angles):
        ax = axes[i]
        for key, values in motor_health.items():
            ax.plot(data[angle], values, label=key)
        
        ax.set_title(f'Motor Health vs. {angle}')
        ax.set_xlabel(f'{angle} Angle')
        ax.set_ylabel('Health Metric (Lower is Better)')
        ax.legend()
        ax.grid(True)
    
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
        motor_health = simulate_motor_health(data, residuals)
        print(motor_health)
        plot_motor_health(data, motor_health)

if __name__ == '__main__':
    main()
