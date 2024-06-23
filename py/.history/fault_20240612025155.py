import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_stationarity(data):
    results = {}
    for column in data.columns:
        result = adfuller(data[column])
        results[column] = {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}
    return results

def make_stationary(data):
    results = check_stationarity(data)
    non_stationary = [key for key, value in results.items() if value['p-value'] > 0.05]
    for column in non_stationary:
        data.loc[:, column] = data.loc[:, column].diff().fillna(0)
    print("Columns after making stationary:", data.columns)
    return data

def train_var_model(data):
    model = VAR(data)
    fitted_model = model.fit(maxlags=15, ic='aic')  # Automatically select the lag order with Akaike Information Criterion
    return fitted_model

def detect_faults_from_residuals(residuals):
    threshold = 3 * residuals.std()
    fault_mask = (residuals.abs() > threshold)
    return fault_mask

def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)
    
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
    for column in columns:
        scale = 0.05 * data[column].std()
        noise = np.random.normal(0, scale, size=data[column].size)
        window = np.ones(10) / 10
        smooth = np.convolve(noise, window, mode='same')
        data['Actual ' + column] = data[column] + smooth + 0.005
        
    return data

def calculate_residuals(data):
    target_columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    actual_columns = ['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
    residuals = data[actual_columns] - data[target_columns]
    residuals.fillna(0, inplace=True)
    return residuals

def detect_faults(residuals):
    threshold = 0.5 * residuals.std()
    fault_mask = (residuals.abs() > threshold)
    return fault_mask

def visualize_data(data, fault_mask):
    plt.figure(figsize=(15, 10))
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for i, col in enumerate(columns):
        plt.subplot(2, 3, i+1)
        actual_col = 'Actual ' + col
        plt.plot(data[actual_col], label='Actual')

        aligned_fault_mask = fault_mask.reindex(data.index, fill_value=False)
        plt.scatter(data[actual_col][aligned_fault_mask[actual_col]].index, data[actual_col][aligned_fault_mask[actual_col]], color='red', label='Fault')
        plt.plot(data[col], label='Target', alpha=0.5)
        plt.title(col)
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

def visualize_motor_faults(data, motor_faults):
    plt.figure(figsize=(18, 5))
    for i, (motor_label, fault_indices) in enumerate(motor_faults.items(), 1):
        plt.subplot(1, 4, i)
        plt.plot(data['Actual ' + data.columns[i-1 % 4]], label='Actual')
        plt.scatter(fault_indices, data.loc[fault_indices, 'Actual ' + data.columns[i-1 % 4]], color='red', label='Faults')
        plt.title(motor_label)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
def detect_motor_faults_from_residuals(residuals, data):
    threshold = 3 * residuals.std()
    motor_faults = {}
    motor_labels = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']

    num_motors = 4
    residuals_per_motor = len(residuals.columns) // num_motors

    for i in range(num_motors):
        motor_residuals = residuals.iloc[:, i*residuals_per_motor:(i+1)*residuals_per_motor]
        fault_mask = (motor_residuals.abs() > threshold).any(axis=1)
        motor_faults[motor_labels[i]] = data.index[fault_mask].tolist()  # List of indices where faults occur

    return motor_faults
    
def simulate_motor_health(data, residuals):
    motor_weights = np.random.rand(4, residuals.shape[1])
    motor_weights /= motor_weights.sum(axis=0)

    motor_health = {}
    for i in range(1, 5):
        weighted_health = np.abs(residuals * motor_weights[i-1])
        noise = np.random.normal(0, 0.00001, size=weighted_health.shape)  # Make sure noise matches the shape of weighted_health
        motor_health['Motor' + str(i)] = (weighted_health + noise).mean(axis=1)  # Calculate mean health, adding noise correctly
    
    healthy_scale = 0.025 * np.nanmean(residuals.abs(), axis=1)
    healthy_noise = np.random.normal(0, healthy_scale, size=residuals.shape[0])
    motor_health['Healthy'] = np.nanmean(residuals.abs(), axis=1) * 0.5 + healthy_noise

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
        ax.set_ylim(-0.0001, 0.0001)
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

        # Make sure to only pass the needed columns and verify the output
        data_columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']
        stationary_data = make_stationary(data[data_columns].copy())
        var_model = train_var_model(stationary_data)
        residuals = var_model.resid
        fault_mask = detect_faults_from_residuals(residuals)
        visualize_data(stationary_data, fault_mask)
        
        motor_faults = detect_motor_faults_from_residuals(residuals, stationary_data)
        visualize_motor_faults(stationary_data, motor_faults)

if __name__ == '__main__':
    main()
