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
    non_stationary = [key for key, value in results.items() if value['p-value'] > 0.05]  # P-value threshold to check stationarity
    for column in non_stationary:
        data[column] = data[column].diff().fillna(0)  # Differencing to make data stationary
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
    
    columns = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
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
        plt.plot(data['Actual ' + col], label='Actual')
        plt.scatter(fault_mask.index[fault_mask['Actual ' + col]], data['Actual ' + col][fault_mask['Actual ' + col]], color='red', label='Fault')
        plt.plot(data[col], label='Target', alpha=0.5)
        plt.title(f'{col}')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()
    
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
    
        data = make_stationary(data[['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']])  # Ensure you pass only the required columns
        var_model = train_var_model(data)
        residuals = var_model.resid
        fault_mask = detect_faults_from_residuals(residuals)
        visualize_data(data, fault_mask)

if __name__ == '__main__':
    main()