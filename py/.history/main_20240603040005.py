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
    severity_index = (residuals.abs() / std_dev_threshold).max(axis=1)
    fault_category = pd.cut(severity_index, bins=[0, 1, 2, np.inf], labels=['Low', 'Medium', 'High'])
    return fault_mask, severity_index, fault_category

def add_fault_data_to_dataframe(data, fault_mask, severity_index, fault_category):
    data['Fault Detected'] = fault_mask.any(axis=1)
    data['Severity Index'] = severity_index
    data['Fault Category'] = fault_category
    return data

def visualize_spectrograms(data):
    plt.figure(figsize=(15, 5))
    titles = ['Roll', 'Pitch', 'Yaw']
    for i, key in enumerate(['Actual Roll', 'Actual Pitch', 'Actual Yaw'], 1):
        plt.subplot(1, 3, i)
        plt.specgram(data[key], NFFT=256, Fs=1, noverlap=128, cmap='viridis')
        plt.title(f'Spectrogram of {titles[i-1]}')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.show()

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

def main():
    file_path = '../droneStateLog.csv'
    data = load_data(file_path)
    if data is not None:
        data = preprocess_data(data)
        residuals = calculate_residuals(data)
        fault_mask, severity_index, fault_category = detect_faults(residuals)
        data = add_fault_data_to_dataframe(data, fault_mask, severity_index, fault_category)
        report_faults(fault_mask)
        visualize_faults(fault_mask, data[['Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw']])
        visualize_spectrograms(data)
        data.to_csv(file_path.replace('.csv', '_enhanced_faults.csv'), index=False)

if __name__ == '__main__':
    main()