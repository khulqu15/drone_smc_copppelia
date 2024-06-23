import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.signal import stft
from sklearn.decomposition import PCA

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
    columns = ['Actual Roll', 'Actual Pitch', 'Actual Yaw']
    titles = ['Roll', 'Pitch', 'Yaw']
    faults = {'Motor 1': 157, 'Motor 2': 137}

    for i, col in enumerate(columns):
        plt.subplot(1, 3, i+1)
        Pxx, freqs, bins, im = plt.specgram(data[col], NFFT=256, Fs=1, noverlap=128, scale='dB', cmap='viridis')
        
        plt.title(f'Spectrogram of {titles[i]}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(im, label='Power [dB/sample]')

        for fault, sample in faults.items():
            plt.axvline(x=sample, color='red', linestyle='--', linewidth=1)
            plt.text(sample + 0.5, 0.9 * plt.ylim()[1], fault, rotation=90, verticalalignment='top', color='red')

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
    # print(fault_counts)
    print("\nDetailed fault instances for each parameter:")
    for column in fault_mask.columns:
        print(f"\n{column} Faults Indexes:")
        # print(fault_mask[fault_mask[column]].index.tolist())

def main():
    file_path = '../droneStateLog.csv'
    data = load_data(file_path)
    
    if data is not None:
        m = 1.0
        g = 9.81
        k_x, k_y, k_z = 0.1, 0.1, 0.1
        u_z = 15.0
        
        initial_state = [0, 0, 0, 0, 0, 0]
        t_span = [0, 100]
        
        orientation_and_z = data[['Roll', 'Pitch', 'Yaw', 'Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw', 'Actual Z']].to_numpy()

        model = VAR(orientation_and_z)
        results = model.fit(maxlags=6)

        irf = results.irf(10)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        variables = ['Roll', 'Pitch', 'Yaw', 'Z']
        actual_variables = ['Actual Roll', 'Actual Pitch', 'Actual Yaw', 'Actual Z']

        for i, (var, actual_var) in enumerate(zip(variables, actual_variables)):
            ax = axes[i//2, i%2]
            response_idx = results.names.index(actual_var)  # index of the response in the VAR results
            impulse_idx = results.names.index(var)          # index of the impulse in the VAR results
            ax.plot(irf.irfs[:, response_idx, impulse_idx], label=f'{var} to {actual_var}')
            ax.set_title(f'IRF from {var} to {actual_var}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Response')
            ax.legend()

        plt.tight_layout()
        plt.show()
        
        fevd = results.fevd(10)
        fig = fevd.plot()
        for i, ax in enumerate(fig.get_axes()):
            ax.set_title('FEVD for ' + labels[i])
        plt.show()

        f, t, Zxx = stft(orientation_and_z[:, 0], nperseg=100)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(orientation_and_z)
        print(principalComponents)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=0.7, edgecolors='w', s=50)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Orientation and Z Data')
        plt.grid(True)
        plt.show()
        
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
