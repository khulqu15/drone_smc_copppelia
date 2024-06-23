import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from statsmodels.tsa.vector_ar.var_model import VAR

# Generate sample data
def generate_data(points):
    t = np.linspace(0, 10, points)
    data = pd.DataFrame({
        'Sine': np.sin(2 * np.pi * t),
        'Cosine': np.cos(2 * np.pi * t)
    })
    return data

# Fit VAR model
def fit_var(data, maxlags=5):
    model = VAR(data)
    fitted_model = model.fit(maxlags=maxlags)
    return fitted_model

# Initialize data
data = generate_data(1000)
fitted_model = fit_var(data)

# Prepare for animation
fig, ax = plt.subplots()
x = np.arange(1000)
lines = {
    'true_sine': ax.plot(x, data['Sine'], label='True Sine', color='blue')[0],
    'predicted_sine': ax.plot(x, np.nan * np.ones_like(data['Sine']), label='Predicted Sine', color='red', linestyle='--')[0]
}
ax.legend()
ax.set_ylim(-1.5, 1.5)

# Animation update function
def update(frame):
    global data
    # Append new points based on the VAR model predictions
    last_obs = data.tail(1)
    pred = fitted_model.forecast(last_obs.values, steps=1)
    new_point = pd.DataFrame(pred, columns=data.columns)
    data = pd.concat([data, new_point]).reset_index(drop=True)
    
    # Update true data
    lines['true_sine'].set_ydata(data['Sine'])
    
    # Display only the last 1000 points for performance
    if len(data) > 1000:
        data = data.iloc[-1000:].reset_index(drop=True)
        x = np.arange(1000)
        lines['true_sine'].set_xdata(x)
        lines['predicted_sine'].set_xdata(x)
    
    # Update predicted data (with some offset to visualize)
    predicted_data = fitted_model.forecast(data[-maxlags:].values, steps=50)
    predicted_sine = predicted_data[:, 0]
    full_predicted_sine = np.concatenate([np.nan * np.ones(1000 - 50), predicted_sine])
    lines['predicted_sine'].set_ydata(full_predicted_sine)
    
    return lines['true_sine'], lines['predicted_sine']

# Start animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
