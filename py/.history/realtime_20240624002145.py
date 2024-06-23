import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Adjusting the backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from statsmodels.tsa.vector_ar.var_model import VAR

maxlags = 5

# Generate data
def generate_data(points):
    t = np.linspace(0, 10, points)
    return pd.DataFrame({
        'Sine': np.sin(2 * np.pi * t),
        'Cosine': np.cos(2 * np.pi * t)
    })

# Fit VAR model
def fit_var(data, maxlags=5):
    model = VAR(data)
    return model.fit(maxlags=maxlags)

# Initialize data
data = generate_data(500)
fitted_model = fit_var(data)

# Animation setup
fig, ax = plt.subplots()
lines = {
    'Sine': ax.plot(data.index, data['Sine'], label='Sine')[0],
    'Cosine': ax.plot(data.index, data['Cosine'], label='Cosine', linestyle='--')[0]
}
ax.legend()

def update(frame):
    global data
    last_obs = data.iloc[-maxlags:]  # Using enough past observations
    pred = fitted_model.forecast(last_obs.values, steps=1)
    new_point = pd.DataFrame(pred, columns=data.columns, index=[data.index[-1] + 1])
    data = pd.concat([data, new_point])
    lines['Sine'].set_data(data.index, data['Sine'])
    lines['Cosine'].set_data(data.index, data['Cosine'])
    ax.set_xlim(data.index[0], data.index[-1] + 1)  # Adjust x-axis to fit new data
    return lines['Sine'], lines['Cosine']

ani = FuncAnimation(fig, update, frames=np.arange(100), interval=50)
plt.show()
