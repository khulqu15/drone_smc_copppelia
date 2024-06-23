import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set the parameters for the plot and animation
x_len = 200  # Number of points to display
y_range = [-1.5, 1.5]  # Y-axis range
fault_interval = 50  # Interval at which faults occur
fault_duration = 20  # Duration of each fault

# Initialize the figure and axes
fig, ax = plt.subplots()
xs = np.linspace(0, 4 * np.pi, x_len)
ys = np.sin(xs)
square_ys = np.sign(ys)
ax.set_ylim(y_range)

# Create the plot lines for the square wave and the sine wave
line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

def update(frame):
    ys = np.sin(xs + frame / 10)
    square_ys = np.sign(ys)

    # Introduce a fault in the square wave
    if frame % fault_interval < fault_duration:
        # Set the faulted part of the wave to 0
        square_ys[(frame % fault_interval):(frame % fault_interval) + fault_duration] = 0

    # Update the data of the plot lines
    line.set_ydata(square_ys)
    sine_line.set_ydata(ys)
    return line, sine_line

# Set up the animation to update the plot periodically
ani = FuncAnimation(fig, update, frames=np.arange(0, 800, 1), interval=50, blit=False)
plt.show()
