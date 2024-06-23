import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the plot
x_len = 200
y_range = [-1.5, 1.5]
fault_start = 80  # Start of the fault in the data array
fault_end = 120   # End of the fault

# Create the figure and axis
fig, ax = plt.subplots()
xs = np.linspace(0, 4 * np.pi, x_len)
ys = np.sin(xs)
square_ys = np.sign(ys)

# Introduce a fixed fault in the square wave
square_ys[fault_start:fault_end] = -1.5  # Set the fault values below the normal range

# Set up the plot
ax.set_ylim(y_range)
line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

def update(frame):
    # In the animation function, we only update the sine line as the square wave does not change
    ys = np.sin(xs + frame / 10)
    sine_line.set_ydata(ys)
    return line, sine_line

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 400, 1), interval=50, blit=False)
plt.show()
