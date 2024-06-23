import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
x_len = 200
y_range = [-2, 2]  # Updated range to accommodate faults

# Create the plot
fig, ax = plt.subplots()
xs = np.linspace(0, 4 * np.pi, x_len)
ys = np.sin(xs)
square_ys = np.sign(ys)

# Introduce a fixed fault in the square wave at specific indices
fault_indices = np.linspace(50, 150, 20, dtype=int)  # Example fault indices
fault_values = np.random.choice([-2, 2], size=fault_indices.size)  # Fault values at -2 or 2 for visibility

for i, idx in enumerate(fault_indices):
    square_ys[idx] = fault_values[i]

# Set plot limits and add the lines
ax.set_ylim(y_range)
line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Faulty Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

# Define the update function for the animation
def update(frame):
    # Update sine wave with the time offset
    ys = np.sin(xs + frame / 10)
    sine_line.set_ydata(ys)
    
    # The square wave with faults remains static
    return line, sine_line

# Set up plot to call update() function periodically
ani = FuncAnimation(fig, update, frames=np.arange(0, 400, 1), interval=50, blit=False)
plt.show()
