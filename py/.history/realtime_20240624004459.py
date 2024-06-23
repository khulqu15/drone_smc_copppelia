import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up parameters for the plot and fault generation
x_len = 200
y_range = [-2, 2]  # Extended range to better visualize faults
fault_probability = 0.1  # Probability that a fault occurs at any given frame
fault_duration = 10  # Duration of the fault in frames
xs = np.linspace(0, 4 * np.pi, x_len)

# Initialize the figure and axes
fig, ax = plt.subplots()
ys = np.sin(xs)
square_ys = np.sign(ys)
ax.set_ylim(y_range)

# Create the plot lines for the square wave and the sine wave
line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

# Initialize variables to manage fault state
fault_countdown = 0

def update(frame):
    global fault_countdown
    ys = np.sin(xs + frame / 10)
    square_ys = np.sign(ys)

    # Decide randomly whether to introduce a fault
    if fault_countdown <= 0 and np.random.rand() < fault_probability:
        fault_countdown = fault_duration  # Reset the fault countdown

    # Apply fault by setting values to zero or another distinct pattern
    if fault_countdown > 0:
        square_ys[(frame % x_len):(frame % x_len + 20)] = 0  # Example fault: zeroing part of the wave
        fault_countdown -= 1  # Decrement the fault countdown

    # Update the data of the plot lines
    line.set_ydata(square_ys)
    sine_line.set_ydata(ys)
    return line, sine_line

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 800, 1), interval=50, blit=False)
plt.show()
