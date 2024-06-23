import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
x_len = 200         # Number of points to display
y_range = [-1.5, 1.5]  # Range of possible Y values to display

# Create figure for plotting
fig, ax = plt.subplots()
xs = list(range(0, x_len))
ys = [0] * x_len
ax.set_ylim(y_range)

# Create a line that we will update
line, = ax.plot(xs, ys)

# Function to update the data
def update(frame):
    # Shift y data left
    ys[:-1] = ys[1:]
    # Add a new value
    ys[-1] = np.sin(frame * np.pi / 15)

    # Update the line
    line.set_ydata(ys)
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(0, 400, 1), interval=50, blit=True)
plt.show()
