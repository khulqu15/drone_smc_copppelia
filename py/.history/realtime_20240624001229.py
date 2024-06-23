import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
x_len = 200         # Number of points to display
y_range = [-10, 10]  # Range of possible Y values to display

# Create figure for plotting
fig, ax = plt.subplots()
xs = np.arange(0, x_len)
ys = np.zeros(x_len)
line, = ax.plot(xs, ys)

# Adjust the plot
ax.set_ylim(y_range)

# This function is called periodically from FuncAnimation
def animate(i, ys):
    # Add y-value to list
    ys = np.roll(ys, -1)
    ys[-1] = 10 * np.sin(np.radians(i))

    # Update line with new Y values
    line.set_ydata(ys)
    return line,

# Set up plot to call animate() function periodically
ani = FuncAnimation(fig, animate, fargs=(ys,), interval=50, blit=True)
plt.show()
