import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_len = 200
y_range = [-1.5, 1.5]

fig, ax = plt.subplots()
xs = list(range(0, x_len))
ys = [0] * x_len
ax.set_ylim(y_range)

line, = ax.plot(xs, ys)

def update(frame):
    ys[:-1] = ys[1:]
    ys[-1] = np.sin(frame * np.pi / 15)

    line.set_ydata(ys)
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(0, 400, 1), interval=50, blit=True)
plt.show()
