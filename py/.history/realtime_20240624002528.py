import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_len = 200
y_range = [-1.5, 1.5]

fig, ax = plt.subplots()
xs = np.linspace(0, 4 * np.pi, x_len)
ys = np.sin(xs)
square_ys = np.sign(ys)
ax.set_ylim(y_range)

line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

def introduce_faults(square_wave, fault_probability=0.05, fault_amplitude=0):
    fault_indices = np.random.rand(len(square_wave)) < fault_probability
    square_wave[fault_indices] = fault_amplitude
    return square_wave

def update(frame):
    ys = np.sin(xs + frame / 10)
    square_ys = np.sign(ys)

    line.set_ydata(square_ys)
    sine_line.set_ydata(ys)
    return line, sine_line

ani = FuncAnimation(fig, update, frames=np.arange(0, 400, 1), interval=50, blit=False)
plt.show()
