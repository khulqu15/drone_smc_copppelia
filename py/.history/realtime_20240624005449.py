import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters and plot
x_len = 200
y_range = [-25, 25]
xs = np.linspace(0, 5 * np.pi + 10, x_len)
ys = np.sin(xs)
square_ys = np.sign(ys)

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
ax.set_ylim(y_range)

line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

frame = 0
try:
    while True:
        ys = np.sin(xs + frame / 10)
        square_ys = np.sign(ys)
        
        line.set_ydata(square_ys)
        sine_line.set_ydata(ys)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)  # Pause for a bit, to control the update rate
        
        frame += 1
except KeyboardInterrupt:
    print("Stopped animation.")
