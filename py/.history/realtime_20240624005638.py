import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters and plot
x_len = 200
y_range = [-1.5, 1.5]
xs = np.linspace(0, 4 * np.pi, x_len)
ys = np.zeros(xs.shape)  # Start with a zero signal
square_ys = np.sign(ys)  # This will also start as zero because sin(0) = 0

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
ax.set_ylim(y_range)

line, = ax.plot(xs, square_ys, 'b-', linewidth=2, label='Square Wave')
sine_line, = ax.plot(xs, ys, 'r--', alpha=0.5, label='Sine Wave')
ax.legend()

frame = 0
try:
    while True:
        # Incrementally update the sine wave
        ys = np.sin(xs + frame * np.pi / 100)  # Increase phase shift slowly
        square_ys = np.sign(ys)  # Update the square wave based on the sine wave

        # Update the plot lines
        line.set_ydata(square_ys)
        sine_line.set_ydata(ys)

        # Refresh the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)  # Pause for a bit, to control the update rate
        
        frame += 1
except KeyboardInterrupt:
    print("Stopped animation.")
