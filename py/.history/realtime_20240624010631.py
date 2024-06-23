import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

amplitude = 1
iterations = 200

t = np.arange(0, iterations)

square_wave = amplitude * np.sign(np.sin(2 * np.pi * t / 20))

def update(frame):
    ax.clear()
    ax.plot(t[:frame], square_wave[:frame], label='Square Wave')
    
    error_percentage = np.random.uniform(5, 50) / 100
    noise = error_percentage * np.random.randn(frame)
    
    noisy_square_wave = square_wave[:frame] + noise
    
    ax.plot(t[:frame], noisy_square_wave, label=f'Square Wave with {error_percentage*100:.1f}% Error')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=iterations, repeat=False)

plt.show()
