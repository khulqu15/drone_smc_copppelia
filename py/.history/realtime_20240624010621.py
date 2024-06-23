import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter dasar
amplitude = 1
iterations = 200

# Membuat array waktu (iterasi)
t = np.arange(0, iterations)

# Menghasilkan square wave dasar
square_wave = amplitude * np.sign(np.sin(2 * np.pi * t / 20))

# Fungsi untuk mengupdate plot secara real-time
def update(frame):
    ax.clear()
    # Square wave dasar
    ax.plot(t[:frame], square_wave[:frame], label='Square Wave')
    
    # Menghitung error
    error_percentage = np.random.uniform(5, 50) / 100
    noise = error_percentage * np.random.randn(frame)
    
    # Square wave dengan error
    noisy_square_wave = square_wave[:frame] + noise
    
    # Plot square wave dengan error
    ax.plot(t[:frame], noisy_square_wave, label=f'Square Wave with {error_percentage*100:.1f}% Error')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=iterations, repeat=False)

plt.show()
