import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameter
samp_per_cycle = 50   # Sampel per siklus
num_cycles = 4        # Jumlah siklus
error_min = 0.05      # Error minimal (5%)
error_max = 0.50      # Error maksimal (50%)

# Membuat sinyal square wave dasar
t = np.linspace(0, num_cycles, num_cycles * samp_per_cycle, endpoint=False)
square_wave = signal.square(2 * np.pi * t)

# Membuat sinyal error
np.random.seed(0)  # Seed untuk konsistensi
errors = np.random.uniform(low=error_min, high=error_max, size=square_wave.shape)
error_signal = square_wave * (1 + errors * np.random.choice([-1, 1], size=square_wave.shape))

# Plotting sinyal asli dan sinyal dengan error
plt.figure(figsize=(10, 6))
plt.plot(t, square_wave, label='Square Wave')
plt.plot(t, error_signal, label='Square Wave with Error', linestyle='--')
plt.title('Square Wave Signal and Square Wave with Random Error')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
