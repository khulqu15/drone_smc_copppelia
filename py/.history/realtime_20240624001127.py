import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time vector
    y = np.sin(2 * np.pi * freq * t)  # Sine wave formula
    return t, y

frequency = 5
sample_rate = 1000
duration = 2

time, amplitude = generate_sine_wave(frequency, sample_rate, duration)

plt.figure(figsize=(10, 4))
plt.plot(time, amplitude, label=f'{frequency} Hz Sine Wave')
plt.title('Continuous Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
