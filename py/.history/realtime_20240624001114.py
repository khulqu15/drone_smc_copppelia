import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time vector
    y = np.sin(2 * np.pi * freq * t)  # Sine wave formula
    return t, y

# Parameters
frequency = 5  # frequency of the sine wave in Hz
sample_rate = 1000  # how many samples per second
duration = 2  # duration of the signal in seconds

# Generate the signal
time, amplitude = generate_sine_wave(frequency, sample_rate, duration)

# Plotting the signal
plt.figure(figsize=(10, 4))
plt.plot(time, amplitude, label=f'{frequency} Hz Sine Wave')
plt.title('Continuous Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
