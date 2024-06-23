import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter dasar untuk sinyal square wave
frekuensi = 1  # Frekuensi sinyal dalam Hz
amplitudo = 1  # Amplitudo sinyal
durasi = 5     # Durasi sinyal dalam detik
sampling_rate = 1000  # Jumlah sampel per detik

# Membuat array waktu berdasarkan durasi dan sampling rate
t = np.linspace(0, durasi, int(sampling_rate * durasi), endpoint=False)

# Fungsi untuk mengenerate square wave
def generate_square_wave(t, frekuensi, amplitudo):
    return amplitudo * np.sign(np.sin(2 * np.pi * frekuensi * t))

# Fungsi untuk mengenerate square wave dengan error
def generate_square_wave_with_error(t, frekuensi, amplitudo, error):
    clean_signal = generate_square_wave(t, frekuensi, amplitudo)
    noise = np.random.uniform(-error, error, size=clean_signal.shape)
    return clean_signal + noise * clean_signal

# Inisialisasi figure dan axis untuk plotting
fig, ax = plt.subplots()
line1, = ax.plot(t, generate_square_wave(t, frekuensi, amplitudo), label='Clean Square Wave')
line2, = ax.plot(t, generate_square_wave_with_error(t, frekuensi, amplitudo, 0.5), label='Square Wave with Error')

# Fungsi update untuk animation
def update(frame):
    error_percentage = np.random.uniform(0.05, 0.5)  # Random error percentage between 5% and 50%
    line2.set_ydata(generate_square_wave_with_error(t, frekuensi, amplitudo, error_percentage))
    ax.set_title(f"Square Wave with {error_percentage*100:.2f}% Error")
    return line2,

# Animasi plot
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.legend()
plt.show()
