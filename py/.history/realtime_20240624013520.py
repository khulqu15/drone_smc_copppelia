import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Parameter dasar
dt = 0.05
tfinal = 100
amplitude = 25  # Amplitude dari square wave
frequency = 0.08  # Frekuensi dari square wave (dalam Hz)
error_threshold = 0.1  # Threshold error 10%

n = int(tfinal / dt)
xtraj = np.zeros(n + 1, float)
xtraj_noisy = np.zeros(n + 1, float)
trange = np.linspace(start=0, stop=tfinal, num=n + 1)

for i in range(n):
    if 70 <= i:
        current_amplitude = amplitude
        xtraj[i + 1] = current_amplitude * np.sign(np.sin(2 * np.pi * frequency * trange[i + 1]))
        error = np.random.normal(0, current_amplitude * 0.1) + np.sin(2 * np.pi * 10 * trange[i + 1]) * 0.5  # Ripple
    else:
        current_amplitude = amplitude
        xtraj[i + 1] = current_amplitude * np.sign(np.sin(2 * np.pi * frequency * trange[i + 1]))
        if i > 0 and xtraj[i + 1] != xtraj[i]:
            error = np.random.normal(0, current_amplitude * 0.1)  # Noise Gaussian dengan std dev 10% dari amplitude
        else:
            error = 0

    xtraj_noisy[i + 1] = xtraj[i + 1] + error

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

def animate(i):
    ax.cla()
    ax.plot(trange[:i], xtraj_noisy[:i], label='Actual Signal', linestyle='--')
    ax.plot(trange[:i], xtraj[:i], label='Square Wave')
    
    for j in range(1, i):
        error_value = abs(xtraj_noisy[j] - xtraj[j]) / amplitude
        if error_value > error_threshold and 80 <= j <= 90:
            ax.text(trange[j], xtraj_noisy[j], f'Error: {error_value*100:.1f}%', color='red')
    
    ax.set_xlim([0, tfinal])
    ax.set_ylim([-1.5 * amplitude, 1.5 * amplitude])
    ax.legend(loc='upper right')
    ax.set_title('Fault Representation')

anim = animation.FuncAnimation(fig, animate, frames=len(trange), interval=1, blit=False)
plt.show()
