import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Parameter dasar
dt = 0.05
tfinal = 100
amplitude = 25  # Amplitude dari square wave
frequency = 0.08  # Frekuensi dari square wave (dalam Hz)
error_threshold = 0.01  # Threshold error 10%

n = int(tfinal / dt)
xtraj = np.zeros(n + 1, float)
xtraj_noisy = np.zeros(n + 1, float)
trange = np.linspace(start=0, stop=tfinal, num=n + 1)

for i in range(n):
    xtraj[i + 1] = amplitude * np.sign(np.sin(2 * np.pi * frequency * trange[i + 1]))
    if i > 0 and xtraj[i + 1] != xtraj[i]:
        error = np.random.normal(0, amplitude * 0.1)  # Noise Gaussian dengan std dev 10% dari amplitude
        xtraj_noisy[i + 1] = xtraj[i + 1] + error
    else:
        xtraj_noisy[i + 1] = xtraj[i + 1]
    xtraj_noisy[i + 1] = xtraj[i + 1] - error

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

def animate(i):
    ax.cla()
    ax.plot(trange[:i], xtraj_noisy[:i], label='Noisy Square Wave', linestyle='--')
    ax.plot(trange[:i], xtraj[:i], label='Square Wave')
    
    for j in range(1, i):
        error_value = abs(xtraj_noisy[j] - xtraj[j]) / amplitude
    
    ax.set_xlim([0, tfinal])
    ax.set_ylim([-1.5 * amplitude, 1.5 * amplitude])
    ax.legend(loc='upper right')
    ax.set_title('Square Wave with Fault Representation')

anim = animation.FuncAnimation(fig, animate, frames=len(trange), interval=1, blit=False)
plt.show()
