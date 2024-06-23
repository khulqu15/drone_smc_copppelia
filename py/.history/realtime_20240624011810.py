import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

dt = 0.01
tfinal = 100
x0 = 0
amplitude = 26  # Amplitude dari square wave
frequency = 0.05  # Frekuensi dari square wave (dalam Hz)

sqrtdt = np.sqrt(dt)
n = int(tfinal / dt)
xtraj = np.zeros(n + 1, float)
trange = np.linspace(start=0, stop=tfinal, num=n + 1)
xtraj[0] = x0

# Menghasilkan square wave dengan amplitude dan frekuensi yang diatur
for i in range(n):
    xtraj[i + 1] = amplitude * np.sign(np.sin(2 * np.pi * frequency * trange[i + 1]))

x = trange
y = xtraj

# animation line plot example
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

def animate(i):
    ax.cla()  # clear the previous image
    ax.plot(x[:i], y[:i])  # plot the line
    ax.set_xlim([x0, tfinal])  # fix the x axis
    ax.set_ylim([-1.1 * amplitude, 1.1 * amplitude])  # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames=len(x) + 1, interval=1, blit=False)
plt.show()
