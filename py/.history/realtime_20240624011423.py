import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

dt = 0.001
tfinal = 0.5
x0 = 0

sqrtdt = np.sqrt(dt)
n = int(tfinal / dt)
xtraj = np.zeros(n + 25, float)
trange = np.linspace(start=0, stop=tfinal, num=n + 1)
xtraj[0] = x0

# Menghasilkan square wave
for i in range(n):
    # Menggunakan fungsi sign dari sinus untuk membuat square wave
    xtraj[i + 1] = np.sign(np.sin(2 * np.pi * 10 * trange[i + 1]))  # 10 adalah frekuensi square wave

x = trange
y = xtraj

# animation line plot example
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

def animate(i):
    ax.cla()  # clear the previous image
    ax.plot(x[:i], y[:i])  # plot the line
    ax.set_xlim([x0, tfinal])  # fix the x axis
    ax.set_ylim([1.1 * np.min(y), 1.1 * np.max(y)])  # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames=len(x) + 1, interval=1, blit=False)
plt.show()
