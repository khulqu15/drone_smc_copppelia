import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

amplitude = 24
iteration = 0
while True:
    square_wave = amplitude * np.sign(np.sin(2 * np.pi * i / 20))
    
    iteration += 1