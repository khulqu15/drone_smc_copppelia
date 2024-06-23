import numpy as np
import matplotlib.pyplot as plt

# Define the bounds of the random vectors
min_value = -1.0
max_value = 1.0

square_wave_half_period_seconds = 30
block_duration_seconds = 5 * 60  # 5 minutes
simulation_duration_seconds = 1 * 60 * 60  # 1 hour * 60 minutes/hour * 60 seconds/minute

# Initialize the load and renewable profiles vectors
ys = np.zeros(simulation_duration_seconds)

# Generate the random bounded vectors
for t_seconds in range(simulation_duration_seconds):
    if t_seconds % (block_duration_seconds) == 0:  # Start a new square value
        amplitude = np.random.uniform(min_value, max_value)

    am_in_first_half_of_square_wave = (t_seconds // square_wave_half_period_seconds) % 2 == 0
    ys[t_seconds] = amplitude * (1 if am_in_first_half_of_square_wave else -1)
    ys[t_seconds] += np.sin(t_seconds * 2 * np.pi) * 0.15 * 1

    # Clip the values to the defined bounds
    ys[t_seconds] = np.clip(ys[t_seconds], min_value, max_value)
   

plt.plot(ys)
plt.show()