from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import pandas as pd
import math
import time
import random

client = RemoteAPIClient()
sim = client.getObject('sim')
clientID = sim.startSimulation()
quadcopter_handles = [sim.getObject(f'/Quadcopter[{i}]') for i in range(5)]

quadcopter_paths = [[[], [], []] for _ in range(5)]
predictions_paths = [[[], []] for _ in range(5)]
quadcopter_orientations = {
    'roll': [[], [], [], [], []],
    'pitch': [[], [], [], [], []],
    'yaw': [[], [], [], [], []]
}

desired_x, desired_y, desired_z = 0.225, 0.250, 0.01
current_x, current_y, current_z = 0.225, 0.250, 0.01
current_orientations = [0.00000, 0.00000, 0.00000]
target_orientations = [0.00000, 0.00000, 0.00000]

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, generations, parameter_bounds):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.parameter_bounds = parameter_bounds
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.rand(self.population_size, self.chromosome_length)

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = np.random.rand()
        return chromosome

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.chromosome_length - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1, parent2

    def select_parent(self, fitness):
        normalized_fitness = fitness / np.sum(fitness)  # Normalisasi fitness
        idx = np.random.choice(np.arange(self.population_size), p=normalized_fitness)  # Gunakan fitness yang dinormalisasi
        return self.population[idx]

    def calculate_fitness(self, individual, current_positions, current_orientations, target_positions, target_orientations, lambda_weight=1.0, epsilon=1e-6):
        total_fitness = 0
        for i, (current_position, current_orientation) in enumerate(zip(current_positions, current_orientations)):
            new_position = current_position  # Asumsikan simulasi posisi berdasarkan individu (parameter)
            target_position = target_positions[i]
            error_pos = np.sqrt(np.sum((np.array(new_position) - np.array(target_position))**2))

            error_ori = np.abs(current_orientation - target_orientations[i])
            
            fitness = 1.0 / (error_pos + lambda_weight * error_ori + epsilon)
            total_fitness += fitness
        return total_fitness / len(current_positions)  # Rata-rata fitness

    def run(self):
        for generation in range(self.generations):
            fitness = np.array([self.calculate_fitness(individual, [current_x, current_y, current_z], current_orientations, [desired_x, desired_y, desired_z], [0.0, 0.0, 0.0]) for individual in self.population])
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.select_parent(fitness)
                parent2 = self.select_parent(fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = np.array(new_population)
            
class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)
        
    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P

class SMCController:
    def __init__(self, k, lambda_, alpha, rho):
        self.k = k
        self.lambda_ = lambda_
        self.alpha = alpha
        self.rho = rho

    def control_law(self, e, e_dot):
        s = e_dot + self.lambda_ * e
        return -self.k * np.sign(s) - self.alpha * np.tanh(self.rho * s)

controllers = {
    'x': SMCController(k=0.8, lambda_=2.0, alpha=0.1, rho=10),
    'y': SMCController(k=0.7, lambda_=2.0, alpha=0.1, rho=10),
    'z': SMCController(k=2.0, lambda_=2.0, alpha=0.1, rho=10)
}

predictions = [ExtendedKalmanFilter(0.3) for _ in range(5)]
parameter_boundaries = [(0.1, 2.0), (0.1, 2.0), (0.01, 0.5), (1.0, 20.0)]
generic = GeneticAlgorithm(10, 4, 0.1, 0.8, 100, parameter_boundaries)

def spiral_trajectory(t, base_x, base_y, quadcopter_index, z_factor=0.09, radius_factor=0.09, angle_speed=0.2, z_start=0, z_end=6):
    angle = t * angle_speed
    x = base_x + math.cos(angle) * radius_factor
    y = base_y + math.sin(angle) * radius_factor
    z = 0.01 + z_factor * t
    z_increment = min(t * z_factor, z_end - z_start)
    z = z_start + z_increment - 2
    return x, y, z

bases = [(0.225, 0.250), (-0.250, 0.700), (0.725, 0.725), (0.250, -0.250), (0.725, 0.250)]

t = 0
while t < 10:
    quadcopter_orientations_generic = []
    for i, handle in enumerate(quadcopter_handles):

        base_x, base_y = bases[i]
        x, y, z = spiral_trajectory(t, base_x, base_y, i)
        desired_x, desired_y, desired_z = x, y, z
        
        orientation = sim.getObjectOrientation(handle, -1)
        roll, pitch, yaw = orientation[0], orientation[1], orientation[2]        
        pos = sim.getObjectPosition(handle, -1)
        current_x, current_y, current_z = pos[0], pos[1], pos[2]
        current_orientations = [roll, pitch, yaw]
        
        generic.run()
        optimized_params = generic.population[np.argmax([generic.calculate_fitness(individual, [pos[0], pos[1], pos[2]], orientation, [x, y, z], [0.0, 0.0, 0.0]) for individual in generic.population])]
        if t % 5 == 0: controllers['x'].k, controllers['x'].lambda_, controllers['x'].alpha, controllers['x'].rho = optimized_params
        
        predictions[i].predict()
        Z = np.array([[x], [y]])
        predictions[i].update(Z)
        estimated_position = predictions[i].x[:2].flatten()
        
        e_x = x - estimated_position[0]
        e_y = y - estimated_position[1]
        e_z = z - pos[2]
        
        e_dot_x = e_dot_y = e_dot_z = 0
        
        control_signal_x = controllers['x'].control_law(e_x, e_dot_x)
        control_signal_y = controllers['y'].control_law(e_y, e_dot_y)
        control_signal_z = controllers['z'].control_law(e_z, e_dot_z)
        quadcopter_orientations['roll'][i].append(roll)
        quadcopter_orientations['pitch'][i].append(pitch)
        quadcopter_orientations['yaw'][i].append(yaw)
        
        sim.setObjectPosition(handle, -1, [(estimated_position[0]) + control_signal_x, 
                                           (estimated_position[1]) + control_signal_y, 
                                           z + control_signal_z
        ])
        
        quadcopter_paths[i][0].append(estimated_position[0] + control_signal_x)
        quadcopter_paths[i][1].append(estimated_position[1] + control_signal_y)
        quadcopter_paths[i][2].append(z + control_signal_z)

        predictions_paths[i][0].append(estimated_position[0] + control_signal_x)
        predictions_paths[i][1].append(estimated_position[1] + control_signal_y)

        print(f'{t}: Parameters for Quadcopter {i + 1}: {optimized_params}')
        print(f'{t}: Quadcopter {i} position: {x + control_signal_x}, {y + control_signal_y}, {z + control_signal_z}')
    
    t += 0.3
    time.sleep(0.01)

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')
labels = ['Quadcopter 1', 'Quadcopter 2', 'Quadcopter 3', 'Quadcopter 4', 'Quadcopter 5']
colors = ['r', 'g', 'b', 'y', 'c']
for i in range(5):
    xs, ys, zs = quadcopter_paths[i]
    ax.plot(xs, ys, zs, c=colors[i], label=labels[i])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()

ay = fig.add_subplot(122, projection='3d')
for i in range(5):
    xs, ys = predictions_paths[i]
    zs = quadcopter_paths[i][2]
    ay.plot(xs, ys, zs, c=colors[i], linestyle='dashed')
ay.set_xlabel('X (m)')
ay.set_ylabel('Y (m)')
ay.set_zlabel('Z (m)')
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
orientations = ['roll', 'pitch', 'yaw']
for i, orientation in enumerate(orientations):
    for quadcopter_index in range(5):
        axs[i].plot(quadcopter_orientations[orientation][quadcopter_index], label=f'Quadcopter {quadcopter_index + 1}')
    axs[i].set_title(f'{orientation.capitalize()} Orientation over Time')
    axs[i].set_xlabel('Time step')
    axs[i].set_ylabel(f'{orientation.capitalize()} (degrees)')
    axs[i].legend()

plt.tight_layout()
plt.show()

with pd.ExcelWriter('spiral_trajectory.xlsx', engine='openpyxl') as writer:
    for i in range(len(quadcopter_handles)):
        df_position = pd.DataFrame({
            'X': quadcopter_paths[i][0],
            'Y': quadcopter_paths[i][1],
            'Z': quadcopter_paths[i][2]
        })
        df_orientation = pd.DataFrame({
            'Roll': quadcopter_orientations['roll'][i],
            'Pitch': quadcopter_orientations['pitch'][i],
            'Yaw': quadcopter_orientations['yaw'][i]
        })
        df_quadcopter = pd.concat([df_position, df_orientation], axis=1)
        sheet_name = f'Quadcopter {i + 1}'
        df_quadcopter.to_excel(writer, sheet_name=sheet_name, index=False)

sim.stopSimulation()
