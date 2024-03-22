from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import pandas as pd
import math
import time


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

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_error = float('inf')

class PSO:
    def __init__(self, objective_function, bounds, num_particles, max_iter, desired_position, desired_orientation):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.desired_position = desired_position
        self.desired_orientation = desired_orientation
        
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_error = float('inf')
        
    def run(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                current_error = self.objective_function(particle.position, self.desired_position, self.desired_orientation)
                
                if current_error < particle.best_error:
                    particle.best_error = current_error
                    particle.best_position = particle.position
                
                if current_error < self.global_best_error:
                    self.global_best_error = current_error
                    self.global_best_position = particle.position
            
            for particle in self.particles:
                particle.velocity = 0.5 * particle.velocity + 0.2 * (particle.best_position - particle.position) + 0.3 * (self.global_best_position - particle.position)
                particle.position += particle.velocity

def objective_function(parameters, desired_position, desired_orientation):
    error_pos = np.sqrt((current_x - desired_x)**2 + (current_y - desired_y)**2 + (current_z - desired_z)**2)
    error_ori = np.linalg.norm(np.array(current_orientations) - np.array(desired_orientation))
    return error_pos + error_ori  

class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.005 # Process noise
        self.R = np.eye(2) * 1 # Measurement noise
        self.x = np.zeros((4, 1)) # State vector
        self.P = np.eye(4) # Covariance matrix
        
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
    'x': SMCController(k=0.1, lambda_=1.0, alpha=0.1, rho=7),
    'y': SMCController(k=0.1, lambda_=1.0, alpha=0.1, rho=7),
    'z': SMCController(k=2.0, lambda_=2.0, alpha=0.1, rho=10)
}
bounds = [(0.1, 2.0), (0.1, 2.0), (0.01, 0.5), (1.0, 20.0)] 
num_particles = 10
max_iter = 100
pso = PSO(objective_function, bounds, num_particles, max_iter, [desired_x, desired_y, desired_z], [0.0, 0.0, 0.0])

predictions = [ExtendedKalmanFilter(0.3) for _ in range(5)]

def calculate_box_position(t, base_x, base_y, quadcopter_index, z_factor=0.2, side_length=7.5, z_start=0, z_end=3):
    segment_length = side_length / 4
    segment = (t % side_length) / segment_length
    
    if segment < 1:
        x = base_x + segment * segment_length
        y = base_y
    elif segment < 2:
        x = base_x + segment_length
        y = base_y + (segment - 1) * segment_length
    elif segment < 3:
        x = base_x + segment_length - (segment - 2) * segment_length
        y = base_y + segment_length
    else:
        x = base_x
        y = base_y + segment_length - (segment - 3) * segment_length
    
    z = 0.01 + z_factor * t
    z_increment = min(t * z_factor, z_end - z_start)
    z = z_start + z_increment - 2
    return x, y, z

bases = [(0.225, 0.250), (-0.250, 0.700), (0.725, 0.725), (0.250, -0.250), (0.725, 0.250)]

t = 0
while t < 30:
    for i, handle in enumerate(quadcopter_handles):
        base_x, base_y = bases[i]
        x, y, z = calculate_box_position(t, base_x, base_y, i)
        
        pso.run()
        optimized_params = pso.global_best_position
        controllers['x'].k, controllers['x'].lambda_, controllers['x'].alpha, controllers['x'].rho = optimized_params
        
        orientation = sim.getObjectOrientation(handle, -1)
        roll, pitch, yaw = orientation[0], orientation[1], orientation[2]        
        pos = sim.getObjectPosition(handle, -1)
        
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
        
        sim.setObjectPosition(handle, -1, [estimated_position[0] + control_signal_x, 
                                           estimated_position[1] + control_signal_y, 
                                           z + control_signal_z
        ])
        
        quadcopter_paths[i][0].append(estimated_position[0] + control_signal_x)
        quadcopter_paths[i][1].append(estimated_position[1] + control_signal_y)
        quadcopter_paths[i][2].append(z + control_signal_z)

        predictions_paths[i][0].append(estimated_position[0] + control_signal_x)
        predictions_paths[i][1].append(estimated_position[1] + control_signal_y)

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
