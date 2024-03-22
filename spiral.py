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
quadcopter_orientations = {'roll': [[], [], [], [], []],
                           'pitch': [[], [], [], [], []],
                           'yaw': [[], [], [], [], []]}

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

def spiral_trajectory(t, base_x, base_y, quadcopter_index, z_factor=0.05, radius_factor=1.5, angle_speed=0.2, z_start=0, z_end=6):
    angle = t * angle_speed
    x = base_x + math.cos(angle) * radius_factor
    y = base_y + math.sin(angle) * radius_factor
    z = 0.01 + z_factor * t
    
    z_increment = min(t * z_factor, z_end - z_start)
    z = z_start + z_increment - 2
    return x, y, z

bases = [(0.225, 0.250), (-0.250, 0.700), (0.725, 0.725), (0.250, -0.250), (0.725, 0.250)]

t = 0
while t < 100:
    for i, handle in enumerate(quadcopter_handles):
        base_x, base_y = bases[i]
        x, y, z = spiral_trajectory(t, base_x, base_y, i)
        orientation = sim.getObjectOrientation(handle, -1)
        roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
        pos = sim.getObjectPosition(handle, -1)
        e_x = x - pos[0]
        e_y = y - pos[1]
        e_z = z - pos[2]
        
        e_dot_x = e_dot_y = e_dot_z = 0
        
        control_signal_x = controllers['x'].control_law(e_x, e_dot_x)
        control_signal_y = controllers['y'].control_law(e_y, e_dot_y)
        control_signal_z = controllers['z'].control_law(e_z, e_dot_z)
        quadcopter_orientations['roll'][i].append(roll)
        quadcopter_orientations['pitch'][i].append(pitch)
        quadcopter_orientations['yaw'][i].append(yaw)
        
        sim.setObjectPosition(handle, -1, [x + control_signal_x, y + control_signal_y, z + control_signal_z])
        
        quadcopter_paths[i][0].append(x + control_signal_x)
        quadcopter_paths[i][1].append(y + control_signal_y)
        quadcopter_paths[i][2].append(z + control_signal_z)

        print(f'{t}: Quadcopter {i} position: {x + control_signal_x}, {y + control_signal_y}, {z + control_signal_z}')
    
    t += 0.3
    time.sleep(0.01)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
labels = ['Quadcopter 1', 'Quadcopter 2', 'Quadcopter 3', 'Quadcopter 4', 'Quadcopter 5']
colors = ['r', 'g', 'b', 'y', 'c']
for i in range(5):
    xs, ys, zs = quadcopter_paths[i]
    ax.plot(xs, ys, zs, c=colors[i], label=labels[i])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
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
