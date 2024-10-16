import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# UR5のDHパラメータ
dh_params = [
    {'a': 0, 'd': 0.089159, 'alpha': np.pi/2},
    {'a': -0.425, 'd': 0, 'alpha': 0},
    {'a': -0.39225, 'd': 0, 'alpha': 0},
    {'a': 0, 'd': 0.10915, 'alpha': np.pi/2},
    {'a': 0, 'd': 0.09465, 'alpha': -np.pi/2},
    {'a': 0, 'd': 0.0823, 'alpha': 0}
]

def forward_kinematics(joint_angles):
    T = np.eye(4)
    positions = [T[:3, 3]]
    for i, params in enumerate(dh_params):
        theta = joint_angles[i]
        a = params['a']
        d = params['d']
        alpha = params['alpha']
        
        T_i = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        T = np.dot(T, T_i)
        positions.append(T[:3, 3])
    return np.array(positions)

def update(frame, lines, joint_angles):
    angles = joint_angles * frame / 100
    positions = forward_kinematics(angles)
    for i in range(len(lines)):
        lines[i].set_data(positions[:i+2, 0], positions[:i+2, 1])
        lines[i].set_3d_properties(positions[:i+2, 2])
    return lines

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)

# 初期ジョイント角度
initial_angles = np.zeros(6)
positions = forward_kinematics(initial_angles)
lines = [ax.plot(positions[:i+2, 0], positions[:i+2, 1], positions[:i+2, 2], 'o-')[0] for i in range(len(positions)-1)]

ani = FuncAnimation(fig, update, frames=100, fargs=(lines, np.pi/2 * np.ones(6)), blit=False)

# アニメーションを保存
ani.save('ur5_animation.mp4', writer='ffmpeg', fps=30)

# plt.show()