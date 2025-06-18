import numpy as np
import matplotlib.pyplot as plt
from vorStream import VorticityStreamPoiseuille
import os

# 参数设置
Lx, Ly = 100.0, 1.0       # 流域尺寸
nx, ny = 1000, 150           # 网格数量
x = np.linspace(0, Lx, nx, endpoint=True)
y = np.linspace(0, Ly, ny, endpoint=True)
dx = x[1] - x[0]
dy = y[1] - y[0]
cfl = 0.005
dt = min(cfl * dx, cfl * dy)                # 时间步长
nu = 1                    # 运动粘度
U0 = 1.0                  # 中心线速度
H = Ly                    # 管道高度
max_iter = 15000
tol = 1e-6

t = np.arange(0, 10, dt)   # 时间数组

# name and folder of the case
name = 'Poiseuille'
folder = 'Proj2\\vorStream'
if not os.path.exists(folder):
    os.makedirs(folder)
file_path = os.path.join(folder, f'{name}.png')

# create object
solver = VorticityStreamPoiseuille(
    name="PoiseuilleFlow",
    dt=dt,
    dx=dx,
    dy=dy,
    x=x,
    y=y,
    t=t,
    nu=nu,
    U0=U0,
    H=H
)

solver.solve(max_iter=max_iter, tol=tol)

u, v = solver.get_velocity_field()

u_outlet = u[-1, :]   # u profile at outlet
v_outlet = v[-1, :]   # v profile at outlet

# plot figure
plt.figure(figsize=(10, 6))

# plot u
plt.subplot(2, 1, 1)
plt.plot(y, u_outlet, 'b-o', linewidth=2, markersize=4, label='u')
plt.grid(True)
plt.title('Downstream Velocity Profile (x = L)')
plt.ylabel('u Velocity')
plt.legend()
plt.xlim(min(y), max(y))

# plot v
plt.subplot(2, 1, 2)
plt.plot(y, v_outlet, 'r-o', linewidth=2, markersize=4, label='v')
plt.grid(True)
plt.ylabel('v Velocity')
plt.xlabel('y Coordinate')
plt.legend()
plt.xlim(min(y), max(y))

plt.tight_layout()
plt.savefig(file_path)

# optional: plot together
plt.figure(figsize=(10, 5))
plt.plot(y, u_outlet, 'b-o', label='u', markersize=4)
plt.plot(y, v_outlet, 'r-o', label='v', markersize=4)
plt.grid(True)
plt.title('Downstream Velocity Profile (x = L)')
plt.xlabel('y Coordinate')
plt.ylabel('Velocity Magnitude')
plt.legend()
plt.xlim(min(y), max(y))
plt.tight_layout()
plt.savefig(f'{file_path}_together.png')