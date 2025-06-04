import numpy as np
import matplotlib.pyplot as plt
from vorStream import VorticityStreamPoiseuille

# 参数设置
Lx, Ly = 50.0, 1.0       # 流域尺寸
nx, ny = 300, 200           # 网格数量
dx, dy = Lx/(nx-1), Ly/(ny-1)
cfl = 0.05
dt = min(cfl * dx, cfl * dy)                # 时间步长
nu = 0.1                  # 运动粘度
U0 = 1.0                  # 中心线速度
H = Ly                    # 管道高度
max_iter = 100000
tol = 1e-5

# 创建坐标网格
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.arange(0, 10, dt)   # 时间数组

# 创建求解器实例
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

# 求解
solver.solve(max_iter=max_iter, tol=tol)

# 获取速度场
u, v = solver.get_velocity_field()

u_outlet = u[-1, :]   # 最后一个x位置的所有y点的u速度
v_outlet = v[-1, :]   # 最后一个x位置的所有y点的v速度

# 创建绘图区域
plt.figure(figsize=(10, 6))

# 绘制u速度分量
plt.subplot(2, 1, 1)
plt.plot(y, u_outlet, 'b-o', linewidth=2, markersize=4, label='u')
plt.grid(True)
plt.title('Downstream Velocity Profile (x = L)')
plt.ylabel('u Velocity')
plt.legend()
plt.xlim(min(y), max(y))

# 绘制v速度分量
plt.subplot(2, 1, 2)
plt.plot(y, v_outlet, 'r-o', linewidth=2, markersize=4, label='v')
plt.grid(True)
plt.ylabel('v Velocity')
plt.xlabel('y Coordinate')
plt.legend()
plt.xlim(min(y), max(y))

plt.tight_layout()
plt.show()

# # 可选：将两个速度分量绘制在同一个坐标系中（便于比较）
# plt.figure(figsize=(10, 5))
# plt.plot(y, u_outlet, 'b-o', label='u', markersize=4)
# plt.plot(y, v_outlet, 'r-o', label='v', markersize=4)
# plt.grid(True)
# plt.title('Downstream Velocity Profile (x = L)')
# plt.xlabel('y Coordinate')
# plt.ylabel('Velocity Magnitude')
# plt.legend()
# plt.xlim(min(y), max(y))
# plt.tight_layout()
# plt.show()