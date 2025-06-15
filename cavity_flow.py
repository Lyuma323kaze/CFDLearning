from Simple import CavitySIMPLE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

# name and folder of the case
name = 'cavity_flow'
folder = 'Proj2\\SIMPLE'
if not os.path.exists(folder):
    os.makedirs(folder)
file_path = os.path.join(folder, f'{name}.png')


# domain and computational parameters
nx, ny = 200, 200
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
cfl = 1e-1
dt = cfl * min(dx, dy)  # time step
Re = 1e-8  # Reynolds number 
U_top = 1.0
alpha_u = 0.8     # velocity relaxation factor
alpha_p = 0.8   # pressure relaxation factor
max_iter = 2000
tol = 1e-10

# 初始化求解器
cavity = CavitySIMPLE(
    name="CavityFlow",
    dt=dt,
    dx=dx,
    x=x,
    t=np.arange(0, 10, dt),
    dy=dy,
    y=y,
    Re=Re,
    U_top=U_top,
    alpha_u=alpha_u,  
    alpha_p=alpha_p,  
    max_iter=max_iter,
    tol=tol
)

# solve
cavity.solve()

# get results
u, v, p = cavity.get_center_velocity()

# print(cavity.u)

x = np.linspace(0.5/(nx), 1-0.5/(nx), nx)  # 控制体中心x坐标
y = np.linspace(0.5/(ny), 1-0.5/(ny), ny)  # 控制体中心y坐标
X, Y = np.meshgrid(x, y, indexing='ij')  # 创建网格坐标

# 创建绘图区域
plt.figure(figsize=(12, 5))

# 流线图 - 左子图
plt.subplot(1, 2, 1)
# 绘制流线图
# u.shape = (nx,ny), v.shape = (nx,ny)
plt.streamplot(X.T, Y.T, u.T, v.T, 
               density=3, color='black', linewidth=1, arrowsize=1)
plt.title('Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')  # 确保坐标轴比例相等

# 压力图 - 右子图
plt.subplot(1, 2, 2)
# 绘制压力云图
contour = plt.contourf(X, Y, p, 20, cmap='coolwarm')
plt.colorbar(contour, label='Pressure')
plt.title('Pressure Contour')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')  # 确保坐标轴比例相等

plt.tight_layout()
plt.savefig(file_path)