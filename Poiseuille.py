import numpy as np
import matplotlib.pyplot as plt
from vorStream import VorticityStreamPoiseuille

# 参数设置
Lx, Ly = 10.0, 1.0       # 流域尺寸
nx, ny = 100, 100           # 网格数量
dx, dy = Lx/(nx-1), Ly/(ny-1)
cfl = 0.05
dt = min(cfl * dx, cfl * dy)                # 时间步长
nu = 0.1                  # 运动粘度
U0 = 1.0                  # 中心线速度
H = Ly                    # 管道高度

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
solver.solve(max_iter=1000, tol=1e-5)

# 获取速度场
u, v = solver.get_velocity_field()