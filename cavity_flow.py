from Simple import CavitySIMPLE
import numpy as np
import matplotlib.pyplot as plt

# 定义计算域和参数
nx, ny = 200, 200
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
cfl = 0.005
dt = cfl * min(dx, dy)  # 时间步长
Re = 400  # 雷诺数
U_top = 1.0

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
    alpha_u=1,  # 速度欠松弛因子
    alpha_p=0.8,  # 压力欠松弛因子
    max_iter=500,
    tol=1e-5
)

# 执行计算
cavity.solve()

# 获取结果
u, v, p = cavity.get_center_velocity()
psi = cavity.calculate_streamfunction()