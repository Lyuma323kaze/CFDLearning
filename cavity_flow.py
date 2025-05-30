from Simple import CavitySIMPLE
import numpy as np
import matplotlib.pyplot as plt

# 定义计算域和参数
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = 0.001
Re = 1000  # 雷诺数
U_top = 1.0
nu = U_top * 1.0 / Re  # 运动粘度 (特征长度=1)

# 初始化求解器
cavity = CavitySIMPLE(
    name="CavityFlow",
    dt=dt,
    dx=dx,
    x=x,
    t=np.arange(0, 10, dt),
    dy=dy,
    y=y,
    nu=nu,
    rho=1.0,
    U_top=U_top,
    alpha_u=0.5,  # 速度欠松弛因子
    alpha_p=0.2,  # 压力欠松弛因子
    max_iter=500,
    tol=1e-5
)

# 执行计算
cavity.solve()

# 获取结果
u, v, p = cavity.get_center_velocity()
psi = cavity.calculate_streamfunction()