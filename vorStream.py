import numpy as np
from Diff_schme import DiffSchemes


class VorticityStreamPoiseuille(DiffSchemes):
    def __init__(self, name, dt, dx, dy, x, y, t, nu, U0, H, 
                 ini_condi=None, bnd_condi=None, folder=None):
        """
        涡量-流函数方法求解泊肃叶流动
        :param nu: 运动粘度
        :param U0: 中心线速度
        :param H: 管道高度（y方向）
        """
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, 
                         ini_condi=ini_condi, bnd_condi=bnd_condi, folder=folder)
        self.nu = nu        # 运动粘度
        self.U0 = U0        # 中心线最大速度
        self.H = H          # 管道高度
        self.ny = len(y)    # y方向网格数
        
        # 初始化流函数和涡量场
        self.psi = None
        self.vorticity = None
        self.initialize_fields()
        
        # 设置边界条件
        self.set_boundary_conditions()

    def initialize_fields(self):
        """初始化流函数和涡量场"""
        nx, ny = len(self.x), len(self.y)
        self.psi = np.zeros((nx, ny))
        self.vorticity = np.zeros((nx, ny))
        
        # 初始条件：抛物线速度剖面
        for j in range(ny):
            y_val = self.y[j]
            u = self.U0 * (1 - (y_val - 0.5*self.H)**2 / (0.5*self.H)**2)
            self.psi[:, j] = np.cumsum([u * self.dy] * nx)  # 积分得到流函数初值

    def set_boundary_conditions(self):
        """设置二维泊肃叶流动的边界条件"""
        # 顶底边界 (无滑移条件)
        for i in range(len(self.x)):
            # 底部壁面 (y=0)
            self.psi[i, 0] = 0
            self.vorticity[i, 0] = -2 * self.psi[i, 1] / self.dy**2
            
            # 顶部壁面 (y=H)
            self.psi[i, -1] = 0
            self.vorticity[i, -1] = -2 * self.psi[i, -2] / self.dy**2
        
        # 入口边界 (给定抛物线剖面)
        for j in range(1, self.ny-1):
            y_val = self.y[j]
            u_in = self.U0 * (1 - (y_val - 0.5*self.H)**2 / (0.5*self.H)**2)
            self.psi[0, j] = u_in * self.y[j]  # 流函数入口条件
            
            # 涡量入口条件 (∂u/∂y)
            if j == 0:
                dudy = (u_in - 0) / self.dy
            elif j == self.ny-1:
                dudy = (0 - u_in) / self.dy
            else:
                u_above = self.U0 * (1 - (self.y[j+1] - 0.5*self.H)**2 / (0.5*self.H)**2)
                u_below = self.U0 * (1 - (self.y[j-1] - 0.5*self.H)**2 / (0.5*self.H)**2)
                dudy = (u_above - u_below) / (2 * self.dy)
            self.vorticity[0, j] = -dudy
        
        # 出口边界 (充分发展流)
        self.vorticity[-1, :] = self.vorticity[-2, :]  # ∂ω/∂x = 0
        self.psi[-1, :] = 2 * self.psi[-2, :] - self.psi[-3, :]  # ∂²ψ/∂x² = 0

    def solve(self, max_iter=10000, tol=1e-6):
        """求解稳态涡量-流函数方程"""
        for iter in range(max_iter):
            psi_old = self.psi.copy()
            
            # 1. 求解涡量输运方程
            self.solve_vorticity_transport()
            
            # 2. 求解泊松方程 (∇²ψ = -ω)
            self.solve_poisson()
            
            # 3. 更新边界条件
            self.set_boundary_conditions()
            
            # 检查收敛性
            diff = np.max(np.abs(self.psi - psi_old))
            if diff < tol:
                print(f"Converged after {iter} iterations")
                break
        else:
            print("Reached maximum iterations")

    def solve_vorticity_transport(self):
        """求解涡量输运方程 (显式方法)"""
        new_vort = np.zeros_like(self.vorticity)
        
        for i in range(1, len(self.x)-1):
            for j in range(1, self.ny-1):
                # 计算速度分量 (u = ∂ψ/∂y, v = -∂ψ/∂x)
                u = (self.psi[i, j+1] - self.psi[i, j-1]) / (2 * self.dy)
                v = -(self.psi[i+1, j] - self.psi[i-1, j]) / (2 * self.dx)
                
                # 涡量导数
                dω_dx = (self.vorticity[i+1, j] - self.vorticity[i-1, j]) / (2 * self.dx)
                dω_dy = (self.vorticity[i, j+1] - self.vorticity[i, j-1]) / (2 * self.dy)
                
                # 扩散项
                laplacian_ω = (self.vorticity[i+1, j] - 2*self.vorticity[i, j] + self.vorticity[i-1, j]) / self.dx**2 + \
                              (self.vorticity[i, j+1] - 2*self.vorticity[i, j] + self.vorticity[i, j-1]) / self.dy**2
                
                # 涡量输运方程: ∂ω/∂t + u·∇ω = ν∇²ω
                # 稳态: u·∇ω = ν∇²ω
                new_vort[i, j] = self.vorticity[i, j] + self.dt * (
                    self.nu * laplacian_ω - (u * dω_dx + v * dω_dy)
                )
        
        # 更新内部点的涡量
        self.vorticity[1:-1, 1:-1] = new_vort[1:-1, 1:-1]

    def solve_poisson(self, max_iter=100, tol=1e-4):
        """使用迭代法求解泊松方程 ∇²ψ = -ω"""
        for _ in range(max_iter):
            psi_old = self.psi.copy()
            
            for i in range(1, len(self.x)-1):
                for j in range(1, self.ny-1):
                    # Jacobi迭代更新
                    self.psi[i, j] = 0.25 * (
                        self.psi[i+1, j] + self.psi[i-1, j] +
                        self.psi[i, j+1] + self.psi[i, j-1] +
                        self.dx**2 * self.vorticity[i, j]
                    )
            
            # 检查收敛
            if np.max(np.abs(self.psi - psi_old)) < tol:
                break

    def get_velocity_field(self):
        """从流函数计算速度场"""
        u = np.zeros_like(self.psi)
        v = np.zeros_like(self.psi)
        
        # 内部点
        for i in range(1, len(self.x)-1):
            for j in range(1, self.ny-1):
                u[i, j] = (self.psi[i, j+1] - self.psi[i, j-1]) / (2 * self.dy)
                v[i, j] = -(self.psi[i+1, j] - self.psi[i-1, j]) / (2 * self.dx)
        
        # 边界处理 (使用单边差分)
        u[:, 0] = (self.psi[:, 1] - self.psi[:, 0]) / self.dy
        u[:, -1] = (self.psi[:, -1] - self.psi[:, -2]) / self.dy
        v[0, :] = -(self.psi[1, :] - self.psi[0, :]) / self.dx
        v[-1, :] = -(self.psi[-1, :] - self.psi[-2, :]) / self.dx
        
        return u, v    