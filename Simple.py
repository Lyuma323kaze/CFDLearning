from Diff_schme import Diffschemes
import numpy as np

class CavitySIMPLE(DiffSchemes):
    def __init__(self, name, dt, dx, x, t, dy, y, nu, rho, U_top, 
                 max_iter=1000, tol=1e-5, **kwargs):
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, **kwargs)
        self.nu = nu          # 运动粘度
        self.rho = rho        # 密度
        self.U_top = U_top    # 顶盖速度
        
        # 网格参数
        self.nx = len(x)      # x方向网格点数
        self.ny = len(y)      # y方向网格点数
        
        # 交错网格定义
        # 压力网格 (中心网格)
        self.p = np.zeros((self.nx, self.ny))
        
        # u速度分量 (位于垂直面中心)
        self.u = np.zeros((self.nx+1, self.ny+2))
        
        # v速度分量 (位于水平面中心)
        self.v = np.zeros((self.nx+2, self.ny+1))
        
        # 临时速度和压力修正
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.p_prime = np.zeros((self.nx, self.ny))
        
        # 收敛控制
        self.max_iter = max_iter
        self.tol = tol

    def apply_boundary_conditions(self):
        """应用方腔驱动流的边界条件"""
        # 顶盖移动 (u速度)
        self.u[:, -1] = 2 * self.U_top - self.u[:, -2]  # 滑移边界条件
        
        # 底部固定
        self.u[:, 0] = -self.u[:, 1]      # u=0
        self.v[1:-1, 0] = -self.v[1:-1, 1] # v=0
        
        # 左侧固定
        self.u[0, :] = -self.u[1, :]      # u=0
        self.v[0, :] = -self.v[1, :]      # v=0
        
        # 右侧固定
        self.u[-1, :] = -self.u[-2, :]    # u=0
        self.v[-1, :] = -self.v[-2, :]    # v=0

    def solve_momentum_u(self):
        """求解u方向动量方程"""
        for i in range(1, self.nx):
            for j in range(1, self.ny+1):
                # 对流项 (迎风格式)
                ue = 0.5 * (self.u[i,j] + self.u[i+1,j])
                uw = 0.5 * (self.u[i,j] + self.u[i-1,j])
                vn = 0.5 * (self.v[i,j] + self.v[i+1,j])
                vs = 0.5 * (self.v[i,j] + self.v[i+1,j-1])
                
                # 扩散项系数
                A_e = -self.nu * self.dt / self.dx**2
                A_w = -self.nu * self.dt / self.dx**2
                A_n = -self.nu * self.dt / self.dy**2
                A_s = -self.nu * self.dt / self.dy**2
                
                # 主对角系数
                A_p = 1 - (A_e + A_w + A_n + A_s)
                
                # 压力梯度项
                dP = self.p[i-1,j-1] - self.p[i,j-1]
                
                # 更新u_star
                self.u_star[i,j] = (self.u[i,j] * A_p + 
                                    A_e * self.u[i+1,j] + 
                                    A_w * self.u[i-1,j] +
                                    A_n * self.u[i,j+1] +
                                    A_s * self.u[i,j-1] +
                                    self.dt * dP / (self.rho * self.dx))

    def solve_momentum_v(self):
        """求解v方向动量方程"""
        for i in range(1, self.nx+1):
            for j in range(1, self.ny):
                # 对流项 (迎风格式)
                ue = 0.5 * (self.u[i,j] + self.u[i,j+1])
                uw = 0.5 * (self.u[i-1,j] + self.u[i-1,j+1])
                vn = 0.5 * (self.v[i,j] + self.v[i,j+1])
                vs = 0.5 * (self.v[i,j] + self.v[i,j-1])
                
                # 扩散项系数
                A_e = -self.nu * self.dt / self.dx**2
                A_w = -self.nu * self.dt / self.dx**2
                A_n = -self.nu * self.dt / self.dy**2
                A_s = -self.nu * self.dt / self.dy**2
                
                # 主对角系数
                A_p = 1 - (A_e + A_w + A_n + A_s)
                
                # 压力梯度项
                dP = self.p[i-1,j] - self.p[i-1,j-1]
                
                # 更新v_star
                self.v_star[i,j] = (self.v[i,j] * A_p + 
                                    A_e * self.v[i+1,j] + 
                                    A_w * self.v[i-1,j] +
                                    A_n * self.v[i,j+1] +
                                    A_s * self.v[i,j-1] +
                                    self.dt * dP / (self.rho * self.dy))

    def solve_pressure_correction(self):
        """求解压力修正方程"""
        # 初始化源项和系数
        b = np.zeros((self.nx, self.ny))
        A_p = np.zeros((self.nx, self.ny))
        
        # 构建压力修正方程
        for i in range(self.nx):
            for j in range(self.ny):
                # 连续性源项
                b[i,j] = self.rho * (
                    (self.u_star[i,j] - self.u_star[i+1,j]) / self.dx +
                    (self.v_star[i,j] - self.v_star[i,j+1]) / self.dy
                )
                
                # 系数计算
                A_e = self.dt / (self.rho * self.dx**2)
                A_w = self.dt / (self.rho * self.dx**2)
                A_n = self.dt / (self.rho * self.dy**2)
                A_s = self.dt / (self.rho * self.dy**2)
                A_p[i,j] = -(A_e + A_w + A_n + A_s)
        
        # 迭代求解压力修正 (Jacobi方法)
        for _ in range(50):  # 内部迭代次数
            p_prime_new = np.copy(self.p_prime)
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    p_prime_new[i,j] = (
                        b[i,j] - 
                        A_w * self.p_prime[i-1,j] -
                        A_e * self.p_prime[i+1,j] -
                        A_s * self.p_prime[i,j-1] -
                        A_n * self.p_prime[i,j+1]
                    ) / A_p[i,j]
            self.p_prime = p_prime_new

    def correct_velocity_pressure(self):
        """修正速度和压力"""
        # 压力修正
        self.p += 0.1 * self.p_prime  # 欠松弛
        
        # u速度修正
        for i in range(1, self.nx):
            for j in range(1, self.ny+1):
                dP_prime = self.p_prime[i,j-1] - self.p_prime[i-1,j-1]
                self.u[i,j] = self.u_star[i,j] + self.dt * dP_prime / (self.rho * self.dx)
        
        # v速度修正
        for i in range(1, self.nx+1):
            for j in range(1, self.ny):
                dP_prime = self.p_prime[i-1,j] - self.p_prime[i-1,j-1]
                self.v[i,j] = self.v_star[i,j] + self.dt * dP_prime / (self.rho * self.dy)

    def solve(self):
        """执行SIMPLE算法主循环"""
        for iter in range(self.max_iter):
            # 保存上一步的速度场
            u_old = np.copy(self.u)
            v_old = np.copy(self.v)
            
            # 应用边界条件
            self.apply_boundary_conditions()
            
            # SIMPLE步骤
            self.solve_momentum_u()        # 求解u*
            self.solve_momentum_v()         # 求解v*
            self.solve_pressure_correction()# 求解p'
            self.correct_velocity_pressure()# 修正速度和压力
            
            # 检查收敛性
            u_res = np.max(np.abs(self.u - u_old))
            v_res = np.max(np.abs(self.v - v_old))
            
            if iter % 100 == 0:
                print(f"Iteration {iter}, U_res: {u_res:.4e}, V_res: {v_res:.4e}")
            
            if u_res < self.tol and v_res < self.tol:
                print(f"Converged at iteration {iter}")
                break

    def get_center_velocity(self):
        """获取网格中心的速度场"""
        u_center = 0.5 * (self.u[1:, 1:-1] + self.u[:-1, 1:-1])
        v_center = 0.5 * (self.v[1:-1, 1:] + self.v[1:-1, :-1])
        return u_center, v_center