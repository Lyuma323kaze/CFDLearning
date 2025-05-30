from Diff_schme import DiffSchemes
import numpy as np

class CavitySIMPLE(DiffSchemes):
    def __init__(self, name, dt, dx, x, t, dy, y, Re, U_top, 
                 max_iter=1000,
                 tol=1e-5,
                 alpha_u=0.7,
                 alpha_p=0.3,
                 **kwargs):
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, **kwargs)
        self.Re = Re          # 运动粘度
        self.U_top = U_top    # 顶盖速度
        
        # 网格参数
        self.nx = len(x)      # x方向网格点数
        self.ny = len(y)      # y方向网格点数
        self.dx = dx
        self.dy = dy
        # 欠松弛因子
        self.alpha_u = alpha_u  # 速度欠松弛
        self.alpha_p = alpha_p  # 压力欠松弛
        
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

        # 设置初始边界条件
        self.apply_boundary_conditions()

    def apply_boundary_conditions(self):
        """Set boundary values"""
        # 顶盖移动 (u速度)
        self.u[:, -1] = self.U_top  # 顶盖x方向速度
        self.v[:, -1] = 0.0         # 顶盖y方向速度
        
        # 底部固定 (无滑移)
        self.u[:, 0] = 0.0   # u=0
        self.v[:, 0] = 0.0   # v=0
        
        # 左侧固定 (无滑移)
        self.u[0, :] = 0.0   # u=0
        self.v[0, :] = 0.0   # v=0
        
        # 右侧固定 (无滑移)
        self.u[-1, :] = 0.0  # u=0
        self.v[-1, :] = 0.0  # v=0

    def solve_momentum_u(self, uworder=1, iter_u=50):
        """求解u方向动量方程"""
        for _ in range(iter_u):
            # upwind coefficients
            u_avr = np.empty((self.nx+1, self.ny+2))
            u_avr[:-1,:] = (self.u[:-1,:] + self.u[1:,:]) / 2 
            u_avr[-1,:] = self.u[-1,:]  # last row is the top boundary
            
            alpha_uxp = np.maximum(u_avr, 0)[:,1:-1]     # nx+1, ny
            alpha_uxm = np.minimum(u_avr, 0)[:,1:-1]     # nx+1, ny
            
            v_avr = (self.v[1:-1,:] + self.v[2:,:]) / 2
            
            alpha_uyp = np.maximum(v_avr, 0)     # nx, ny+1
            alpha_uym = np.minimum(v_avr, 0)     # nx, ny+1
            gamma_ux = np.zeros_like(alpha_uxp)  # nx+1, ny
            gamma_uy = np.zeros_like(alpha_uyp)  # nx, ny+1
            if uworder == 2:
                gamma_ux[1:-1] = 0.5 * (alpha_uxp[1:-1] * (self.u[1:-2,1:-1] - self.u[:-3,1:-1]) +
                                        alpha_uxm * (self.u[2:-1,1:-1] - self.u[3:,1:-1]))
                gamma_ux[0] = 0.5 * alpha_uxm[0] * (self.u[1,1:-1] - self.u[2,1:-1])
                gamma_ux[-1] = 0.5 * alpha_uxp[-1] * (self.u[-2,1:-1] - self.u[-3,1:-1])
                
                gamma_uy[:,1:-2] = 0.5 * (alpha_uyp[:,1:-2] * (self.u[1:,1:-2] - self.u[1:,:-3]) +
                                        alpha_uym * (self.u[1:,2:-1] - self.u[1:,3:]))
                gamma_uy[:,0] = 0.5 * alpha_uym[:,0] * (self.u[1:,1] - self.u[1:,2])
                gamma_uy[:,-1] = 0.5 * alpha_uyp[:,-1] * (self.u[1:,-1] - self.u[1:,-2])
                gamma_uy[:,-2] = 0.5 * alpha_uyp[:,-2] * (self.u[1:,-2] - self.u[1:,-3])
                
            # discretization coefficients
            a_p = (self.dx * self.dy / self.dt) +\
                    self.dy * (alpha_uxp[1:] - alpha_uxm[:-1] + (2 / (self.Re * self.dx))) +\
                    self.dx * (alpha_uyp[:,1:] - alpha_uyp[:,:-1] + (2 / (self.Re * self.dy)))
            a_w = self.dy * (alpha_uxp[:-1] + 1 / (self.Re * self.dx))
            a_e = self.dy * (-alpha_uxm[1:] + 1 / (self.Re * self.dx))
            a_s = self.dx * (alpha_uyp[:,:-1] + 1 / (self.Re * self.dy))
            a_n = self.dx * (-alpha_uym[:,1:] + 1 / (self.Re * self.dy))
            a_hat = self.dy * (gamma_ux[1:] - gamma_ux[:-1]) +\
                    self.dx * (gamma_uy[:,1:] - gamma_uy[:,:-1])
            
            # 压力梯度项
            dP = -(self.p[1:] - self.p[:-1]) * self.dy
            
            # 源项
            b = dP
            
            # 更新u_star (使用欠松弛)
            self.u_star = (1 - self.alpha_u) * self.u + self.alpha_u * (
                (a_e * self.u + 
                    a_w * self.u +
                    a_n * self.u +
                    a_s * self.u + b) / a_p
            )

    def solve_momentum_v(self, uworder=1):
        """求解v方向动量方程"""
        for i in range(1, self.nx+1):
            for j in range(1, self.ny):
                # 对流项 (迎风格式)
                u_e = 0.5 * (self.u[i, j] + self.u[i, j+1])
                u_w = 0.5 * (self.u[i-1, j] + self.u[i-1, j+1])
                v_n = 0.5 * (self.v[i, j] + self.v[i, j+1])
                v_s = 0.5 * (self.v[i, j] + self.v[i, j-1])
                
                # 对流项系数
                F_e = -0.5 * u_e * self.dy
                F_w = 0.5 * u_w * self.dy
                F_n = -0.5 * v_n * self.dx
                F_s = 0.5 * v_s * self.dx
                
                # 扩散项系数
                D_e = self.nu * self.dy / self.dx
                D_w = self.nu * self.dy / self.dx
                D_n = self.nu * self.dx / self.dy
                D_s = self.nu * self.dx / self.dy
                
                # 总系数
                a_e = D_e + max(-F_e, 0)
                a_w = D_w + max(F_w, 0)
                a_n = D_n + max(-F_n, 0)
                a_s = D_s + max(F_s, 0)
                
                # 主对角系数
                a_p = a_e + a_w + a_n + a_s + (F_e - F_w + F_n - F_s)
                
                # 压力梯度项
                dP = (self.p[i-1, j] - self.p[i-1, j-1]) * self.dx
                
                # 源项
                b = dP
                
                # 更新v_star (使用欠松弛)
                self.v_star[i, j] = (1 - self.alpha_u) * self.v[i, j] + self.alpha_u * (
                    (a_e * self.v[i+1, j] + 
                     a_w * self.v[i-1, j] +
                     a_n * self.v[i, j+1] +
                     a_s * self.v[i, j-1] + b) / a_p
                )

    def solve_pressure_correction(self, uworder=1):
        """求解压力修正方程"""
        # 初始化源项和系数
        b = np.zeros((self.nx, self.ny))
        a_p = np.zeros((self.nx, self.ny))
        
        # 构建压力修正方程
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                # 连续性源项 (质量不平衡)
                b[i, j] = self.rho * (
                    (self.u_star[i, j] - self.u_star[i+1, j]) / self.dx +
                    (self.v_star[i, j] - self.v_star[i, j+1]) / self.dy
                ) * self.dx * self.dy
                
                # 系数计算 (使用SIMPLE算法中的d系数)
                d_e = self.dy / (self.rho * (self.u_star[i+1, j] - self.u_star[i, j] + 1e-10))
                d_w = self.dy / (self.rho * (self.u_star[i, j] - self.u_star[i-1, j] + 1e-10))
                d_n = self.dx / (self.rho * (self.v_star[i, j+1] - self.v_star[i, j] + 1e-10))
                d_s = self.dx / (self.rho * (self.v_star[i, j] - self.v_star[i, j-1] + 1e-10))
                
                # 邻居系数
                a_E = self.rho * d_e * self.dy
                a_W = self.rho * d_w * self.dy
                a_N = self.rho * d_n * self.dx
                a_S = self.rho * d_s * self.dx
                
                # 主对角系数
                a_p[i, j] = a_E + a_W + a_N + a_S
                
                # 更新b项
                b[i, j] -= a_E * self.p_prime[i+1, j] + a_W * self.p_prime[i-1, j] + \
                            a_N * self.p_prime[i, j+1] + a_S * self.p_prime[i, j-1]
        
        # 边界条件: 压力修正的Neumann条件
        # 左右边界
        self.p_prime[0, :] = self.p_prime[1, :]
        self.p_prime[-1, :] = self.p_prime[-2, :]
        # 上下边界
        self.p_prime[:, 0] = self.p_prime[:, 1]
        self.p_prime[:, -1] = self.p_prime[:, -2]
        
        # 迭代求解压力修正 (Gauss-Seidel方法)
        max_inner_iter = 100
        tol_inner = 1e-3
        for _ in range(max_inner_iter):
            p_prime_old = self.p_prime.copy()
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    self.p_prime[i, j] = (b[i, j] + 
                                         a_E * self.p_prime[i+1, j] + 
                                         a_W * self.p_prime[i-1, j] + 
                                         a_N * self.p_prime[i, j+1] + 
                                         a_S * self.p_prime[i, j-1]) / a_p[i, j]
            
            # 内部迭代收敛检查
            res = np.max(np.abs(self.p_prime - p_prime_old))
            if res < tol_inner:
                break

    def correct_velocity_pressure(self):
        """修正速度和压力"""
        # 压力修正 (使用欠松弛)
        self.p += self.alpha_p * self.p_prime
        
        # u速度修正
        for i in range(1, self.nx):
            for j in range(1, self.ny+1):
                # 压力梯度修正
                dP_prime = self.p_prime[i, j-1] - self.p_prime[i-1, j-1]
                # 速度修正量
                du = self.dt * dP_prime / (self.rho * self.dx)
                # 应用修正
                self.u[i, j] = self.u_star[i, j] + du
        
        # v速度修正
        for i in range(1, self.nx+1):
            for j in range(1, self.ny):
                # 压力梯度修正
                dP_prime = self.p_prime[i-1, j] - self.p_prime[i-1, j-1]
                # 速度修正量
                dv = self.dt * dP_prime / (self.rho * self.dy)
                # 应用修正
                self.v[i, j] = self.v_star[i, j] + dv

    def solve(self, uworder=1):
        """执行SIMPLE算法主循环"""
        for iter in range(self.max_iter):
            # 保存上一步的速度场
            u_old = np.copy(self.u)
            v_old = np.copy(self.v)
            
            # 应用边界条件
            self.apply_boundary_conditions()
            
            # SIMPLE步骤
            self.solve_momentum_u(uworder)        # 求解u*
            self.solve_momentum_v(uworder)         # 求解v*
            self.solve_pressure_correction(uworder) # 求解p'
            self.correct_velocity_pressure()# 修正速度和压力
            
            # 应用边界条件 (确保边界值不变)
            self.apply_boundary_conditions()
            
            # 计算连续性误差 (质量守恒)
            mass_error = 0.0
            for i in range(1, self.nx):
                for j in range(1, self.ny):
                    mass_error += abs(
                        (self.u[i, j] - self.u[i+1, j]) / self.dx +
                        (self.v[i, j] - self.v[i, j+1]) / self.dy
                    )
            mass_error /= (self.nx * self.ny)
            
            # 检查收敛性
            u_res = np.max(np.abs(self.u - u_old))
            v_res = np.max(np.abs(self.v - v_old))
            
            if iter % 10 == 0:
                print(f"Iter {iter}: U_res={u_res:.2e}, V_res={v_res:.2e}, Mass_err={mass_error:.2e}")
            
            if u_res < self.tol and v_res < self.tol and mass_error < 1e-4:
                print(f"Converged at iteration {iter}")
                break

    def get_center_velocity(self):
        """获取网格中心的速度场"""
        # u在x方向中心，y方向需要平均
        u_center = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                u_center[i, j] = 0.5 * (self.u[i, j+1] + self.u[i+1, j+1])
        
        # v在y方向中心，x方向需要平均
        v_center = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                v_center[i, j] = 0.5 * (self.v[i+1, j] + self.v[i+1, j+1])
        
        return u_center, v_center, self.p

    def calculate_streamfunction(self):
        """计算流函数"""
        psi = np.zeros((self.nx, self.ny))
        
        # 从底部开始积分
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                # 使用中心差分计算速度
                u_ij = 0.5 * (self.u[i, j+1] + self.u[i+1, j+1])
                v_ij = 0.5 * (self.v[i+1, j] + self.v[i+1, j+1])
                
                # 更新流函数
                psi[i, j] = psi[i-1, j] - v_ij * self.dx
                psi[i, j] = psi[i, j-1] + u_ij * self.dy
        
        # 归一化
        psi = psi - np.min(psi)
        return psi / np.max(psi)