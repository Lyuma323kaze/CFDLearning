import numpy as np
from Diff_schme import DiffSchemes


class VorticityStreamPoiseuille(DiffSchemes):
    def __init__(self, name, dt, dx, dy, x, y, t, nu, U0, H, 
                 ini_condi=None, bnd_condi=None, folder=None):
        """
        涡量-流函数方法求解泊肃叶流动
        :param nu: 运动粘度
        :param U0: 中心线速度
        :param H: 管道高度(y方向)
        """
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, 
                         ini_condi=ini_condi, bnd_condi=bnd_condi, folder=folder)
        self.nu = nu        # 运动粘度
        self.U0 = U0        # 中心线最大速度
        self.H = H          # 管道高度
        self.Re = U0 * H / nu  # 雷诺数
        self.ny = len(y)    # y方向网格数
        
        # initialize fields (nx, ny)
        self.psi = None
        self.vorticity = None
        self.u = None
        self.v = None
        self.initialize_fields()
        
        # 设置边界条件
        self.set_boundary_conditions()

    def initialize_fields(self):
        """初始化流函数和涡量场"""
        nx, ny = len(self.x), len(self.y)
        self.psi = np.zeros((nx, ny))
        self.vorticity = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        
        # uniform inlet and initial condition
        self.u = self.U0 * np.ones((nx, ny))  # initial value
        self.psi = np.cumsum(self.u, axis=0) * self.dy  # 积分得到流函数初值
        
    def set_boundary_conditions(self):
        """设置二维泊肃叶流动的边界条件"""
        # upper and lower wall conditions
        self.psi[:, 0] = 0
        self.psi[:, -1] = self.psi[0, -1]
        
        self.u[:, -1] = 0, self.u[:, 0] = 0  
        self.v[:, 0] = 0, self.v[:, -1] = 0  
        
        self.vorticity[:, 0] = 2 * self.psi[:, 1] / self.dy**2  
        self.vorticity[:, -1] = 2 * self.psi[:, -2] / self.dy**2  
        # inlet
        self.vorticity[0, 1:-1] = 2 * (self.psi[1, 1:-1] - self.psi[0, 1:-1]) / self.dx**2 +\
                                (self.psi[0, 2:] - 2 * self.psi[0, 1:-1] + self.psi[0, :-2]) / self.dx**2     

        # outlet (fully developed flow)
        self.vorticity[-1, :] = self.vorticity[-2, :]  
        self.psi[-1, :] = 2 * self.psi[-2, :] - self.psi[-3, :]
        
        # update u,v values for solving vorticity
        self.u[1:-1, 1:-1] = (self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dx)
        self.u[-1] = self.u[-2]  # outlet condition
        self.v[-1] = self.v[-2]

    def solve(self, max_iter=10000, tol=1e-6):
        """求解稳态涡量-流函数方程"""
        for iter in range(max_iter):
            psi_old = self.psi.copy()
            
            # 1. 求解涡量输运方程
            self.solve_vorticity_transport()
            
            # 2. 求解泊松方程 (∇²ψ = ω)
            self.solve_psi_poisson()
            
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
        """solve vorticity transport equation (FTCS)"""
        new_vort = np.copy(self.vorticity)
        
        # change values [1:-1, 1:-1]
        omega_change = (self.dt / self.Re) * ((self.vorticity[2:,1:-1] 
                                               - 2 * self.vorticity[1:-1,1:-1] 
                                               + self.vorticity[:-2,1:-1]) / self.dx**2 + 
                                              (self.vorticity[1:-1,2:]
                                               - 2 * self.vorticity[1:-1,1:-1]
                                               + self.vorticity[1:-1,:-2]) / self.dy**2) + \
                        (self.dt / self.dx) * (self.u[1:-1, 1:-1] *
                                               (self.vorticity[2:, 1:-1] - self.vorticity[:-2, 1:-1]) / 2) + \
                        (self.dt / self.dy) * (self.v[1:-1, 1:-1] *
                                               (self.vorticity[1:-1, 2:] - self.vorticity[1:-1, :-2]) / 2)
        # update values for inner points
        new_vort[1:-1,1:-1] = self.vorticity[1:-1, 1:-1] + omega_change
        self.vorticity[1:-1, 1:-1] = new_vort[1:-1, 1:-1]

    def solve_psi_poisson(self, max_iter=100, tol=1e-4):
        """solve Poisson equation for stream function"""
        for _ in range(max_iter):
            psi_old = self.psi.copy()
            
            # container for [1:-1, 1:-1] points
            psi_new = np.copy(self.psi)
            psi_new[1:-2, 1:-1] = 0.5 * (self.dx ** (-2) + self.dy ** (-2)) ** -1 * (
                (self.psi[2:-1, 1:-1] + self.psi[:-3, 1:-1]) / self.dx ** 2 +
                (self.psi[1:-1, 2:] + self.psi[1:-1, :-2]) / self.dy ** 2 -
                self.vorticity[1:-2, 1:-1]
            )
            psi_new[-2, 1:-1] = 0.5 * (-self.dy ** 2 * self.vorticity[-2, 1:-1] + 
                self.psi[-2, 2:] + self.psi[-2, :-2])
            
            self.psi[1:-1, 1:-1] = psi_new[1:-1, 1:-1]
            
            # 检查收敛
            if np.max(np.abs(self.psi - psi_old)) < tol:
                break

    def get_velocity_field(self):
        """return velocity field (u, v)"""
        return self.u, self.v  