import numpy as np
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
import os
from jax.experimental import sparse

class OGridLaplaceGenerator:
    """
    Generates O-type 2D grids by solving Laplace equations.
    
    This method solves the following system of partial differential equations 
    to obtain physical coordinates (x,y) as functions of computational coordinates (xi, eta):
        x_xixi + x_etaeta = 0
        y_xixi + y_etaeta = 0
    The equations are discretized using finite difference method and solved iteratively.
    """

    def __init__(self, NI, NJ, inner_boundary_func, outer_boundary_func):
        """
        Initialize grid generator.

        Parameters:
            NI (int): Number of points in circumferential (xi) direction
            NJ (int): Number of points in radial (eta) direction
            inner_boundary_func (callable): Function defining inner boundary t -> (x, y)
                                           't' is parameter from 0 to 1 representing
                                           normalized arc length or angle
                                           Should return tuple (x, y)
            outer_boundary_func (callable): Function defining outer boundary t -> (x, y)
                                           Similar to inner_boundary_func
        """
        if NI <= 1 or NJ <= 1:
            raise ValueError("NI and NJ must be greater than 1")
        # mesh parameters and BDCs
        self.NI = NI
        self.NJ = NJ
        self.inner_boundary_func = inner_boundary_func
        self.outer_boundary_func = outer_boundary_func

        # Physical coordinates (x, y)
        self.x = np.zeros((NI, NJ), dtype=float)
        self.y = np.zeros((NI, NJ), dtype=float)
        # Computational coordinates (unnormalized)
        self.xi, self.eta = np.meshgrid(np.arange(self.NI), np.arange(self.NJ), indexing='ij')
        
        # Computational coordinates (normalized) for generating BDC
        # xi_comp will vary from 0 to (NI-1)/NI
        # eta_comp will vary from 0 to 1
        self.xi_comp = np.zeros((NI, NJ), dtype=float)
        self.eta_comp = np.zeros((NI, NJ), dtype=float)
        # Initialize derivatives and Jacobian properties
        self.x_xi = None
        self.x_eta = None
        self.y_xi = None
        self.y_eta = None
        self.J_jacobian = None
        self.xi_x = None
        self.xi_y = None
        self.eta_x = None
        self.eta_y = None
        # For O-grids, xi represents angular-like coordinate
        # Common normalization ranges from 0 to almost 1 (or 2*pi)
        self.xi_comp = self.xi / NI
        self.eta_comp = self.eta / (NJ - 1)  # eta values are 0, 1/(NJ-1), ..., 1
        
        self._initialize_boundaries()
        self._initialize_interior_guess()

    def _initialize_boundaries(self):
        """Set x,y coordinates for inner boundary (eta=0) and outer boundary (eta=NJ-1)"""
        # t_params are values for boundary function parameter 't'
        # These correspond to our normalized xi coordinates
        t_params = np.linspace(0, 1, self.NI, endpoint=False)  # NI points: 0, 1/NI, ..., (NI-1)/NI

        for i in range(self.NI):
            # ti is equivalent to self.xi_comp[i, 0] or self.xi_comp[i, self.NJ-1]
            ti = t_params[i] 
            
            # Inner boundary (j=0)
            self.x[i, 0], self.y[i, 0] = self.inner_boundary_func(ti)
            
            # Outer boundary (j=NJ-1)
            self.x[i, self.NJ - 1], self.y[i, self.NJ - 1] = self.outer_boundary_func(ti)

    def _initialize_interior_guess(self):
        """Initialize interior grid points using linear interpolation"""
        for j in range(1, self.NJ - 1):
            eta = self.eta_comp[0, j]
            for i in range(self.NI):
                xi = self.xi_comp[i, j]
                # Boundary contributions
                U_0 = np.array([self.x[i, 0], self.y[i, 0]])
                U_1 = np.array([self.x[i, -1], self.y[i, -1]])
                V_0 = np.array([self.x[0, j], self.y[0, j]])
                V_1 = np.array([self.x[-1, j], self.y[-1, j]])

                C_00 = np.array([self.x[0, 0], self.y[0, 0]])
                C_10 = np.array([self.x[-1, 0], self.y[-1, 0]])
                C_01 = np.array([self.x[0, -1], self.y[0, -1]])
                C_11 = np.array([self.x[-1, -1], self.y[-1, -1]])
                # TFI formula
                term1 = (1 - eta) * U_0 + eta * U_1
                term2 = (1 - xi) * V_0 + xi * V_1
                term3 = (1 - xi) * (1 - eta) * C_00 + xi * (1 - eta) * C_10 + \
                        (1 - xi) * eta * C_01 + xi * eta * C_11

                # TFI formula
                self.x[i, j], self.y[i, j] = term1 + term2 - term3

    # TODO: jit to be added
    def solve_laplace_equations(self, max_iterations=10000, tolerance=1e-6):
        valid_ls = [self.x_xi, self.x_eta, self.y_xi, self.y_eta]
        if any(item is None for item in valid_ls):
            self.compute_derivatives_phy_to_com_and_jacobian()
        # 2d arrays for xi, eta indices (shape = (NI, NJ))
        idx_xi, idx_eta = self.xi, self.eta
        # Jacobian iteration
        for iteration in range(max_iterations):
            # derivatives in inner region
            x_xi = self.x_xi[:, 1:-1]
            x_eta = self.x_eta[:,1:-1]
            y_xi = self.y_xi[:, 1:-1]
            y_eta = self.y_eta[:, 1:-1]
            # values of last iteration
            x_old_iter = self.x.copy()
            y_old_iter = self.y.copy()
            x_new = np.zeros_like(self.x)
            x_new[:, 0] = x_old_iter[:, 0]  # inner boundary
            x_new[:, -1] = x_old_iter[:, -1]  # outer boundary
            y_new = np.zeros_like(self.y)
            y_new[:, 0] = y_old_iter[:, 0]  # inner boundary
            y_new[:, -1] = y_old_iter[:, -1]  # outer boundary
            max_diff_iter = 0.0
            # the transitioned indices
            xi_p1 = np.roll(idx_xi, -1, axis=0)[:, 1:-1]
            xi_m1 = np.roll(idx_xi, 1, axis=0)[:, 1:-1]
            xi_0 = idx_xi[:, 1:-1]
            eta_p1 = idx_eta[:, 2:]
            eta_m1 = idx_eta[:, :-2]
            eta_0 = idx_eta[:, 1:-1]
            # coefficients
            b_we = x_eta ** 2 + y_eta ** 2
            beta = x_xi * x_eta + y_xi * y_eta
            b_sn = x_xi ** 2 + y_xi ** 2
            b_p = 2 * b_we + 2 * b_sn
            b_p = np.where(b_p == 0, 1e-8, b_p)  # Avoid division by zero
            # here I would like to generate a reduced matrix, shape = (NI, NJ-2), and the 2 indices are
            # given by the xi and eta indices.
            c_px = - beta * (x_old_iter[xi_p1, eta_p1] - x_old_iter[xi_m1, eta_p1] +
                             x_old_iter[xi_m1, eta_m1] - x_old_iter[xi_p1, eta_m1]) / 2
            c_py = - beta * (y_old_iter[xi_p1, eta_p1] - y_old_iter[xi_m1, eta_p1] +
                             y_old_iter[xi_m1, eta_m1] - y_old_iter[xi_p1, eta_m1]) / 2
            # TODO: in-place assignment to be fixed when using jax
            x_new[xi_0, eta_0] = (b_we * x_old_iter[xi_m1, eta_0] + b_we * x_old_iter[xi_p1, eta_0] +
                     b_sn * x_old_iter[xi_0, eta_m1] + b_sn * x_old_iter[xi_0, eta_p1] +
                     c_px) / b_p
            y_new[xi_0, eta_0] = (b_we * y_old_iter[xi_m1, eta_0] + b_we * y_old_iter[xi_p1, eta_0] +
                     b_sn * y_old_iter[xi_0, eta_m1] + b_sn * y_old_iter[xi_0, eta_p1] +
                     c_py) / b_p
            current_max_diff = np.maximum(np.abs(x_new - x_old_iter), np.abs(y_new - y_old_iter))
            current_max_diff = np.max(current_max_diff)
            self.x = x_new
            self.y = y_new
            self.compute_derivatives_phy_to_com_and_jacobian()
            if current_max_diff > max_diff_iter:
                max_diff_iter = current_max_diff
            
            if iteration % 200 == 0 or iteration == max_iterations -1 :  # Periodic progress reporting
                print(f"Iteration {iteration + 1}/{max_iterations}, Max difference: {max_diff_iter:.2e}")

            if max_diff_iter < tolerance:
                print(f"Converged after {iteration + 1} iterations. Max difference: {max_diff_iter:.2e}")
                break
  
    def compute_derivatives_phy_to_com_and_jacobian(self):
        """
        Compute partial derivatives of x,y w.r.t xi,eta and Jacobian J.
        Uses central differences for interior points, second-order one-sided for boundaries.
        Results stored in self.x_xi, self.x_eta, self.y_xi, self.y_eta, self.J_jacobian
        """
        if self.x is None or self.y is None:
            print("Physical coordinates (x,y) not available. Run solver first.")
            return
        if np.sum(self.x**2) == 0 and np.sum(self.y**2) == 0 :  # Simple check for initial zeros
             print("Warning: Physical coordinates may not be properly computed (possibly still initial zeros). Derivatives may be meaningless.")


        self.x_xi = np.zeros((self.NI, self.NJ), dtype=float)
        self.x_eta = np.zeros((self.NI, self.NJ), dtype=float)
        self.y_xi = np.zeros((self.NI, self.NJ), dtype=float)
        self.y_eta = np.zeros((self.NI, self.NJ), dtype=float)
        self.J_jacobian = np.zeros((self.NI, self.NJ), dtype=float)

        # Step sizes in computational coordinates
        # xi_comp values are k/NI, k = 0, ..., NI-1. So step size is 1/NI
        d_xi = 1.0
        
        # eta_comp values are k/(NJ-1), k = 0, ..., NJ-1. So step size is 1/(NJ-1)
        # self.NJ guaranteed > 1 (controlled by __init__)
        d_eta = 1.0

        for i in range(self.NI):
            for j in range(self.NJ):
                # --- Xi derivatives (circumferential) ---
                # Use central differences with periodic boundary handling
                ip1 = (i + 1) % self.NI
                im1 = (i - 1 + self.NI) % self.NI
                
                self.x_xi[i, j] = (self.x[ip1, j] - self.x[im1, j]) / (2.0 * d_xi)
                self.y_xi[i, j] = (self.y[ip1, j] - self.y[im1, j]) / (2.0 * d_xi)

                # --- Eta derivatives (radial) ---
                # self.NJ guaranteed >= 2
                if self.NJ == 2:  # Only two lines (j=0, j=1)
                    # Use first-order one-sided differences. d_eta_norm = 1.0/(2-1) = 1.0
                    if j == 0:  # Inner boundary (j=0)
                        self.x_eta[i, j] = (self.x[i, j + 1] - self.x[i, j]) / d_eta
                        self.y_eta[i, j] = (self.y[i, j + 1] - self.y[i, j]) / d_eta
                    else:  # Outer boundary (j=1)
                        self.x_eta[i, j] = (self.x[i, j] - self.x[i, j - 1]) / d_eta
                        self.y_eta[i, j] = (self.y[i, j] - self.y[i, j - 1]) / d_eta
                else:  # self.NJ >= 3, can use second-order one-sided on boundaries
                    if j == 0:  # Inner boundary (j=0, eta=0) - second-order forward
                        # f'(x0) = (-3f0 + 4f1 - f2)/(2h)
                        self.x_eta[i, j] = (-3.0 * self.x[i, 0] + 4.0 * self.x[i, 1] - self.x[i, 2]) / (2.0 * d_eta)
                        self.y_eta[i, j] = (-3.0 * self.y[i, 0] + 4.0 * self.y[i, 1] - self.y[i, 2]) / (2.0 * d_eta)
                    elif j == self.NJ - 1:  # Outer boundary (j=NJ-1, eta=1) - second-order backward
                        # f'(x_N) = (3f_N -4f_{N-1} +f_{N-2})/(2h)
                        self.x_eta[i, j] = (3.0 * self.x[i, self.NJ - 1] - 4.0 * self.x[i, self.NJ - 2] + self.x[i, self.NJ - 3]) / (2.0 * d_eta)
                        self.y_eta[i, j] = (3.0 * self.y[i, self.NJ - 1] - 4.0 * self.y[i, self.NJ - 2] + self.y[i, self.NJ - 3]) / (2.0 * d_eta)
                    else:  # Interior points (0 < j < NJ-1) - central differences
                        self.x_eta[i, j] = (self.x[i, j + 1] - self.x[i, j - 1]) / (2.0 * d_eta)
                        self.y_eta[i, j] = (self.y[i, j + 1] - self.y[i, j - 1]) / (2.0 * d_eta)
                
                # --- Compute Jacobian determinant J ---
                # J = x_xi * y_eta - x_eta * y_xi
                self.J_jacobian[i, j] = self.x_xi[i, j] * self.y_eta[i, j] - self.x_eta[i, j] * self.y_xi[i, j]
        self.J_jacobian = np.where(self.J_jacobian == 0, 1e-8, self.J_jacobian)  # Avoid division by zero
        
        # print("Derivatives (x_xi, x_eta, y_xi, y_eta) and Jacobian computed and stored")

    def compute_derivatives_com_to_phy(self):
        self.xi_x = self.y_eta / self.J_jacobian
        self.xi_y = -self.x_eta / self.J_jacobian
        self.eta_x = -self.y_xi / self.J_jacobian
        self.eta_y = self.x_xi / self.J_jacobian
        # print("Computed derivatives from computational to physical coordinates (xi_x, xi_y, eta_x, eta_y)")

    def get_coordinates(self):
        """
        Return stored physical and computational coordinates.

        Returns:
            tuple: (x_coords, y_coords, xi_coords, eta_coords)
                   x_coords (np.ndarray): NIxNJ physical x-coordinate array
                   y_coords (np.ndarray): NIxNJ physical y-coordinate array
                   xi_coords (np.ndarray): NIxNJ computational xi-coordinate array (normalized)
                   eta_coords (np.ndarray): NIxNJ computational eta-coordinate array (normalized)
        """
        return self.x, self.y, self.xi_comp, self.eta_comp

    def get_derivatives_and_jacobian(self):
        """
        Return stored derivatives and Jacobian
        """
        if self.J_jacobian is None:
            print("Derivatives and Jacobian not computed yet. Call compute_derivatives_and_jacobian() first")
            return None, None, None, None, None
        return self.xi_x, self.xi_y, self.eta_x, self.eta_y, self.J_jacobian
    
    def output_to_file(self, filename):
        """Output grid coordinates to a file"""
        # TODO: output an array of self.x, self.y, self.xi, self.eta
        ...
        # TODO: output the derivatives and Jacobian in an array
        self.compute_derivatives_phy_to_com_and_jacobian()
        self.compute_derivatives_com_to_phy()
        ...
    
    def plot_physical_grid(self, show_points=False):
        """Plot generated grid in physical (x,y) plane"""
        if 'matplotlib' not in globals() and 'plt' not in globals(): 
            print("Matplotlib not imported. Cannot plot grid")
            return

        plt.figure(figsize=(10, 8))
        
        # Plot constant-eta lines (look like "rings" or "shells" in O-grid)
        # self.x[:, j] is line of constant j (eta)
        for j in range(self.NJ):  # For each eta=constant line
            # Connect last point to first to close O-grid loop
            line_x = np.append(self.x[:, j], self.x[0, j])
            line_y = np.append(self.y[:, j], self.y[0, j])
            plt.plot(line_x, line_y, 'b-', linewidth=0.8, label='Eta lines' if j == 0 else "")


        # Plot constant-xi lines (look like "radial lines" or "spokes")
        # self.x[i, :] is line of constant i (xi)
        for i in range(self.NI):  # For each xi=constant line
            plt.plot(self.x[i, :], self.y[i, :], 'r-', linewidth=0.8, label='Xi lines' if i == 0 else "")
        
        plt.xlabel("x (Physical)")
        plt.ylabel("y (Physical)")
        plt.title(f"Generated O-grid (NI={self.NI}, NJ={self.NJ})")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        if self.NI > 0 and self.NJ > 0 : plt.legend()
        plt.show()

    def plot_computational_grid(self):
        """Plot grid in computational (xi,eta) plane"""
        if 'matplotlib' not in globals() and 'plt' not in globals():
            print("Matplotlib not imported. Cannot plot grid")
            return
        
        # Adjust figure size to better match computational domain proportions
        aspect_ratio = (self.NJ / (self.NJ -1 )) / (self.NI / self.NI) if self.NI > 0 and self.NJ > 1 else 1.0
        fig_width = 8
        fig_height = fig_width * aspect_ratio 
        if self.NJ <=1 :
            fig_height = fig_width * 0.2  # Very thin if NJ=1

        plt.figure(figsize=(fig_width, fig_height if fig_height > 1 else 2))  # Ensure reasonable height
        
        # Plot constant-eta lines (horizontal)
        for j in range(self.NJ):
            plt.plot(self.xi_comp[:, j], self.eta_comp[:, j], 'b-', linewidth=0.8)

        # Plot constant-xi lines (vertical)
        for i in range(self.NI):
            plt.plot(self.xi_comp[i, :], self.eta_comp[i, :], 'r-', linewidth=0.8)
        
        plt.xlabel("ξ (Computational)")
        plt.ylabel("η (Computational)")
        plt.title("Computational Grid (Normalized)")
        plt.grid(True, linestyle=':', alpha=0.5)
        # Set axis ranges to better display normalized coordinates
        plt.xlim([-0.05, 1.05 if self.NI == 1 else (self.NI-1)/self.NI + 0.05])  # xi from 0 to (NI-1)/NI
        plt.ylim([-0.05, 1.05])  # eta from 0 to 1
        plt.gca().set_aspect('auto', adjustable='box')  # Or 'equal' for equal aspect ratio
        plt.show()

# 示例用法:
if __name__ == '__main__':
    # 定义边界函数
    # 示例: 用于环形区域的内圆和外圆
    # t 是一个从 0 到 1 的参数
    def inner_circle_boundary(t, radius=1.0, center_x=0.0, center_y=0.0):
        angle = 2 * np.pi * t
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        return x, y

    def outer_circle_boundary(t, radius=3.0, center_x=0.0, center_y=0.0):
        angle = 2 * np.pi * t
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        return x, y

    # 定义一个更复杂的内边界示例：椭圆
    def inner_ellipse_boundary(t, a=2.0, b=0.8, center_x=0.0, center_y=0.0): # a: 半长轴, b: 半短轴
        angle = 2 * np.pi * t
        x = center_x + a * np.cos(angle)
        y = center_y + b * np.sin(angle)
        return x, y

    # 网格参数
    NI_points = 80  # 周向点数
    NJ_points = 40  # 径向点数

    print(f"正在生成 O 型网格，NI={NI_points}, NJ={NJ_points}...")

    # === 使用同心圆 ===
    # generator = OGridLaplaceGenerator(NI_points, NJ_points,
    #                                   inner_boundary_func=lambda t: inner_circle_boundary(t, radius=1.0),
    #                                   outer_boundary_func=lambda t: outer_circle_boundary(t, radius=3.0))

    # === 使用椭圆作为内边界，圆作为外边界 ===
    generator = OGridLaplaceGenerator(NI_points, NJ_points,
                                      inner_boundary_func=lambda t: inner_ellipse_boundary(t, a=2.0, b=1),
                                      outer_boundary_func=lambda t: outer_circle_boundary(t, radius=40.0))


    print("\n边界和内部点初始猜测已设定。")
    print(f"内边界 x[0,0]: ({generator.x[0,0]:.3f}, {generator.y[0,0]:.3f})")
    print(f"外边界 x[0,NJ-1]: ({generator.x[0,NJ_points-1]:.3f}, {generator.y[0,NJ_points-1]:.3f})")
    
    # 求解拉普拉斯方程以生成网格
    # omega 通常在 (1.0, 2.0) 之间以获得SOR的良好性能。
    # 对于JOR（当前实现），omega=1.0 对应于雅可比方法。
    # 较大的 omega 值 (如 1.8, 1.9) 通常可以加速收敛，但最佳值取决于具体问题。
    print("\n开始求解拉普拉斯方程...")
    generator.solve_laplace_equations(max_iterations=30000, tolerance=1e-8)

    # 获取坐标
    x_physical, y_physical, xi_computational, eta_computational = generator.get_coordinates()

    print("\n网格点坐标示例:")
    print(f"x[0,0] (物理): {x_physical[0,0]:.3f}, y[0,0] (物理): {y_physical[0,0]:.3f}")
    print(f"xi[0,0] (计算): {xi_computational[0,0]:.3f}, eta[0,0] (计算): {eta_computational[0,0]:.3f}")
    
    print(f"x[NI/2, NJ/2] (物理): {x_physical[NI_points//2, NJ_points//2]:.3f}, y[NI/2, NJ/2] (物理): {y_physical[NI_points//2, NJ_points//2]:.3f}")
    print(f"xi[NI/2, NJ/2] (计算): {xi_computational[NI_points//2, NJ_points//2]:.3f}, eta[NI/2, NJ/2] (计算): {eta_computational[NI_points//2, NJ_points//2]:.3f}")

    # 绘制网格 

    print("\n正在绘制物理网格...")
    generator.plot_physical_grid(show_points=False)
    print("正在绘制计算网格...")
    # generator.plot_computational_grid()
    print("绘图完成。请查看弹出的窗口。")
