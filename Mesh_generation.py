import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange


class OGridLaplaceGenerator:
    """
    Generates O-type 2D grids by solving Laplace equations.
    
    This method solves the following system of partial differential equations 
    to obtain physical coordinates (x,y) as functions of computational coordinates (xi, eta):
        x_xixi + x_etaeta = 0
        y_xixi + y_etaeta = 0
    The equations are discretized using finite difference method and solved iteratively.
    """

    def __init__(self, NI, NJ, inner_boundary_func, outer_boundary_func, symmetric=False):
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
        
        

        self._initialize_boundaries(symmetric)
        self._initialize_interior_guess()

    def _initialize_boundaries(self, symmetric_endpoint=False, prt = False):
        """Set x,y coordinates for inner boundary (eta=0) and outer boundary (eta=NJ-1)"""
        # t_params are values for boundary function parameter 't'
        # These correspond to our normalized xi coordinates
        if symmetric_endpoint:
            t_params_1 = np.linspace(0, 0.5, self.NI//2 + 1, endpoint=True)[:-1]  # 0→0.5（不包含0.5）
            t_params_2 = np.linspace(0.5, 1, self.NI - len(t_params_1) + 1, endpoint=True)  # 0.5→1（包含1）
            t_params = np.concatenate((t_params_1, t_params_2))
        else:
            t_params = np.linspace(0, 1, self.NI, endpoint=False)  # NI points: 0, 1/NI, ..., (NI-1)/NI

        for i in range(self.NI):
            # ti is equivalent to self.xi_comp[i, 0] or self.xi_comp[i, self.NJ-1]
            ti = t_params[i] 
            
            # Inner boundary (j=0)
            self.x[i, 0], self.y[i, 0] = self.inner_boundary_func(ti)
            
            # Outer boundary (j=NJ-1)
            self.x[i, self.NJ - 1], self.y[i, self.NJ - 1] = self.outer_boundary_func(ti)

    def _initialize_interior_guess(self):
        self.x = (1 - self.eta_comp) * self.x[:, 0, np.newaxis] + self.eta_comp * self.x[:, -1, np.newaxis]
        self.y = (1 - self.eta_comp) * self.y[:, 0, np.newaxis] + self.eta_comp * self.y[:, -1, np.newaxis]
    
    def solve_laplace_equations(self, max_iterations=10000, tolerance=1e-6, alpha = 0.5):
        valid_ls = [self.x_xi, self.x_eta, self.y_xi, self.y_eta]
        if any(item is None for item in valid_ls):
            self.compute_derivatives_phy_to_com_and_jacobian()
        # 2d arrays for xi, eta indices (shape = (NI, NJ))
        idx_xi, idx_eta = self.xi, self.eta
        # static parameters for numba
        NI = self.NI
        NJ = self.NJ
        x = self.x
        y = self.y
        x_xi = self.x_xi
        y_xi = self.y_xi
        x_eta = self.x_eta
        y_eta = self.y_eta
        J_jacobian = self.J_jacobian
        # Jacobian iteration
        @jit
        def comput_drv_phy_to_com_and_jcbn(x, y, J_jacobian, d_xi=1.0, d_eta=1.0):
            x_xi = np.zeros((NI, NJ), dtype=np.float64)
            y_xi = np.zeros((NI, NJ), dtype=np.float64)
            x_eta = np.zeros((NI, NJ), dtype=np.float64)
            y_eta = np.zeros((NI, NJ), dtype=np.float64)
            for i in prange(NI):
                for j in prange(NJ):
                    # --- Xi derivatives (circumferential) ---
                    # Use central differences with periodic boundary handling
                    ip1 = (i + 1) % NI
                    im1 = (i - 1 + NI) % NI
                    
                    x_xi[i, j] = (x[ip1, j] - x[im1, j]) / (2.0 * d_xi)
                    y_xi[i, j] = (y[ip1, j] - y[im1, j]) / (2.0 * d_xi)

                    # --- Eta derivatives (radial) ---
                    # NJ guaranteed >= 2
                    if NJ == 2:  # Only two lines (j=0, j=1)
                        # Use first-order one-sided differences. d_eta_norm = 1.0/(2-1) = 1.0
                        if j == 0:  # Inner boundary (j=0)
                            x_eta[i, j] = (x[i, j + 1] - x[i, j]) / d_eta
                            y_eta[i, j] = (y[i, j + 1] - y[i, j]) / d_eta
                        else:  # Outer boundary (j=1)
                            x_eta[i, j] = (x[i, j] - x[i, j - 1]) / d_eta
                            y_eta[i, j] = (y[i, j] - y[i, j - 1]) / d_eta
                    else:  # NJ >= 3, can use second-order one-sided on boundaries
                        if j == 0:  # Inner boundary (j=0, eta=0) - second-order forward
                            # f'(x0) = (-3f0 + 4f1 - f2)/(2h)
                            x_eta[i, j] = (-3.0 * x[i, 0] + 4.0 * x[i, 1] - x[i, 2]) / (2.0 * d_eta)
                            y_eta[i, j] = (-3.0 * y[i, 0] + 4.0 * y[i, 1] - y[i, 2]) / (2.0 * d_eta)
                        elif j == NJ - 1:  # Outer boundary (j=NJ-1, eta=1) - second-order backward
                            # f'(x_N) = (3f_N -4f_{N-1} +f_{N-2})/(2h)
                            x_eta[i, j] = (3.0 * x[i, NJ - 1] - 4.0 * x[i, NJ - 2] + x[i, NJ - 3]) / (2.0 * d_eta)
                            y_eta[i, j] = (3.0 * y[i, NJ - 1] - 4.0 * y[i, NJ - 2] + y[i, NJ - 3]) / (2.0 * d_eta)
                        else:  # Interior points (0 < j < NJ-1) - central differences
                            x_eta[i, j] = (x[i, j + 1] - x[i, j - 1]) / (2.0 * d_eta)
                            y_eta[i, j] = (y[i, j + 1] - y[i, j - 1]) / (2.0 * d_eta)
                    
                    # --- Compute Jacobian determinant J ---
                    # J = x_xi * y_eta - x_eta * y_xi
                    J_jacobian[i, j] = x_xi[i, j] * y_eta[i, j] - x_eta[i, j] * y_xi[i, j]
            
            J_jacobian = np.where(J_jacobian == 0, 1e-8, J_jacobian)  # Avoid division by zero
            return x_xi, y_xi, x_eta, y_eta, J_jacobian
        
        @jit
        def sub_loop(x, y, x_xi, y_xi, x_eta, y_eta, J_jacobian, max_iterations_, tolerance_):
            x_xi_ = x_xi.copy()
            y_xi_ = y_xi.copy()
            x_eta_ = x_eta.copy()
            y_eta_ = y_eta.copy()
            # the transitioned indices
            xi_p1 = np.zeros((NI, NJ), dtype=np.int32)
            xi_m1 = np.zeros((NI, NJ), dtype=np.int32)
            # indices
            for j in prange(NJ):
                for i in prange(NI):
                    xi_p1[i, j] = idx_xi[(i - 1) % NI, j]  # manual np.roll(idx_xi, -1, axis=0)
            xi_p1 = xi_p1[:, 1:-1]
            for j in prange(NJ):
                for i in prange(NI):
                    xi_m1[i, j] = idx_xi[(i + 1) % NI, j]  # manual np.roll(idx_xi, 1, axis=0)
            xi_m1 = xi_m1[:, 1:-1]
            # xi_p1 = np.roll(idx_xi, -1, axis=0)[:, 1:-1]
            # xi_m1 = np.roll(idx_xi, 1, axis=0)[:, 1:-1]
            xi_0 = idx_xi[:, 1:-1]
            eta_p1 = idx_eta[:, 2:]
            eta_m1 = idx_eta[:, :-2]
            eta_0 = idx_eta[:, 1:-1]
            # derivatives in inner region
            x_xi_ = x_xi[:, 1:-1]
            x_eta_ = x_eta[:,1:-1]
            y_xi_ = y_xi[:, 1:-1]
            y_eta_ = y_eta[:, 1:-1]
            for iteration in prange(max_iterations_):
                # values of last iteration
                x_old_iter_ = x.copy()
                y_old_iter_ = y.copy()
                x_new = np.zeros_like(x)
                x_new[:, 0] = x_old_iter_[:, 0]  # inner boundary
                x_new[:, -1] = x_old_iter_[:, -1]  # outer boundary
                y_new = np.zeros_like(y)
                y_new[:, 0] = y_old_iter_[:, 0]  # inner boundary
                y_new[:, -1] = y_old_iter_[:, -1]  # outer boundary
                max_diff_iter = 0.0
                
                # coefficients
                b_we = x_eta_ ** 2 + y_eta_ ** 2
                beta = x_xi_ * x_eta_ + y_xi_ * y_eta_
                b_sn = x_xi_ ** 2 + y_xi_ ** 2
                b_p = 2 * b_we + 2 * b_sn
                b_p = np.where(b_p == 0, 1e-8, b_p)  # Avoid division by zero
                # here I would like to generate a reduced matrix, shape = (NI, NJ-2), and the 2 indices are
                # given by the xi and eta indices.
                
                # compute c_px , c_py
                c_px = np.zeros((NI, NJ-2), dtype=np.float64)
                c_py = np.zeros((NI, NJ-2), dtype=np.float64)
                for i in prange(NI):
                    for j in prange(NJ-2):
                        xi_p1_idx = xi_p1[i, j]
                        eta_p1_idx = eta_p1[i, j]
                        xi_m1_idx = xi_m1[i, j]
                        eta_m1_idx = eta_m1[i, j]
                        c_px[i, j] = -beta[i, j] * (
                            x_old_iter_[xi_p1_idx, eta_p1_idx] - x_old_iter_[xi_m1_idx, eta_p1_idx] +
                            x_old_iter_[xi_m1_idx, eta_m1_idx] - x_old_iter_[xi_p1_idx, eta_m1_idx]
                        ) / 2
                        c_py[i, j] = -beta[i, j] * (
                            y_old_iter_[xi_p1_idx, eta_p1_idx] - y_old_iter_[xi_m1_idx, eta_p1_idx] +
                            y_old_iter_[xi_m1_idx, eta_m1_idx] - y_old_iter_[xi_p1_idx, eta_m1_idx]
                        ) / 2
                # Update x_new , y_new
                for i in prange(NI):
                    for j in prange(NJ-2):
                        xi_0_idx = xi_0[i, j]
                        eta_0_idx = eta_0[i, j]
                        x_new[xi_0_idx, eta_0_idx] = (
                            b_we[i, j] * x_old_iter_[xi_m1[i, j], eta_0_idx] +
                            b_we[i, j] * x_old_iter_[xi_p1[i, j], eta_0_idx] +
                            b_sn[i, j] * x_old_iter_[xi_0_idx, eta_m1[i, j]] +
                            b_sn[i, j] * x_old_iter_[xi_0_idx, eta_p1[i, j]] +
                            c_px[i, j]
                        ) / b_p[i, j]
                        y_new[xi_0_idx, eta_0_idx] = (
                            b_we[i, j] * y_old_iter_[xi_m1[i, j], eta_0_idx] +
                            b_we[i, j] * y_old_iter_[xi_p1[i, j], eta_0_idx] +
                            b_sn[i, j] * y_old_iter_[xi_0_idx, eta_m1[i, j]] +
                            b_sn[i, j] * y_old_iter_[xi_0_idx, eta_p1[i, j]] +
                            c_py[i, j]
                        ) / b_p[i, j]
                
                # ----------------Here are the method not supported by numba---------------
                # c_px = - beta * (x_old_iter_[xi_p1, eta_p1] - x_old_iter_[xi_m1, eta_p1] +
                #                 x_old_iter_[xi_m1, eta_m1] - x_old_iter_[xi_p1, eta_m1]) / 2
                # c_py = - beta * (y_old_iter_[xi_p1, eta_p1] - y_old_iter_[xi_m1, eta_p1] +
                #                 y_old_iter_[xi_m1, eta_m1] - y_old_iter_[xi_p1, eta_m1]) / 2
                # x_new[xi_0, eta_0] = (b_we * x_old_iter_[xi_m1, eta_0] + b_we * x_old_iter_[xi_p1, eta_0] +
                #         b_sn * x_old_iter_[xi_0, eta_m1] + b_sn * x_old_iter_[xi_0, eta_p1] +
                #         c_px) / b_p
                # y_new[xi_0, eta_0] = (b_we * y_old_iter_[xi_m1, eta_0] + b_we * y_old_iter_[xi_p1, eta_0] +
                #         b_sn * y_old_iter_[xi_0, eta_m1] + b_sn * y_old_iter_[xi_0, eta_p1] +
                #         c_py) / b_p
                # -----------------Here are the method not supported by numba----------------
                
                current_max_diff = np.maximum(np.abs(x_new - x_old_iter_), np.abs(y_new - y_old_iter_))
                current_max_diff = np.max(current_max_diff)
                x = (alpha) * x_new + (1 - alpha) * x_old_iter_
                y = (alpha) * y_new + (1 - alpha) * y_old_iter_
                x_xi_, x_eta_, y_xi_, y_eta_, J_jacobian_ = comput_drv_phy_to_com_and_jcbn(x, y, J_jacobian)
                if current_max_diff > max_diff_iter:
                    max_diff_iter = current_max_diff
                
                if iteration % 200 == 0 or iteration == max_iterations -1 :  # Periodic progress reporting
                    iteration_str = (iteration + 1)
                    max_diff_str = max_diff_iter
                    print("Iteration", iteration_str, "Max difference:", max_diff_str)

                if max_diff_iter < tolerance_:
                    fin_iter = iteration + 1
                    max_diff_fin = max_diff_iter
                    print("Converged after", fin_iter, "iterations. Max difference:" , max_diff_fin)
                    break
            return x, y, x_xi_, y_xi_, x_eta_, y_eta_, J_jacobian_ 
        
        def main_loop(x, y, x_xi, y_xi, x_eta, y_eta, J_jacobian, max_iterations=max_iterations, tolerance=tolerance):
            self.x, self.y, self.x_xi, self.y_xi, self.x_eta, self.y_eta, self.J_jacobian = \
            sub_loop(x, y, x_xi, y_xi, x_eta, y_eta, J_jacobian, max_iterations, tolerance)
            
        main_loop(x, y, x_xi, y_xi, x_eta, y_eta, J_jacobian)
  
    def compute_derivatives_phy_to_com_and_jacobian(self):
        """
        Compute partial derivatives of x,y w.r.t xi,eta and Jacobian J.
        Uses central differences for interior points, second-order one-sided for boundaries.
        Results stored in self.x_xi, self.x_eta, self.y_xi, self.y_eta, self.J_jacobian
        """
        # if self.x is None or self.y is None:
        #     print("Physical coordinates (x,y) not available. Run solver first.")
        #     return
        # if jnp.sum(self.x**2) == 0 and jnp.sum(self.y**2) == 0 :  # Simple check for initial zeros
        #      print("Warning: Physical coordinates may not be properly computed (possibly still initial zeros). Derivatives may be meaningless.")


        self.x_xi = np.zeros((self.NI, self.NJ), dtype=float)
        self.x_eta = np.zeros((self.NI, self.NJ), dtype=float)
        self.y_xi = np.zeros((self.NI, self.NJ), dtype=float)
        self.y_eta = np.zeros((self.NI, self.NJ), dtype=float)
        self.J_jacobian = np.zeros((self.NI, self.NJ), dtype=float)

        # static parameters for numba
        NI = self.NI
        NJ = self.NJ
        x = self.x
        y = self.y
        x_xi = self.x_xi
        y_xi = self.y_xi
        x_eta = self.x_eta
        y_eta = self.y_eta
        # set as writable
        x_xi.setflags(write=True)
        y_xi.setflags(write=True)
        x_eta.setflags(write=True)
        y_eta.setflags(write=True)
        
        
        @jit
        def main_loop(x_xi, y_xi, x_eta, y_eta, d_xi=1.0, d_eta=1.0):
            J_jacobian = np.zeros((NI, NJ), dtype=float)
            for i in prange(NI):
                for j in prange(NJ):
                    # --- Xi derivatives (circumferential) ---
                    # Use central differences with periodic boundary handling
                    ip1 = (i + 1) % NI
                    im1 = (i - 1 + NI) % NI
                    
                    x_xi[i, j] = (x[ip1, j] - x[im1, j]) / (2.0 * d_xi)
                    y_xi[i, j] = (y[ip1, j] - y[im1, j]) / (2.0 * d_xi)

                    # --- Eta derivatives (radial) ---
                    # NJ guaranteed >= 2
                    if NJ == 2:  # Only two lines (j=0, j=1)
                        # Use first-order one-sided differences. d_eta_norm = 1.0/(2-1) = 1.0
                        if j == 0:  # Inner boundary (j=0)
                            x_eta[i, j] = (x[i, j + 1] - x[i, j]) / d_eta
                            y_eta[i, j] = (y[i, j + 1] - y[i, j]) / d_eta
                        else:  # Outer boundary (j=1)
                            x_eta[i, j] = (x[i, j] - x[i, j - 1]) / d_eta
                            y_eta[i, j] = (y[i, j] - y[i, j - 1]) / d_eta
                    else:  # NJ >= 3, can use second-order one-sided on boundaries
                        if j == 0:  # Inner boundary (j=0, eta=0) - second-order forward
                            # f'(x0) = (-3f0 + 4f1 - f2)/(2h)
                            x_eta[i, j] = (-3.0 * x[i, 0] + 4.0 * x[i, 1] - x[i, 2]) / (2.0 * d_eta)
                            y_eta[i, j] = (-3.0 * y[i, 0] + 4.0 * y[i, 1] - y[i, 2]) / (2.0 * d_eta)
                        elif j == NJ - 1:  # Outer boundary (j=NJ-1, eta=1) - second-order backward
                            # f'(x_N) = (3f_N -4f_{N-1} +f_{N-2})/(2h)
                            x_eta[i, j] = (3.0 * x[i, NJ - 1] - 4.0 * x[i, NJ - 2] + x[i, NJ - 3]) / (2.0 * d_eta)
                            y_eta[i, j] = (3.0 * y[i, NJ - 1] - 4.0 * y[i, NJ - 2] + y[i, NJ - 3]) / (2.0 * d_eta)
                        else:  # Interior points (0 < j < NJ-1) - central differences
                            x_eta[i, j] = (x[i, j + 1] - x[i, j - 1]) / (2.0 * d_eta)
                            y_eta[i, j] = (y[i, j + 1] - y[i, j - 1]) / (2.0 * d_eta)
                    
                    # --- Compute Jacobian determinant J ---
                    # J = x_xi * y_eta - x_eta * y_xi
                    J_jacobian[i, j] = x_xi[i, j] * y_eta[i, j] - x_eta[i, j] * y_xi[i, j]
            
            J_jacobian = np.where(J_jacobian == 0, 1e-8, J_jacobian)  # Avoid division by zero
            return x_xi, y_xi, x_eta, y_eta, J_jacobian
        self.x_xi, self.y_xi, self.x_eta, self.y_eta, self.J_jacobian = \
        main_loop(x_xi, y_xi, x_eta, y_eta)
        
        # print("Derivatives (x_xi, x_eta, y_xi, y_eta) and Jacobian computed and stored")
    
    def compute_derivatives_com_to_phy(self):
        self.xi_x = self.y_eta / self.J_jacobian
        self.xi_y = -self.x_eta / self.J_jacobian
        self.eta_x = -self.y_xi / self.J_jacobian
        self.eta_y = self.x_xi / self.J_jacobian
        # print("Computed derivatives from computational to physical coordinates (xi_x, xi_y, eta_x, eta_y)")
    
    def output_to_file(self, filename):
        """Output grid coordinates to a file"""
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
    
    def plot_physical_grid(self, show_points=False):
        """Plot generated grid in physical (x,y) plane"""
        if 'matplotlib' not in globals() and 'plt' not in globals(): 
            print("Matplotlib not imported. Cannot plot grid")
            return
        x_phys = self.x
        y_phys = self.y
        plt.figure(figsize=(10, 8))
        
        # Plot constant-eta lines (look like "rings" or "shells" in O-grid)
        # self.x[:, j] is line of constant j (eta)
        for j in range(self.NJ):  # For each eta=constant line
            # Connect last point to first to close O-grid loop
            line_x = np.append(x_phys[:, j], x_phys[0, j])
            line_y = np.append(y_phys[:, j], y_phys[0, j])
            plt.plot(line_x, line_y, 'b-', linewidth=0.8, label='Eta lines' if j == 0 else "")


        # Plot constant-xi lines (look like "radial lines" or "spokes")
        # self.x[i, :] is line of constant i (xi)
        for i in range(self.NI):  # For each xi=constant line
            plt.plot(x_phys[i, :], y_phys[i, :], 'r-', linewidth=0.8, label='Xi lines' if i == 0 else "")
        
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

        xi_comp = self.xi_comp
        eta_comp = self.eta_comp
        plt.figure(figsize=(fig_width, fig_height if fig_height > 1 else 2))  # Ensure reasonable height
        
        # Plot constant-eta lines (horizontal)
        for j in range(self.NJ):
            plt.plot(xi_comp[:, j], eta_comp[:, j], 'b-', linewidth=0.8)

        # Plot constant-xi lines (vertical)
        for i in range(self.NI):
            plt.plot(xi_comp[i, :], eta_comp[i, :], 'r-', linewidth=0.8)
        
        plt.xlabel("ξ (Computational)")
        plt.ylabel("η (Computational)")
        plt.title("Computational Grid (Normalized)")
        plt.grid(True, linestyle=':', alpha=0.5)
        # Set axis ranges to better display normalized coordinates
        plt.xlim([-0.05, 1.05 if self.NI == 1 else (self.NI-1)/self.NI + 0.05])  # xi from 0 to (NI-1)/NI
        plt.ylim([-0.05, 1.05])  # eta from 0 to 1
        plt.gca().set_aspect('auto', adjustable='box')  # Or 'equal' for equal aspect ratio
        plt.show()

if __name__ == '__main__':
    # define inner and outer boundary functions
    # inner boundary: circle with radius 1.0
    # outer boundary: circle with radius 3.0
    # t ranges from 0 to 1
    
    def outer_circle_boundary(t, radius=80.0, center_x=0.0, center_y=0.0):
        angle = 2 * np.pi * t
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        return x, y

    # ellipse boundary
    def inner_ellipse_boundary(t, a=2.0, b=0.8, center_x=0.0, center_y=0.0): 
        angle = 2 * np.pi * t
        x = center_x + a * np.cos(angle)
        y = center_y + b * np.sin(angle)
        return x, y

    # NACA0012 airfoil boundary
    def naca0012_airfoil(t, chord_length=1.0):
        """closed NACA0012 airfoil profile"""
        max_thickness = 0.12 * chord_length  # max thickness
        # split the t range into two halves
        t_scaled = 2 * t  # scale t to [0, 2]
        
        if np.isclose(t, 1.0, atol=1e-12):
            return 1.0, 0.0  # force the trailing edge to be at (1,0)
        
        t_scaled = 2 * t
        if t_scaled <= 1.0:
            x = 1 - t_scaled  # upper surface
            y_sign = 1
        else:
            x = t_scaled - 1  # lower surface
            y_sign = -1
        
        # front point as 0.0
        if np.isclose(x, 0.0, atol=1e-6):
            return 0.0, 0.0
        
        # compute thickness distribution
        yt = (0.12/0.2) * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        return x * chord_length, y_sign * yt * chord_length

    # mesh generation parameters
    NI_points = 200  # angular points number
    NJ_points = 100  # axial points number
    alpha = 0.8     # relaxation factor
    tol = 1e-5      # convergence tolerance
    symmetric = True  # symmetric endpoint for inner boundary
    chord_length = 1.0  # chord length for NACA0012 airfoil
    radius = 20.0  # radius for outer circle boundary
    max_iterations = 200000  # max iterations for Laplace solver
    compute = True  # compute the grid or not
    

    print(f"generating O type mesh, NI={NI_points}, NJ={NJ_points}...")

    # === ellipse as inner boundary, circle as outer ===
    generator = OGridLaplaceGenerator(NI_points, NJ_points,
                                      inner_boundary_func=lambda t: naca0012_airfoil(t, chord_length=chord_length),
                                      outer_boundary_func=lambda t: outer_circle_boundary(t, radius=radius,
                                                                                          center_x=0.5 * chord_length,
                                                                                          center_y=0.0),
                                      symmetric=symmetric)


    print("\ninitializing grid...")
    
    # print boundary points
    # generator.plot_boundary(show_points=True)
    
    # print initial guess
    # generator.plot_init_guess(show_points=True)
    
    if compute:
        # mesh generation by solving Laplace equations
        # Jacobian iteration
        print("\nbeginning to solve Laplace equations...")
        generator.solve_laplace_equations(max_iterations=max_iterations, tolerance=tol, alpha=alpha)

        # Output mesh data to file
        # generator.output_to_file("mesh_data.txt")
        print("solve_laplace_equations() completed.")
        
        # plotting the grid
        print("\n plotting the mesh...")
        generator.plot_physical_grid(show_points=False)
        print("Plotting completed.")
