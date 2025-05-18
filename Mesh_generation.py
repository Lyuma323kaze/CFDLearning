import numpy as np
import matplotlib.pyplot as plt
from jax import jit, lax, debug
import jax.numpy as jnp
import os
from jax.experimental import sparse

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

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
        self.x = jnp.zeros((NI, NJ), dtype=float)
        self.y = jnp.zeros((NI, NJ), dtype=float)
        # Computational coordinates (unnormalized)
        self.xi, self.eta = jnp.meshgrid(jnp.arange(self.NI), jnp.arange(self.NJ), indexing='ij')
        
        # Computational coordinates (normalized) for generating BDC
        # xi_comp will vary from 0 to (NI-1)/NI
        # eta_comp will vary from 0 to 1
        self.xi_comp = jnp.zeros((NI, NJ), dtype=float)
        self.eta_comp = jnp.zeros((NI, NJ), dtype=float)
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
        
        # label for validation
        # self._is_valid = False
        
        self._initialize_boundaries()
        self._initialize_interior_guess()

    def _initialize_boundaries(self):
        """Set x,y coordinates for inner boundary (eta=0) and outer boundary (eta=NJ-1)"""
        # t_params are values for boundary function parameter 't'
        # These correspond to our normalized xi coordinates
        t_params = jnp.linspace(0, 1, self.NI, endpoint=False)  # NI points: 0, 1/NI, ..., (NI-1)/NI

        for i in range(self.NI):
            # ti is equivalent to self.xi_comp[i, 0] or self.xi_comp[i, self.NJ-1]
            ti = t_params[i] 
            
            # Inner boundary (j=0)
            self.x.at[i, 0].set(self.inner_boundary_func(ti)[0])
            self.y.at[i, 0].set(self.inner_boundary_func(ti)[1])
            
            # Outer boundary (j=NJ-1)
            self.x.at[i, self.NJ - 1].set(self.outer_boundary_func(ti)[0])
            self.y.at[i, self.NJ - 1].set(self.outer_boundary_func(ti)[1])

    def _initialize_interior_guess(self):
        """Initialize interior grid points using linear interpolation"""
        for j in range(1, self.NJ - 1):
            eta = self.eta_comp[0, j]
            for i in range(self.NI):
                xi = self.xi_comp[i, j]
                # Boundary contributions
                U_0 = jnp.array([self.x[i, 0], self.y[i, 0]])
                U_1 = jnp.array([self.x[i, -1], self.y[i, -1]])
                V_0 = jnp.array([self.x[0, j], self.y[0, j]])
                V_1 = jnp.array([self.x[-1, j], self.y[-1, j]])

                C_00 = jnp.array([self.x[0, 0], self.y[0, 0]])
                C_10 = jnp.array([self.x[-1, 0], self.y[-1, 0]])
                C_01 = jnp.array([self.x[0, -1], self.y[0, -1]])
                C_11 = jnp.array([self.x[-1, -1], self.y[-1, -1]])
                # TFI formula
                term1 = (1 - eta) * U_0 + eta * U_1
                term2 = (1 - xi) * V_0 + xi * V_1
                term3 = (1 - xi) * (1 - eta) * C_00 + xi * (1 - eta) * C_10 + \
                        (1 - xi) * eta * C_01 + xi * eta * C_11

                # TFI formula
                result = term1 + term2 - term3
                self.x.at[i, j].set(result[0])
                self.y.at[i, j].set(result[1])

    # TODO: jit to be added
    def solve_laplace_equations(self, max_iterations=10000, tolerance=1e-6):
        self.compute_derivatives_phy_to_com_and_jacobian()
        # 2d arrays for xi, eta indices (shape = (NI, NJ))
        idx_xi, idx_eta = self.xi, self.eta
        # Jacobian iteration
        @jit
        def main_loop(idx_xi_, idx_eta_):
            for iteration in range(max_iterations):
                # derivatives in inner region
                x_xi = self.x_xi[:, 1:-1]
                x_eta = self.x_eta[:, 1:-1]
                y_xi = self.y_xi[:, 1:-1]
                y_eta = self.y_eta[:, 1:-1]
                # values of last iteration
                x_old_iter = self.x.copy()
                y_old_iter = self.y.copy()
                x_new = jnp.zeros_like(self.x)
                x_new = x_new.at[:, 0].set(x_old_iter[:, 0])  # inner boundary
                x_new = x_new.at[:, -1].set(x_old_iter[:, -1])  # outer boundary
                y_new = jnp.zeros_like(self.y)
                y_new = y_new.at[:, 0].set(y_old_iter[:, 0])  # inner boundary
                y_new = y_new.at[:, -1].set(y_old_iter[:, -1])  # outer boundary
                max_diff_iter = 0.0
                # the transitioned indices
                xi_p1 = jnp.roll(idx_xi_, -1, axis=0)[:, 1:-1]
                xi_m1 = jnp.roll(idx_xi_, 1, axis=0)[:, 1:-1]
                xi_0 = idx_xi_[:, 1:-1]
                eta_p1 = idx_eta_[:, 2:]
                eta_m1 = idx_eta_[:, :-2]
                eta_0 = idx_eta_[:, 1:-1]
                # coefficients
                b_we = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                b_sn = x_xi ** 2 + y_xi ** 2
                b_p = 2 * b_we + 2 * b_sn
                b_p = jnp.where(b_p == 0, 1e-8, b_p)  # Avoid division by zero
                # here I would like to generate a reduced matrix, shape = (NI, NJ-2), and the 2 indices are
                # given by the xi and eta indices.
                c_px = - beta * (x_old_iter[xi_p1, eta_p1] - x_old_iter[xi_m1, eta_p1] +
                                x_old_iter[xi_m1, eta_m1] - x_old_iter[xi_p1, eta_m1]) / 2
                c_py = - beta * (y_old_iter[xi_p1, eta_p1] - y_old_iter[xi_m1, eta_p1] +
                                y_old_iter[xi_m1, eta_m1] - y_old_iter[xi_p1, eta_m1]) / 2
                # Update internal points using at[].set()
                x_new = x_new.at[xi_0, eta_0].set(
                    (b_we * x_old_iter[xi_m1, eta_0] + b_we * x_old_iter[xi_p1, eta_0] +
                    b_sn * x_old_iter[xi_0, eta_m1] + b_sn * x_old_iter[xi_0, eta_p1] +
                    c_px) / b_p
                )
                y_new = y_new.at[xi_0, eta_0].set(
                    (b_we * y_old_iter[xi_m1, eta_0] + b_we * y_old_iter[xi_p1, eta_0] +
                    b_sn * y_old_iter[xi_0, eta_m1] + b_sn * y_old_iter[xi_0, eta_p1] +
                    c_py) / b_p
                )
                current_max_diff = jnp.maximum(jnp.abs(x_new - x_old_iter), jnp.abs(y_new - y_old_iter))
                current_max_diff = jnp.max(current_max_diff)
                self.x = x_new
                self.y = y_new
                self.compute_derivatives_phy_to_com_and_jacobian()
                
                max_diff_iter = jnp.maximum(max_diff_iter, current_max_diff)
                
                print_condition = (iteration % 200 == 0) | (iteration == max_iterations - 1)
                _ = lax.cond(
                    print_condition,
                    lambda: debug.print("Iteration {iter}/{max}, Max diff: {diff:.2e}",
                                        iter=iteration+1, max=max_iterations, diff=max_diff_iter),
                    lambda: None,
                )

                if max_diff_iter < tolerance:
                    print(f"Converged after {iteration + 1} iterations. Max difference: {max_diff_iter:.2e}")
                    break
        main_loop(idx_xi, idx_eta)
    
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


        self.x_xi = jnp.zeros((self.NI, self.NJ), dtype=float)
        self.x_eta = jnp.zeros((self.NI, self.NJ), dtype=float)
        self.y_xi = jnp.zeros((self.NI, self.NJ), dtype=float)
        self.y_eta = jnp.zeros((self.NI, self.NJ), dtype=float)
        self.J_jacobian = jnp.zeros((self.NI, self.NJ), dtype=float)

        # Step sizes in computational coordinates
        # xi_comp values are k/NI, k = 0, ..., NI-1. So step size is 1/NI
        d_xi = 1.0
        
        # eta_comp values are k/(NJ-1), k = 0, ..., NJ-1. So step size is 1/(NJ-1)
        # self.NJ guaranteed > 1 (controlled by __init__)
        d_eta = 1.0

        @jit
        def main_loop(d_xi_, d_eta_):
            for i in range(self.NI):
                for j in range(self.NJ):
                    # --- Xi derivatives (circumferential) ---
                    # Use central differences with periodic boundary handling
                    ip1 = (i + 1) % self.NI
                    im1 = (i - 1 + self.NI) % self.NI
                    
                    self.x_xi.at[i, j].set((self.x[ip1, j] - self.x[im1, j]) / (2.0 * d_xi_))
                    self.y_xi.at[i, j].set((self.y[ip1, j] - self.y[im1, j]) / (2.0 * d_xi_))

                    # --- Eta derivatives (radial) ---
                    # self.NJ guaranteed >= 2
                    if self.NJ == 2:  # Only two lines (j=0, j=1)
                        # Use first-order one-sided differences. d_eta_norm = 1.0/(2-1) = 1.0
                        if j == 0:  # Inner boundary (j=0)
                            self.x_eta.at[i, j].set((self.x[i, j + 1] - self.x[i, j]) / d_eta_)
                            self.y_eta.at[i, j].set((self.y[i, j + 1] - self.y[i, j]) / d_eta_)
                        else:  # Outer boundary (j=1)
                            self.x_eta.at[i, j].set((self.x[i, j] - self.x[i, j - 1]) / d_eta_)
                            self.y_eta.at[i, j].set((self.y[i, j] - self.y[i, j - 1]) / d_eta_)
                    else:  # self.NJ >= 3, can use second-order one-sided on boundaries
                        if j == 0:  # Inner boundary (j=0, eta=0) - second-order forward
                            # f'(x0) = (-3f0 + 4f1 - f2)/(2h)
                            self.x_eta.at[i, j].set((-3.0 * self.x[i, 0] + 4.0 * self.x[i, 1] - self.x[i, 2]) / (2.0 * d_eta_))
                            self.y_eta.at[i, j].set((-3.0 * self.y[i, 0] + 4.0 * self.y[i, 1] - self.y[i, 2]) / (2.0 * d_eta_))
                        elif j == self.NJ - 1:  # Outer boundary (j=NJ-1, eta=1) - second-order backward
                            # f'(x_N) = (3f_N -4f_{N-1} +f_{N-2})/(2h)
                            self.x_eta.at[i, j].set((3.0 * self.x[i, self.NJ - 1] - 4.0 * self.x[i, self.NJ - 2] + self.x[i, self.NJ - 3]) / (2.0 * d_eta_))
                            self.y_eta.at[i, j].set((3.0 * self.y[i, self.NJ - 1] - 4.0 * self.y[i, self.NJ - 2] + self.y[i, self.NJ - 3]) / (2.0 * d_eta_))
                        else:  # Interior points (0 < j < NJ-1) - central differences
                            self.x_eta.at[i, j].set((self.x[i, j + 1] - self.x[i, j - 1]) / (2.0 * d_eta_))
                            self.y_eta.at[i, j].set((self.y[i, j + 1] - self.y[i, j - 1]) / (2.0 * d_eta_))
                    
                    # --- Compute Jacobian determinant J ---
                    # J = x_xi * y_eta - x_eta * y_xi
                    self.J_jacobian.at[i, j].set(self.x_xi[i, j] * self.y_eta[i, j] - self.x_eta[i, j] * self.y_xi[i, j]) 
            self.J_jacobian = jnp.where(self.J_jacobian == 0, 1e-8, self.J_jacobian)  # Avoid division by zero
        main_loop(d_xi, d_eta)
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
        x_phys = self.x.numpy()
        y_phys = self.y.numpy()
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

        xi_comp = self.xi_comp.numpy()
        eta_comp = self.eta_comp.numpy()
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
    @jit
    def inner_circle_boundary(t, radius=1.0, center_x=0.0, center_y=0.0):
        angle = 2 * jnp.pi * t
        x = center_x + radius * jnp.cos(angle)
        y = center_y + radius * jnp.sin(angle)
        return x, y

    @jit
    def outer_circle_boundary(t, radius=3.0, center_x=0.0, center_y=0.0):
        angle = 2 * jnp.pi * t
        x = center_x + radius * jnp.cos(angle)
        y = center_y + radius * jnp.sin(angle)
        return x, y

    # ellipse boundary
    @jit
    def inner_ellipse_boundary(t, a=2.0, b=0.8, center_x=0.0, center_y=0.0): # a: 半长轴, b: 半短轴
        angle = 2 * jnp.pi * t
        x = center_x + a * jnp.cos(angle)
        y = center_y + b * jnp.sin(angle)
        return x, y

    # mesh generation parameters
    NI_points = 80  # angular points number
    NJ_points = 40  # axial points number

    print(f"generating O type mesh, NI={NI_points}, NJ={NJ_points}...")

    # === ellipse as inner boundary, circle as outer ===
    generator = OGridLaplaceGenerator(NI_points, NJ_points,
                                      inner_boundary_func=lambda t: inner_ellipse_boundary(t, a=2.0, b=1),
                                      outer_boundary_func=lambda t: outer_circle_boundary(t, radius=40.0))


    print("\ninitializing grid...")
    
    # mesh generation by solving Laplace equations
    # Jacobian iteration
    print("\nbeginning to solve Laplace equations...")
    generator.solve_laplace_equations(max_iterations=30000, tolerance=1e-8)

    # Output mesh data to file
    generator.output_to_file("mesh_data.txt")
    print("solve_laplace_equations() completed.")
    
    # plotting the grid
    print("\n plotting the mesh...")
    generator.plot_physical_grid(show_points=False)
    print("Plotting completed.")
