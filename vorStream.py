import numpy as np
from Diff_schme import DiffSchemes


class VorticityStreamPoiseuille(DiffSchemes):
    def __init__(self, name, dt, dx, dy, x, y, t, nu, U0, H, 
                 ini_condi=None, bnd_condi=None, folder=None):
        """
        vorticity-stream function for Poiseuille
        """
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, 
                         ini_condi=ini_condi, bnd_condi=bnd_condi, folder=folder)
        self.nu = nu        # dynamic viscosity
        self.U0 = U0        # centerline velocity 
        self.H = H          # channel height
        self.Re = U0 * H / nu  # Reynolds number 
        self.ny = len(y)    # y mesh number 
        
        # initialize fields (nx, ny)
        self.psi = None
        self.vorticity = None
        self.u = None
        self.v = None
        self.initialize_fields()
        
        # apply BDC
        self.set_boundary_conditions()

    def initialize_fields(self):
        """initialize stream function and vorticity"""
        nx, ny = len(self.x), len(self.y)
        self.psi = np.zeros((nx, ny))
        self.vorticity = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        
        # uniform inlet and initial condition
        self.u = self.U0 * np.ones((nx, ny))  # initial value
        self.psi = np.cumsum(self.u, axis=0) * self.dy  # integrate for initial stream function
        
    def set_boundary_conditions(self):
        """BDC for 2D Poiseuille flow"""
        # upper and lower wall conditions
        self.psi[:, 0] = 0
        self.psi[:, -1] = self.psi[0, -1]
        
        self.u[:, -1] = 0
        self.u[:, 0] = 0  
        self.v[:, 0] = 0
        self.v[:, -1] = 0  
        
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
        """steady vorStrm solution"""
        for iter in range(max_iter):
            psi_old = self.psi.copy()
            
            # solve vorticity transport
            self.solve_vorticity_transport()
            
            # solve Strm by poisson equation
            self.solve_psi_poisson()
            
            # apply BDC
            self.set_boundary_conditions()
            
            # check convergency
            diff = np.max(np.abs(self.psi - psi_old))
            if iter % 500 == 0:
                print(f"Iter {iter}: diff={diff:.2e}")
            
            if diff < tol:
                print(f"Converged after {iter} iterations")
                break
        else:
            print("Reached maximum iterations")

    def solve_vorticity_transport(self, alpha_vorticity=0.5):
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
        # update values for inner points with relaxation
        new_vort[1:-1,1:-1] = self.vorticity[1:-1, 1:-1] + alpha_vorticity * omega_change
        self.vorticity[1:-1, 1:-1] = new_vort[1:-1, 1:-1]

    def solve_psi_poisson(self, max_iter=100, tol=1e-4, alpha_psi=0.5):
        """solve Poisson equation for stream function"""
        for _ in range(max_iter):
            psi_old = self.psi.copy()
            
            # container for [1:-1, 1:-1] points
            psi_new = np.copy(self.psi)
            psi_new[1:-2, 1:-1] = 0.5 * (self.dx ** (-2) + self.dy ** (-2)) ** -1 * (
                (self.psi[2:-1, 1:-1] + self.psi[:-3, 1:-1]) / self.dx ** 2 +
                (self.psi[1:-2, 2:] + self.psi[1:-2, :-2]) / self.dy ** 2 -
                self.vorticity[1:-2, 1:-1]
            )
            psi_new[-2, 1:-1] = 0.5 * (-self.dy ** 2 * self.vorticity[-2, 1:-1] + 
                self.psi[-2, 2:] + self.psi[-2, :-2])
            
            self.psi[1:-1, 1:-1] = alpha_psi * psi_new[1:-1, 1:-1] + (1 - alpha_psi) * self.psi[1:-1, 1:-1]
            
            # check convergency
            if np.max(np.abs(self.psi - psi_old)) < tol:
                break

    def get_velocity_field(self):
        """return velocity field (u, v)"""
        return self.u, self.v  