from Diff_schme import DiffSchemes
import numpy as np
import pyamg
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

class CavitySIMPLE(DiffSchemes):
    def __init__(self, name, dt, dx, x, t, dy, y, Re, U_top, 
                 max_iter=1000,
                 tol=1e-5,
                 alpha_u=1,
                 alpha_v=0.1,
                 alpha_p=0.8,
                 **kwargs):
        super().__init__(name, dt, dx, x, t, dy=dy, y=y, **kwargs)
        self.Re = Re          # kinematic viscosity
        self.U_top = U_top    # upper lid velocity
        
        # mesh parameters
        self.nx = len(x)      # x prime point number
        self.ny = len(y)      # y prime point number
        self.dx = dx
        self.dy = dy
        # lack relaxation factors
        self.alpha_u = alpha_u  # velocity lack relaxation
        self.alpha_v = alpha_v
        self.alpha_p = alpha_p  # pressure lack relaxation
        
        # mesh
        # pressure (prime)
        self.p = np.zeros((self.nx, self.ny))
        
        # u mesh
        self.u = 0 * np.ones((self.nx+1, self.ny+2))
        # self.u[:, 1:-1] = 0.1 * (U_top * (2 * self.y[np.newaxis, :] - 1))
        
        # v mesh
        self.v = np.zeros((self.nx+2, self.ny+1))
        # self.v[1:-1, :] = 0.1 * U_top * (1 - 2 * self.x[:, np.newaxis])
        
        # modification variables
        self.u_star = np.copy(self.u)
        self.v_star = np.copy(self.v)
        self.p_prime = np.zeros((self.nx, self.ny))
        # displaced p values used in p_prime computation
        self.p_prime_u = np.zeros_like(self.p_prime)
        self.p_prime_d = np.zeros_like(self.p_prime)
        self.p_prime_l = np.zeros_like(self.p_prime)
        self.p_prime_r = np.zeros_like(self.p_prime)
        
        # convergency
        self.max_iter = max_iter
        self.tol = tol
        self.chat = 1.
        self.res = 1.

        # BDC
        self.apply_boundary_conditions()
        self.apply_boundary_conditions_star()

    def apply_boundary_conditions(self):
        """Set boundary values"""
        # bottom fixed (no slip)
        # self.u[:, 0] = -self.u[:, 1]   # u=0 by virtual node
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0   # v=0
        
        # left side fixed (no slip)
        self.u[0, :] = 0.0   # u=0
        # self.v[0, :] = -self.v[1, :]   # v=0 by virtual node
        self.v[0, :] = 0.0   # v=0 by virtual node
        
        # right side fixed (no slip)
        self.u[-1, :] = 0.0  # u=0
        # self.v[-1, :] = -self.v[-2, :]  # v=0 by virtual node
        self.v[-1, :] = 0.0  # v=0 by virtual node
        
        # Upper lid moving (u velocity)
        # self.u[:, -1] = 2 * self.U_top - self.u[:, -2]  # upper x cover by virtual node
        self.u[:, -1] = self.U_top  # upper x cover by virtual node
        self.v[:, -1] = 0.0         # v velocity on the top

    def apply_boundary_conditions_star(self):
        """Set boundary values for prediction values"""
        # bottom fixed (no slip)
        # self.u_star[:, 0] = -self.u_star[:, 1]   # u=0 by virtual node
        self.u_star[:, 0] = 0.0
        self.v_star[:, 0] = 0.0   # v=0
        
        # left side fixed (no slip)
        self.u_star[0, :] = 0.0   # u=0
        # self.v_star[0, :] = -self.v_star[1, :]   # v=0 by virtual node
        self.v_star[0, :] = 0.0
        
        # right side fixed (no slip)
        self.u_star[-1, :] = 0.0  # u=0
        # self.v_star[-1, :] = -self.v_star[-2, :]  # v=0 by virtual node
        self.v_star[-1, :] = 0.0
        
        # upper lid moving (u velocity)
        # self.u_star[:, -1] = 2 * self.U_top - self.u_star[:, -2]  # upper x cover by virtual node
        self.u_star[:, -1] = self.U_top  # upper x cover by virtual node
        self.v_star[:, -1] = 0.0         # v velocity on the top

    def get_transitioned(self):
        """update the virtual p_primes"""
        self.p_prime_u[:, :-1] = self.p_prime[:, 1:]
        self.p_prime_u[:, -1] = self.p_prime[:, -1]
        
        self.p_prime_d[:, 1:] = self.p_prime[:, :-1]
        self.p_prime_d[:, 0] = self.p_prime[:, 0]
        
        self.p_prime_l[1:] = self.p_prime[:-1]
        self.p_prime_l[0] = self.p_prime[0]
        
        self.p_prime_r[:-1] = self.p_prime[1:]
        self.p_prime_r[-1] = self.p_prime[-1]
        return

    def get_inv_val(self, c_p, c_ew, c_ns):
        inv_c_p = 1. / (c_p + 1e-12)
        inv_c_l = 1. / ((c_ew[0,1:-1] +
                        c_ns[0,1:] + 
                        c_ns[0,:-1]) + 1e-12)
        inv_c_r = 1. / ((c_ew[-1,1:-1] +
                            c_ns[-1,1:] +
                            c_ns[-1,:-1]) + 1e-12)
        inv_c_u = 1. / ((c_ns[1:-1,-1] +
                        c_ew[:-1,-1] +
                        c_ew[1:,-1]) + 1e-12)
        inv_c_d = 1. / ((c_ns[1:-1,0] +
                        c_ew[:-1,0] +
                        c_ew[1:,0]) + 1e-12)
        inv_c_lu = 1. / ((c_ew[0,-1] + c_ns[0,-1]) + 1e-12)
        inv_c_ld = 1. / ((c_ew[0,0] + c_ns[0,0]) + 1e-12)
        inv_c_ru = 1. / ((c_ew[-1,-1] + c_ns[-1,-1]) + 1e-12)
        inv_c_rd = 1. / ((c_ew[-1,0] + c_ns[-1,0]) + 1e-12)
        
        return (inv_c_p, inv_c_l, inv_c_r, inv_c_u, inv_c_d, 
                inv_c_lu, inv_c_ld, inv_c_ru, inv_c_rd)

    def build_A_matrix(self, c_ew, c_ns):
        """
        construct sparse coef matrix A for p' (5 point form)
        input:
            c_ew: (nx-1, ny)
            c_ns: (nx, ny-1)
        return:
            A: (nx*ny, nx*ny) sparse matrix
        """
        nx, ny = c_ns.shape
        N = nx * ny

        def idx(i, j):
            return i * ny + j

        # 网格内所有点坐标
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        center = idx(I, J).flatten()

        data = []
        row = []
        col = []

        # 对角线项
        diag = np.zeros((nx, ny))
        if nx > 1:
            diag += np.pad(c_ew[:-1, :], ((1, 0), (0, 0)), 'constant')
            diag += np.pad(c_ew, ((0, 1), (0, 0)), 'constant')
        if ny > 1:
            diag += np.pad(c_ns[:, :-1], ((0, 0), (1, 0)), 'constant')
            diag += np.pad(c_ns, ((0, 0), (0, 1)), 'constant')

        data.append(diag.flatten())
        row.append(center)
        col.append(center)

        # 左邻点（西）
        if nx > 1:
            left = idx(I[1:], J[1:]).flatten()
            center_left = idx(I[1:], J[1:]-1).flatten()
            row.append(center_left)
            col.append(left)
            data.append((-c_ew[:-1, 1:]).flatten())

        # 右邻点（东）
        if nx > 1:
            right = idx(I[:-1], J[:-1]).flatten()
            center_right = idx(I[:-1], J[:-1]+1).flatten()
            row.append(center_right)
            col.append(right)
            data.append((-c_ew[:, :-1]).flatten())

        # 下邻点（南）
        if ny > 1:
            down = idx(I[1:-1], J[:-1]).flatten()
            center_down = idx(I[1:-1], J[1:]).flatten()
            row.append(center_down)
            col.append(down)
            data.append((-c_ns[1:-1, :-1]).flatten())

        # 上邻点（北）
        if ny > 1:
            up = idx(I[1:-1], J[1:]).flatten()
            center_up = idx(I[1:-1], J[:-1]).flatten()
            row.append(center_up)
            col.append(up)
            data.append((-c_ns[1:-1, :-1]).flatten())

        A = coo_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
                    shape=(N, N)).tocsr()
        return A
    
    def solve_momentum_u_star(self, uworder=1, iter_u=50):
        """solve u-momentum equation"""
        # the domain shape is (nx,ny), the same as the shape of self.p
        # the self.u shape is (nx+1,ny+2), the virtual nodes out of the domain with fixed boundary value
        # the self.v shape is (nx+2,ny+1), the virtual nodes out of the domain with fixed boundary value
        self.u_star = np.copy(self.u)  # initialize u_star
        
        # upwind coefficients
        # (nx,ny)
        u_avr_x = (self.u[:-1,1:-1] + self.u[1:,1:-1]) / 2 
        
        alpha_uxp = np.maximum(u_avr_x, 0)     # nx, ny
        alpha_uxm = np.minimum(u_avr_x, 0)     # nx, ny
        
        # (nx-1,ny+1)
        v_avr_x = (self.v[1:-2,:] + self.v[2:-1,:]) / 2
        
        alpha_uyp = np.maximum(v_avr_x, 0)     # nx-1,ny+1
        alpha_uym = np.minimum(v_avr_x, 0)     # nx-1,ny+1
        gamma_ux = np.zeros_like(alpha_uxp)  # nx, ny
        gamma_uy = np.zeros_like(alpha_uyp)  # nx-1,ny+1
        if uworder == 2:
            ...
        
        # discretization coefficients (nx-1,ny for n,s,e,w,p,hat)
        a_w = self.dy * (alpha_uxp[:-1,:] + 1 / (self.Re * self.dx))
        a_e = self.dy * (-alpha_uxm[1:,:] + 1 / (self.Re * self.dx))
        a_s = self.dx * (alpha_uyp[:,:-1] + 1 / (self.Re * self.dy))
        a_n = self.dx * (-alpha_uym[:,1:] + 1 / (self.Re * self.dy))
        a_n[:,-1] *= 2
        a_s[:,0] *= 2
    
        a_p = (self.dx * self.dy / self.dt) +\
                self.dy * (alpha_uxp[1:,:] - alpha_uxm[:-1,:] + (2 / (self.Re * self.dx))) +\
                self.dx * (alpha_uyp[:,1:] - alpha_uym[:,:-1] + (2 / (self.Re * self.dy)))
        a_p[:,-1] += self.dx / (self.Re * self.dy)
        a_p[:,0] += self.dx / (self.Re * self.dy)
        
        a_hat = self.dy * (gamma_ux[1:] - gamma_ux[:-1]) +\
                self.dx * (gamma_uy[:,1:] - gamma_uy[:,:-1])
        # pressure gradient(nx-1,ny)
        dP = -(self.p[1:] - self.p[:-1]) * self.dy
        value_old = np.empty_like(self.u_star)
        u_source = self.u[1:-1,1:-1]
        for _ in range(iter_u):
            # update
            np.copyto(value_old,self.u_star)
            self.u_star[1:-1,1:-1] = self.alpha_u * (
                (a_e * value_old[2:,1:-1] + 
                a_w * value_old[:-2,1:-1] +
                a_n * value_old[1:-1,2:] +
                a_s * value_old[1:-1,:-2] +
                dP + a_hat + 
                self.dy * self.dx * u_source / self.dt
                ) + (1 - self.alpha_u) / self.alpha_u * a_p * value_old[1:-1,1:-1] 
            ) / (a_p + 1e-12)
            self.apply_boundary_conditions_star()  # ensure boundary conditions are applied
            diff = np.max(np.abs(self.u_star[1:-1,1:-1] - value_old[1:-1,1:-1]))
            if diff < self.tol:
                # print('break by tol')
                break
        
        # nx-1,ny
        return a_p

    def solve_momentum_v_star(self, uworder=1, iter_v=50):
        """solve momentum equation"""
        self.v_star = np.copy(self.v)  # initialize u_star
        # upwind coefficients
        # (nx,ny)
        v_avr_y = (self.v[1:-1,:-1] + self.v[1:-1,1:]) / 2 
        
        alpha_vyp = np.maximum(v_avr_y, 0)     # nx, ny
        alpha_vym = np.minimum(v_avr_y, 0)     # nx, ny
        
        # (nx+1,ny-1)
        u_avr_y = (self.u[:,2:-1] + self.u[:,1:-2]) / 2
        
        alpha_vxp = np.maximum(u_avr_y, 0)     # nx+1, ny-1
        alpha_vxm = np.minimum(u_avr_y, 0)     # nx+1, ny-1
        gamma_vy = np.zeros_like(alpha_vyp)  # nx, ny
        gamma_vx = np.zeros_like(alpha_vxp)  # nx+1, ny-1
        if uworder == 2:
            ...
            
        # discretization coefficients (nx,ny-1 for w,e,n,s,p,hat)
        a_s = self.dx * (alpha_vyp[:,:-1] + 1 / (self.Re * self.dy))
        a_n = self.dx * (-alpha_vym[:,1:] + 1 / (self.Re * self.dy))
        a_w = self.dy * (alpha_vxp[:-1,:] + 1 / (self.Re * self.dx))
        a_e = self.dy * (-alpha_vxm[1:,:] + 1 / (self.Re * self.dx))
        a_w[0] *= 2
        a_e[-1] *= 2
        
        a_p = (self.dy * self.dx / self.dt) +\
                self.dx * (alpha_vyp[:,1:] - alpha_vym[:,:-1] + (2 / (self.Re * self.dy))) +\
                self.dy * (alpha_vxp[1:,:] - alpha_vxm[:-1,:] + (2 / (self.Re * self.dx)))
        # a_p = (self.dy * self.dx / self.dt) +\
        #         a_e + a_w + a_n + a_s
        a_p[0] += self.dy / (self.Re * self.dx)
        a_p[-1] += self.dy / (self.Re * self.dx)
        
        a_hat = self.dx * (gamma_vy[:,1:] - gamma_vy[:,:-1]) +\
                self.dy * (gamma_vx[1:,:] - gamma_vx[:-1,:])
        # print(np.max(np.abs(a_p))/np.max(np.abs(a_n)))
        # pressure gradient(nx,ny-1)
        dP = -(self.p[:,1:] - self.p[:,:-1]) * self.dx
        
        value_old = np.empty_like(self.v_star)
        v_source = self.v[1:-1,1:-1]
        for _ in range(iter_v):
            # update
            np.copyto(value_old, self.v_star)
            self.v_star[1:-1,1:-1] = self.alpha_v * (
                (a_n * value_old[1:-1,2:] + 
                a_s * value_old[1:-1,:-2] +
                a_e * value_old[2:,1:-1] +
                a_w * value_old[:-2,1:-1] +
                dP + a_hat + 
                self.dy * self.dx * v_source / self.dt
                ) + (1 - self.alpha_v) / self.alpha_v * a_p * value_old[1:-1, 1:-1] 
            ) / (a_p + 1e-12)
            self.apply_boundary_conditions_star()  # ensure boundary conditions are applied
            diff = np.max(np.abs(self.v_star[1:-1,1:-1] - value_old[1:-1,1:-1]))
            if diff < self.tol:
                break
        
        # nx,ny-1
        return a_p
    
    def solve_pressure_correction(self, a_p, b_p, iter_p=10000):
        """solve pressure correction equation"""
        # a_p is (nx-1,ny), b_p is (nx,ny-1)
        # self.u is (nx+1,ny+2) with virtual nodes, self.v is (nx+2,ny+1) with virtual nodes
        # w,e,u,d with BDC (nx,ny)
        
        # coefficients
        # (nx-1,ny)
        c_ew = self.dy ** 2 / a_p
        # (nx,ny-1)
        c_ns = self.dx ** 2 / b_p
        # (nx-2,ny-2)
        c_p = (c_ew[1:,1:-1] +
               c_ew[:-1,1:-1] +
               c_ns[1:-1,1:] +
               c_ns[1:-1,:-1])
        
        (inv_c_p,
         inv_c_l, inv_c_r, inv_c_u, inv_c_d, 
         inv_c_lu, inv_c_ld, inv_c_ru, inv_c_rd) = self.get_inv_val(c_p, c_ew, c_ns)

        c_hat = -(
            self.dy * (self.u_star[1:,1:-1] - self.u_star[:-1,1:-1]) +
            self.dx * (self.v_star[1:-1,1:] - self.v_star[1:-1,:-1])
        )
        
        value_old = np.empty_like(self.p_prime)
        self.chat = np.sum(np.abs(c_hat))
        for _ in range(iter_p):
            np.copyto(value_old, self.p_prime)
            # jacobian p_prime update with [100,100] the reference
            # inner points (nx-2,ny-2)
            self.p_prime[1:-1,1:-1] = inv_c_p * (
                c_ew[1:,1:-1] * self.p_prime[2:,1:-1] +
                c_ew[:-1,1:-1] * self.p_prime[:-2,1:-1] +
                c_ns[1:-1,1:] * self.p_prime[1:-1,2:] +
                c_ns[1:-1,:-1] * self.p_prime[1:-1,:-2] +
                c_hat[1:-1,1:-1]
            )
            # boundary points (edge)
            # left edge
            self.p_prime[0,1:-1] = inv_c_l * (
                c_ew[0,1:-1] * self.p_prime[1,1:-1] +
                c_ns[0,1:] * self.p_prime[0,2:] +
                c_ns[0,:-1] * self.p_prime[0,:-2] +
                c_hat[0,1:-1]
            )
            # right edge
            self.p_prime[-1,1:-1] = inv_c_r * (
                c_ew[-1,1:-1] * self.p_prime[-2,1:-1] +
                c_ns[-1,1:] * self.p_prime[-1,2:] +
                c_ns[-1,:-1] * self.p_prime[-1,:-2] +
                c_hat[-1,1:-1]
            )
            # lower edge
            self.p_prime[1:-1,0] = inv_c_d * (
                c_ns[1:-1,0] * self.p_prime[1:-1,1] +
                c_ew[:-1,0] * self.p_prime[:-2,0] +
                c_ew[1:,0] * self.p_prime[2:,0] +
                c_hat[1:-1,0]
            )
            # upper edge
            self.p_prime[1:-1,-1] = inv_c_u * (
                c_ns[1:-1,-1] * self.p_prime[1:-1,-2] +
                c_ew[:-1,-1] * self.p_prime[:-2,-1] +
                c_ew[1:,-1] * self.p_prime[2:,-1] +
                c_hat[1:-1,-1]
            )
            
            # boundary points (corner)
            self.p_prime[0,0] = inv_c_ld * (
                c_ew[0,0] * self.p_prime[1,0] +
                c_ns[0,0] * self.p_prime[0,1] +
                c_hat[0,0]
            )
            self.p_prime[0,-1] = inv_c_lu * (
                c_ew[0,-1] * self.p_prime[1,-1] +
                c_ns[0,-1] * self.p_prime[0,-2] +
                c_hat[0,-1]
            )
            self.p_prime[-1,0] = inv_c_rd * (
                c_ew[-1,0] * self.p_prime[-2,0] +
                c_ns[-1,0] * self.p_prime[-1,1] +
                c_hat[-1,0]
            )
            self.p_prime[-1,-1] = inv_c_ru * (
                c_ew[-1,-1] * self.p_prime[-2,-1] +
                c_ns[-1,-1] * self.p_prime[-1,-2] +
                c_hat[-1,-1]
            )
            self.p_prime[int(self.nx/2),int(self.ny/2)] = 0
            # check inner convergence
            res = np.sum(np.abs(self.p_prime - value_old))
            self.res = res
            mxm = np.max(np.abs(self.p_prime))
            if (res / mxm) < 1e-5:
                print('converged by tol')
                break
        return

    def solve_pressure_correction_amg(self, a_p, b_p):
        """solve pressure correction using PyAMG"""
        nx, ny = self.nx, self.ny
        
        # (nx-1,ny), (nx,ny-1)
        c_ew = self.dy**2 / (a_p + 1e-20)
        c_ns = self.dx**2 / (b_p + 1e-20)
        
        # construct sparse matrix A (on (nx,ny) mesh)
        N = nx * ny
        A = lil_matrix((N, N))

        def idx(i, j):
            """mapping (i,j) to row index"""
            return i * ny + j

        for i in range(nx):
            for j in range(ny):
                row = idx(i, j)
                diag = 0.0
                
                # left neighbor
                if i > 0:
                    coeff = c_ew[i - 1, j]
                    A[row, idx(i - 1, j)] = -coeff
                    diag += coeff
                # right neighbor
                if i < nx - 1:
                    coeff = c_ew[i, j]
                    A[row, idx(i + 1, j)] = -coeff
                    diag += coeff
                # lower neighbor
                if j > 0:
                    coeff = c_ns[i, j - 1]
                    A[row, idx(i, j - 1)] = -coeff
                    diag += coeff
                # upper neighbor
                if j < ny - 1:
                    coeff = c_ns[i, j]
                    A[row, idx(i, j + 1)] = -coeff
                    diag += coeff
                # diagonal
                A[row, row] = diag

        # construct RHS c_hat (nx, ny)
        c_hat = -(
            self.dy * (self.u_star[1:, 1:-1] - self.u_star[:-1, 1:-1]) +
            self.dx * (self.v_star[1:-1, 1:] - self.v_star[1:-1, :-1])
        )
        b = c_hat.reshape(-1)

        # reference added for avoiding singularity
        ref_i, ref_j = nx // 2, ny // 2
        ref_index = idx(ref_i, ref_j)
        A[ref_index, :] = 0
        A[ref_index, ref_index] = 1.0
        b[ref_index] = 0.0

     
        # solve p_prime vector
        A = A.tocsr()
        ml = pyamg.ruge_stuben_solver(A, coarse_solver='cg')
        x = ml.solve(b, tol=1e-8)

        # write back to self.p_prime
        self.p_prime[:, :] = x.reshape((nx, ny))

    def correct_velocity_pressure(self, a_p, b_p):
        """modify velocity and pressure based on pressure correction"""
        # pressure correction with relaxation
        self.p += self.alpha_p * self.p_prime
        # modify u with relaxation
        # print(self.u_star.shape, self.p_prime.shape, a_e.shape)
        self.u[1:-1,1:-1] = (self.u_star[1:-1,1:-1] - self.dy *\
            (self.p_prime[1:] - self.p_prime[:-1]) / a_p)
        # modify v with relaxation
        # print(self.v_star.shape, self.v.shape, a_e.shape, b_n.shape)
        self.v[1:-1,1:-1] = (self.v_star[1:-1,1:-1] - self.dx *\
            (self.p_prime[:,1:] - self.p_prime[:,:-1]) / b_p)
        self.apply_boundary_conditions()  # apply BDC
        
    def solve(self, uworder=1, tune=True, amg=True):
        """SIMPLE main loop"""
        if tune:
            prt = 20
        else:
            prt = 200
        # BDC
        self.apply_boundary_conditions()
        self.apply_boundary_conditions_star()
        
        print('time vs Re')
        print(
            (self.dx * self.dy / self.dt) * self.Re
        )
        if amg:
            for iter in range(self.max_iter):
                # velocity old values
                u_old = np.copy(self.u)
                v_old = np.copy(self.v)
                
                # SIMPLE steps
                a_p = self.solve_momentum_u_star(uworder=uworder)        # solve u*
                b_p = self.solve_momentum_v_star(uworder=uworder)         # solve v*
                self.solve_pressure_correction_amg(a_p, b_p) # solve p'
                self.correct_velocity_pressure(a_p, b_p)# correct u,v,p
                
                # BDC
                self.apply_boundary_conditions()
                self.apply_boundary_conditions_star()
                # mass conservation check
                mass_error = np.sum(np.abs(
                    (self.u[:-1, 1:-1] - self.u[1:, 1:-1]) * self.dy +
                    (self.v[1:-1, :-1] - self.v[1:-1, 1:]) * self.dx
                ))
                mass_error /= (self.dx * self.dy)
                # convergence check
                u_res = np.max(np.abs(self.u - u_old))
                v_res = np.max(np.abs(self.v - v_old))
                
                
                if (iter+1) % prt == 0:
                    print(f"Iter {iter+1}: U_res={u_res:.2e}, V_res={v_res:.2e}, Mass_err={mass_error:.2e}, c_hat={self.chat:.2e}")
                
                if (u_res < self.tol) and (v_res < self.tol):
                    print(f"Converged at iteration {iter}")
                    break
        elif not amg:
            for iter in range(self.max_iter):
                # velocity old values
                u_old = np.copy(self.u)
                v_old = np.copy(self.v)
                
                # SIMPLE steps
                a_p = self.solve_momentum_u_star(uworder=uworder)        # solve u*
                b_p = self.solve_momentum_v_star(uworder=uworder)         # solve v*
                self.solve_pressure_correction(a_p, b_p) # solve p'
                self.correct_velocity_pressure(a_p, b_p)# correct u,v,p
                
                # BDC
                self.apply_boundary_conditions()
                self.apply_boundary_conditions_star()
                # mass conservation check
                mass_error = np.sum(np.abs(
                    (self.u[:-1, 1:-1] - self.u[1:, 1:-1]) * self.dy +
                    (self.v[1:-1, :-1] - self.v[1:-1, 1:]) * self.dx
                ))
                mass_error /= (self.dx * self.dy)
                # convergence check
                u_res = np.max(np.abs(self.u - u_old))
                v_res = np.max(np.abs(self.v - v_old))
                
                
                if (iter+1) % prt == 0:
                    print(f"Iter {iter+1}: U_res={u_res:.2e}, V_res={v_res:.2e}, Mass_err={mass_error:.2e}, c_hat={self.chat:.2e}")
                
                if (u_res < self.tol) and (v_res < self.tol):
                    print(f"Converged at iteration {iter}")
                    break

    def get_center_velocity(self):
        """velocity at cell centers"""
        # averaged u values in principle nodes
        u_center = np.zeros((self.nx, self.ny))
        u_center = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        # averaged v values in principle nodes
        v_center = np.zeros((self.nx, self.ny))
        v_center = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        with np.printoptions(precision=2, suppress=False, threshold=np.inf):
            print('the reference')
            print(self.p[int(self.nx/2),int(self.ny/2)])
        return u_center, v_center, self.p
