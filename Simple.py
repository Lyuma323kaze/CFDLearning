from Diff_schme import DiffSchemes
import numpy as np

class CavitySIMPLE(DiffSchemes):
    def __init__(self, name, dt, dx, x, t, dy, y, Re, U_top, 
                 max_iter=1000,
                 tol=1e-5,
                 alpha_u=1,
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
        self.alpha_p = alpha_p  # pressure lack relaxation
        
        # mesh
        # pressure (prime)
        self.p = np.zeros((self.nx, self.ny))
        
        # u mesh
        self.u = 0 * np.ones((self.nx+1, self.ny+2))
        # self.u[:, 1:-1] = 0.1 * (U_top * (2 * self.y[np.newaxis, :] - 1))
        # print(self.u)
        
        # v mesh
        self.v = np.zeros((self.nx+2, self.ny+1))
        # self.v[1:-1, :] = 0.1 * U_top * (1 - 2 * self.x[:, np.newaxis])
        # print(self.v)
        
        # modification variables
        self.u_star = np.copy(self.u)
        self.v_star = np.copy(self.v)
        self.p_prime = np.zeros((self.nx, self.ny))
        
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
        self.u[:, 0] = -self.u[:, 1]   # u=0 by virtual node
        # self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0   # v=0
        
        # left side fixed (no slip)
        self.u[0, :] = 0.0   # u=0
        self.v[0, :] = -self.v[1, :]   # v=0 by virtual node
        # self.v[0, :] = 0.0   # v=0 by virtual node
        
        # right side fixed (no slip)
        self.u[-1, :] = 0.0  # u=0
        self.v[-1, :] = -self.v[-2, :]  # v=0 by virtual node
        # self.v[-1, :] = 0.0  # v=0 by virtual node
        
        # Upper lid moving (u velocity)
        self.u[:, -1] = 2 * self.U_top - self.u[:, -2]  # upper x cover by virtual node
        # self.u[:, -1] = self.U_top  # upper x cover by virtual node
        self.v[:, -1] = 0.0         # v velocity on the top

    def apply_boundary_conditions_star(self):
        """Set boundary values for prediction values"""
        # bottom fixed (no slip)
        self.u_star[:, 0] = -self.u_star[:, 1]   # u=0 by virtual node
        # self.u_star[:, 0] = 0.0
        self.v_star[:, 0] = 0.0   # v=0
        
        # left side fixed (no slip)
        self.u_star[0, :] = 0.0   # u=0
        self.v_star[0, :] = -self.v_star[1, :]   # v=0 by virtual node
        # self.v_star[0, :] = 0.0
        
        # right side fixed (no slip)
        self.u_star[-1, :] = 0.0  # u=0
        self.v_star[-1, :] = -self.v_star[-2, :]  # v=0 by virtual node
        # self.v_star[-1, :] = 0.0
        
        # upper lid moving (u velocity)
        self.u_star[:, -1] = 2 * self.U_top - self.u_star[:, -2]  # upper x cover by virtual node
        # self.u_star[:, -1] = self.U_top  # upper x cover by virtual node
        self.v_star[:, -1] = 0.0         # v velocity on the top

    def solve_momentum_u_star(self, uworder=1, iter_u=200):
        """solve u-momentum equation"""
        self.u_star = np.copy(self.u)  # initialize u_star
        
        # upwind coefficients
        u_avr_x = np.empty((self.nx+1, self.ny+2))
        u_avr_x[:-1,:] = (self.u[:-1,:] + self.u[1:,:]) / 2 
        u_avr_x[-1,:] = 0.5 * self.u[-1,:]  # last row is the right boundary
        
        alpha_uxp = np.maximum(u_avr_x, 0)[:,1:-1]     # nx+1, ny
        alpha_uxm = np.minimum(u_avr_x, 0)[:,1:-1]     # nx+1, ny
        # print(alpha_uxp.shape, alpha_uxm.shape)
        
        v_avr_x = (self.v[1:-1,:] + self.v[2:,:]) / 2
        
        alpha_uyp = np.maximum(v_avr_x, 0)     # nx, ny+1
        alpha_uym = np.minimum(v_avr_x, 0)     # nx, ny+1
        # print(alpha_uyp.shape, alpha_uym.shape)
        gamma_ux = np.zeros_like(alpha_uxp)  # nx+1, ny
        gamma_uy = np.zeros_like(alpha_uyp)  # nx, ny+1
        if uworder == 2:
            # TODO: fix the expression
            gamma_ux[1:-1] = 0.5 * (alpha_uxp[1:-1] * (self.u[1:-2,1:-1] - self.u[:-3,1:-1]) +
                                    alpha_uxm * (self.u[2:-1,1:-1] - self.u[3:,1:-1]))
            gamma_ux[0] = 0.5 * alpha_uxm[0] * (self.u[1,1:-1] - self.u[2,1:-1])
            gamma_ux[-1] = 0.5 * alpha_uxp[-1] * (self.u[-2,1:-1] - self.u[-3,1:-1])
            
            gamma_uy[:,1:-2] = 0.5 * (alpha_uyp[:,1:-2] * (self.u[1:,1:-2] - self.u[1:,:-3]) +
                                    alpha_uym * (self.u[1:,2:-1] - self.u[1:,3:]))
            gamma_uy[:,0] = 0.5 * alpha_uym[:,0] * (self.u[1:,1] - self.u[1:,2])
            gamma_uy[:,-1] = 0.5 * alpha_uyp[:,-1] * (self.u[1:,-1] - self.u[1:,-2])
            gamma_uy[:,-2] = 0.5 * alpha_uyp[:,-2] * (self.u[1:,-2] - self.u[1:,-3])
            
        # discretization coefficients (nx-1,ny for n,s; nx,ny for e,w, nx-1,ny for p,hat)
        a_p = (self.dx * self.dy / self.dt) +\
                self.dy * (alpha_uxp[1:-1] - alpha_uxm[:-2] + (2 / (self.Re * self.dx))) +\
                self.dx * (alpha_uyp[:-1,1:] - alpha_uym[:-1,:-1] + (2 / (self.Re * self.dy)))
        a_w = self.dy * (alpha_uxp[:-1] + 1 / (self.Re * self.dx))
        a_e = self.dy * (-alpha_uxm[1:] + 1 / (self.Re * self.dx))
        a_s = self.dx * (alpha_uyp[:-1,:-1] + 1 / (self.Re * self.dy))
        a_n = self.dx * (-alpha_uym[:-1,1:] + 1 / (self.Re * self.dy))
        a_hat = self.dy * (gamma_ux[1:-1] - gamma_ux[:-2]) +\
                self.dx * (gamma_uy[:-1,1:] - gamma_uy[:-1,:-1])
        if np.min(a_n / a_p) < 0:
            print(np.min(a_n / a_p))
            raise ValueError('positive coefficient rule was ruined')
        # pressure gradient(nx-1,ny)
        dP = -(self.p[1:] - self.p[:-1]) * self.dy
            
        for _ in range(iter_u):
            # update
            value_old = np.copy(self.u_star[1:-1,1:-1])
            self.u_star[1:-1,1:-1] =(self.alpha_u * (
                                    a_e[:-1] * self.u_star[2:,1:-1] + 
                                    a_w[:-1] * self.u_star[:-2,1:-1] +
                                    a_n * self.u_star[1:-1,2:] +
                                    a_s * self.u_star[1:-1,:-2] +
                                    dP + a_hat
                                    ) + (1 - self.alpha_u) * self.dx * self.dy / self.dt * value_old
                                ) / (a_p + 1e-8)
            self.apply_boundary_conditions_star()  # ensure boundary conditions are applied
            diff = np.max(np.abs(self.u_star[1:-1,1:-1] - value_old))
            if diff < self.tol:
                return a_e, a_w
        # nx,ny
        return a_e, a_w

    def solve_momentum_v_star(self, uworder=1, iter_v=200):
        """solve momentum equation"""
        self.v_star = np.copy(self.v)  # initialize u_star
        # upwind coefficients
        v_avr_y = np.empty((self.nx+2, self.ny+1))
        v_avr_y[:,:-1] = (self.v[:,:-1] + self.v[:,1:]) / 2 
        v_avr_y[:,-1] = 0.5 * self.v[:,-1]  # last column is the right boundary
        
        alpha_vyp = np.maximum(v_avr_y, 0)[1:-1,:]     # nx, ny+1
        alpha_vym = np.minimum(v_avr_y, 0)[1:-1,:]     # nx, ny+1
        
        u_avr_y = (self.u[:,1:-1] + self.u[:,2:]) / 2
        
        alpha_vxp = np.maximum(u_avr_y, 0)     # nx+1, ny
        alpha_vxm = np.minimum(u_avr_y, 0)     # nx+1, ny
        gamma_vy = np.zeros_like(alpha_vyp)  # nx, ny+1
        gamma_vx = np.zeros_like(alpha_vxp)  # nx+1, ny
        if uworder == 2:
            gamma_vy[:,1:-1] = 0.5 * (alpha_vyp[:,1:-1] * (self.v[1:-1,1:-2] - self.v[1:-1,:-3]) +
                                    alpha_vym * (self.v[1:-1,2:-1] - self.v[1:-1,3]))
            gamma_vy[:,0] = 0.5 * alpha_vym[:,0] * (self.v[1:-1,1] - self.v[1:-1,2])
            gamma_vy[:,-1] = 0.5 * alpha_vyp[:,-1] * (self.v[1:-1,-2] - self.v[-1:-1,-3])
            
            gamma_vx[1:-2] = 0.5 * (alpha_vxp[1:-2,:] * (self.v[1:-2,1:] - self.v[:-3,1:]) +
                                    alpha_vxm * (self.v[2:-1,1:] - self.v[3:,1:]))
            gamma_vx[0] = 0.5 * alpha_vxm[0] * (self.v[1:,1] - self.v[2:,1])
            gamma_vx[-1] = 0.5 * alpha_vxp[-1] * (self.v[-1,1:] - self.v[-2,1:])
            gamma_vx[-2] = 0.5 * alpha_vxp[-2] * (self.v[-2,1:] - self.v[-3,1:])
            
        # discretization coefficients (nx,ny-1 for w,e; nx,ny for n,s, nx,ny-1 for p,hat)
        a_p = (self.dy * self.dx / self.dt) +\
                self.dx * (alpha_vyp[:,1:-1] - alpha_vym[:,:-2] + (2 / (self.Re * self.dy))) +\
                self.dy * (alpha_vxp[1:,:-1] - alpha_vxm[:-1,:-1] + (2 / (self.Re * self.dx)))
        a_s = self.dx * (alpha_vyp[:,:-1] + 1 / (self.Re * self.dy))
        a_n = self.dx * (-alpha_vym[:,1:] + 1 / (self.Re * self.dy))
        a_w = self.dy * (alpha_vxp[:-1,:-1] + 1 / (self.Re * self.dx))
        a_e = self.dy * (-alpha_vxm[1:,:-1] + 1 / (self.Re * self.dx))
        a_hat = self.dx * (gamma_vy[:,1:-1] - gamma_vy[:,:-2]) +\
                self.dy * (gamma_vx[1:,:-1] - gamma_vx[:-1,:-1])
        
        # pressure gradient(nx,ny-1)
        dP = -(self.p[:,1:] - self.p[:,:-1]) * self.dx
            
        for _ in range(iter_v):
            # update
            value_old = np.copy(self.v_star[1:-1,1:-1])
            self.v_star[1:-1,1:-1] = (self.alpha_u * (a_e * self.v_star[1:-1,2:] + 
                                     a_w * self.v_star[1:-1,:-2] +
                                     a_n[:,:-1] * self.v_star[2:,1:-1] +
                                     a_s[:,:-1] * self.v_star[:-2,1:-1] +
                                     dP + a_hat
                                     ) + (1 - self.alpha_u) * self.dy * self.dx / self.dt * value_old
                                    ) / (a_p + 1e-8)
            self.apply_boundary_conditions_star()  # ensure boundary conditions are applied
            diff = np.max(np.abs(self.v_star[1:-1,1:-1] - value_old))
            if diff < self.tol:
                return a_n, a_s
        # nx,ny
        return a_n, a_s
    
    def solve_pressure_correction(self, a_e, a_w, b_n, b_s, iter_p=500):
        """solve pressure correction equation"""
        def get_transitioned():
            p_virtual_r = np.concatenate((self.p_prime[:, 1:], self.p_prime[:, -1][:,np.newaxis]), axis=1)
            p_virtual_l = np.concatenate((self.p_prime[:, 0][:,np.newaxis], self.p_prime[:, :-1]), axis=1)
            p_virtual_d = np.concatenate((self.p_prime[0, :][np.newaxis,:], self.p_prime[:-1, :]), axis=0)
            p_virtual_u = np.concatenate((self.p_prime[1:, :], self.p_prime[-1, :][np.newaxis,:]), axis=0)
            return p_virtual_u, p_virtual_d, p_virtual_l, p_virtual_r
        # w,e,u,d with BDC (nx,ny)
        p_virtual_u, p_virtual_d, p_virtual_l, p_virtual_r = get_transitioned()
        # p_u = self.p_prime[:, 1:]
        # p_d = self.p_prime[:, :-1]
        # p_l = self.p_prime[:-1]
        # p_r = self.p_prime[1:]
        # coefficients (nx,ny)
        c_e = self.dy ** 2 / a_e
        c_w = self.dy ** 2 / a_w
        c_n = self.dx ** 2 / b_n
        c_s = self.dx ** 2 / b_s
        c_p = c_e + c_w + c_n + c_s
        # c_p[0] -= c_w[0]
        # c_p[-1] -= c_e[-1]
        # c_p[:, -1] -= c_n[:, -1]
        # c_p[:, 0] -= c_s[:, 0]
        c_hat = -(
            self.dy * (self.u_star[1:,1:-1] - self.u_star[:-1,1:-1]) +
            self.dx * (self.v_star[1:-1,1:] - self.v_star[1:-1,:-1])
        )
        self.chat = np.max(np.abs(c_hat))
        for _ in range(iter_p):
            value_old = np.copy(self.p_prime)
            
            # jacobian p_prime update with [0,0] the reference
            # self.p_prime[1:,1:] = (1/(c_p[1:,1:] + 1e-8)) * (
            #     c_e[1:,1:] * p_virtual_r[1:,1:] +
            #     c_w[1:,1:] * p_virtual_l[1:,1:] +
            #     c_n[1:,1:] * p_virtual_u[1:,1:] +
            #     c_s[1:,1:] * p_virtual_d[1:,1:] +
            #     c_hat[1:,1:]
            # )
            # self.p_prime[0,2:] = (1/(c_p[0,2:] + 1e-8)) * (
            #     c_e[0,2:] * p_virtual_r[0,2:] +
            #     c_w[0,2:] * p_virtual_l[0,2:] +
            #     c_n[0,2:] * p_virtual_u[0,2:] +
            #     c_s[0,2:] * p_virtual_d[0,2:] +
            #     c_hat[0,2:]
            # )
            # self.p_prime[2:,0] = (1/(c_p[0,2:] + 1e-8)) * (
            #     c_e[2:,0] * p_virtual_r[2:,0] +
            #     c_w[2:,0] * p_virtual_l[2:,0] +
            #     c_n[2:,0] * p_virtual_u[2:,0] +
            #     c_s[2:,0] * p_virtual_d[2:,0] +
            #     c_hat[2:,0]
            # )
            # self.p_prime[0,1] = (1/((c_e+c_n)[0,1] + 1e-8)) * (
            #     c_e[0,1] * p_virtual_r[0,1] +
            #     c_n[0,1] * p_virtual_u[0,1] +
            #     c_hat[0,1]
            # )
            # self.p_prime[1,0] = (1/((c_e+c_n)[1,0] + 1e-8)) * (
            #     c_e[1,0] * p_virtual_r[1,0] +
            #     c_n[1,0] * p_virtual_u[1,0] +
            #     c_hat[1,0]
            # )
            
            
            # jacobian p_prime update with [100,100] the reference
            self.p_prime[:,101:] = (1/(c_p[:,101:] + 1e-8)) * (
                c_e[:,101:] * p_virtual_r[:,101:] +
                c_w[:,101:] * p_virtual_l[:,101:] +
                c_n[:,101:] * p_virtual_u[:,101:] +
                c_s[:,101:] * p_virtual_d[:,101:] +
                c_hat[:,101:]
            )
            self.p_prime[:,:100] = (1/(c_p[:,:100] + 1e-8)) * (
                c_e[:,:100] * p_virtual_r[:,:100] +
                c_w[:,:100] * p_virtual_l[:,:100] +
                c_n[:,:100] * p_virtual_u[:,:100] +
                c_s[:,:100] * p_virtual_d[:,:100] +
                c_hat[:,:100]
            )
            self.p_prime[:100,100] = (1/(c_p[:100,100] + 1e-8)) * (
                c_e[:100,100] * p_virtual_r[:100,100] +
                c_w[:100,100] * p_virtual_l[:100,100] +
                c_n[:100,100] * p_virtual_u[:100,100] +
                c_s[:100,100] * p_virtual_d[:100,100] +
                c_hat[:100,100]
            )
            self.p_prime[101:,100] = (1/(c_p[101:,100] + 1e-8)) * (
                c_e[101:,100] * p_virtual_r[101:,100] +
                c_w[101:,100] * p_virtual_l[101:,100] +
                c_n[101:,100] * p_virtual_u[101:,100] +
                c_s[101:,100] * p_virtual_d[101:,100] +
                c_hat[101:,100]
            )
            self.p_prime[99,100] = (1/((c_w+c_n+c_s)[99,100] + 1e-8)) * (
                c_w[99,100] * p_virtual_l[99,100] +
                c_n[99,100] * p_virtual_u[99,100] +
                c_s[99,100] * p_virtual_d[99,100] +
                c_hat[99,100]
            )
            self.p_prime[101,100] = (1/((c_e+c_n+c_s)[101,100] + 1e-8)) * (
                c_e[101,100] * p_virtual_r[101,100] +
                c_n[101,100] * p_virtual_u[101,100] +
                c_s[101,100] * p_virtual_d[101,100] +
                c_hat[101,100]
            )
            self.p_prime[100,101] = (1/((c_e+c_n+c_w)[100,101] + 1e-8)) * (
                c_e[100,101] * p_virtual_r[100,101] +
                c_n[100,101] * p_virtual_u[100,101] +
                c_w[100,101] * p_virtual_l[100,101] +
                c_hat[100,101]
            )
            self.p_prime[100,101] = (1/((c_e+c_n+c_w)[100,101] + 1e-8)) * (
                c_e[100,101] * p_virtual_r[100,101] +
                c_n[100,101] * p_virtual_u[100,101] +
                c_w[100,101] * p_virtual_l[100,101] +
                c_hat[100,101]
            )
            
            # linear G-S iteration
            # TODO: ...
            
            # simple jacobian update (same as )
            # self.p_prime = (1/(c_p + 1e-8)) * (
            #     c_e * p_virtual_r +
            #     c_w * p_virtual_l +
            #     c_n * p_virtual_u +
            #     c_s * p_virtual_d +
            #     c_hat
            # )

            # w,e,u,d with BDC (nx,ny)
            p_virtual_u, p_virtual_d, p_virtual_l, p_virtual_r = get_transitioned()
            # check inner convergence
            res = np.sum(np.abs(self.p_prime - value_old))
            self.res = res
            if res < 1e-7:
                break
        return
    
    def correct_velocity_pressure(self, a_e, b_n):
        """modify velocity and pressure based on pressure correction"""
        # pressure correction with relaxation
        self.p += self.alpha_p * self.p_prime
        # modify u with relaxation
        # print(self.u_star.shape, self.p_prime.shape, a_e.shape)
        self.u[1:-1,1:-1] = (self.u_star[1:-1,1:-1] + self.dy *\
            (self.p_prime[:-1] - self.p_prime[1:]) / a_e[:-1])
        # modify v with relaxation
        # print(self.v_star.shape, self.v.shape, a_e.shape, b_n.shape)
        self.v[1:-1,1:-1] = (self.v_star[1:-1,1:-1] + self.dx *\
            (self.p_prime[:,:-1] - self.p_prime[:,1:]) / b_n[:,:-1])
        self.apply_boundary_conditions()  # apply BDC
        
    def solve(self, uworder=1):
        """SIMPLE main loop"""
        # BDC
        self.apply_boundary_conditions()
        self.apply_boundary_conditions_star()
        for iter in range(self.max_iter):
            # velocity old values
            u_old = np.copy(self.u)
            v_old = np.copy(self.v)
    
            # SIMPLE steps
            a_e, a_w = self.solve_momentum_u_star(uworder=uworder)        # solve u*
            b_n, b_s = self.solve_momentum_v_star(uworder=uworder)         # solve v*
            self.solve_pressure_correction(a_e, a_w, b_n, b_s) # solve p'
            self.correct_velocity_pressure(a_e, b_n)# correct u,v,p
            
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
            
            
            if iter % 200 == 0:
                print(f"Iter {iter}: U_res={u_res:.2e}, V_res={v_res:.2e}, Mass_err={mass_error:.2e}, c_hat={self.chat:.2e}")
            
            if u_res < self.tol and v_res < self.tol and mass_error < 1e-4:
                print(f"Converged at iteration {iter}")
                break

    def get_center_velocity(self):
        """velocity at cell centers"""
        # u在x方向中心，y方向需要平均
        u_center = np.zeros((self.nx, self.ny))
        u_center = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        # v在y方向中心，x方向需要平均
        v_center = np.zeros((self.nx, self.ny))
        v_center = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        print('self.u')
        print(self.u)
        print('self.p')
        print(self.p)
        return u_center, v_center, self.p
