import numpy as np
import matplotlib.pyplot as plt
# import time, sys
import os

# Saving folder


class DiffSchemes:
    def __init__(self, name, dt, dx, x, t, sigma = None, c = None, a = None, gamma = None, ini_condi = None, bnd_condi = None, folder = None):
        self.dt = dt
        self.dx = dx
        self.sigma = sigma
        self.c = c
        self.gamma = gamma
        self.x = x
        self.t = t
        self.init_condition = ini_condi
        self.bound_condition = bnd_condi
        self.name = name
        self.left_x = x[0]
        self.right_x = x[-1]
        self.file_folder = folder

    def _plot_sigma(self, result, time, scheme):
        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)
        file_folder = self.file_folder
        x = self.x
        y = result
        if not os.path.exists(os.path.join(file_folder, f'{self.name}@{scheme}@sigma = {self.sigma}')):
            os.makedirs(os.path.join(file_folder, f'{self.name}@{scheme}@sigma = {self.sigma}'))
        file_subfolder = os.path.join(file_folder, f'{self.name}@{scheme}@sigma = {self.sigma}')
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Temperature')
        plt.title(f"Solution at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel("Temperature")
        plt.ylim(0, 3.3)
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(file_subfolder, f'Solution at {time:.3f}.png')
        plt.savefig(file_path)
        plt.close()

    def _plot_cfl(self, result, time, scheme, mesh = None, cfl = True, k_2 = None, k_4 = None):
        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)
        file_folder = self.file_folder
        x = self.x
        y = result
        file_subfolder = file_folder
        if cfl:
            file_subfolder = os.path.join(file_folder, f'{self.name}@{scheme}@CFL = {self.c}')
            if not os.path.exists(file_subfolder):
                os.makedirs(file_subfolder)
        elif mesh is not None:
            file_subfolder = os.path.join(file_folder, f'{self.name}@{scheme}@mesh = {len(self.x)}')
            if k_2 is not None:
                file_subfolder = os.path.join(file_subfolder, f'@$k_2$ = {k_2}')
            if k_4 is not None:
                file_subfolder = os.path.join(file_subfolder, f'@$k_4$ = {k_4}')
            if not os.path.exists(file_subfolder):
                os.makedirs(file_subfolder)
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Temperature')
        plt.title(f"Solution at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel("Temperature")
        plt.ylim(0, 3.3)
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(file_subfolder, f'Solution at {time:.3f}.png')
        plt.savefig(file_path)
        plt.close()

    def ftcs(self, t_plot):
        # initial and boundary
        scheme = 'FTCS'
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        bd_vec = self.bound_condition
        # compute
        for i in range(1, len(self.t)):
            rhs = matrx[:, i - 1].copy()
            rhs[0], rhs[-1] = bd_vec(i * self.dt)
            matrx[:, i - 1] = rhs
            for j in range(1, len(self.x) - 1):
                matrx[j, i] = self.sigma * matrx[j + 1, i - 1] + (1 - 2 * self.sigma)\
                              * matrx[j, i - 1] + self.sigma * matrx[j - 1, i - 1]
        # plot
        for time in t_plot:
            self._plot_sigma(matrx[:, int(time / self.dt)], time, scheme)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def btcs(self, t_plot):
        # initial and boundary
        scheme = 'BTCS'
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        bd_vec = self.bound_condition
        # define coefficient matrix
        co_matrx = np.zeros([len(self.x), len(self.x)])
        co_matrx[0, 0] = 1
        co_matrx[-1, -1] = 1
        for j in range(len(self.x) - 2):
            co_matrx[j + 1, j + 1] = (1 + 2 * self.sigma)
            co_matrx[j + 1, j] = - self.sigma
            co_matrx[j + 1, j + 2] = - self.sigma
        # compute
        for i in range(1, len(self.t)):
            rhs = matrx[:, i - 1].copy()
            rhs[0], rhs[-1] = bd_vec(i * self.dt)
            matrx[:, i] = np.linalg.solve(co_matrx, rhs)
        # plot
        for time in t_plot:
            self._plot_sigma(matrx[:, int(time / self.dt)], time, scheme)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def ctcs(self, t_plot):
        # initial and boundary
        scheme = 'CTCS'
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        bd_vec = self.bound_condition
        # get the first step
        for j in range(1, len(self.x) - 1):
            matrx[j, 1] = self.sigma * matrx[j + 1, 0] + (1 - 2 * self.sigma) * matrx[j, 0] + self.sigma * matrx[j - 1, 0]
        # compute
        for i in range(2, len(self.t)):
            for j in range(1, len(self.x) - 1):
                matrx[j, i] = matrx[j, i - 2] + 2 * self.sigma * (matrx[j + 1, i - 1] - 2 * matrx[j, i - 1] + matrx[j - 1, i - 1])
        # plot
        for time in t_plot:
            self._plot_sigma(matrx[:, int(time / self.dt)], time, scheme)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def one_upwind(self, t_plot):
        # initial and boundary
        scheme = 'One_upwind'
        c = self.c
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        # compute
        for i in range(1, len(self.t)):
            matrx[0, i] = (1 - c) * matrx[0, i - 1] + c * matrx[-1, i - 1]
            for j in range(1, len(self.x)):
                matrx[j, i] = (1 - c) * matrx[j, i - 1] + c * matrx[j - 1, i - 1]
        # plot
        for time in t_plot: self._plot_cfl(matrx[:, int(time / self.dt)], time, scheme)
        print(f'case: {self.name}, scheme: FTCS')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def euler_implicit(self, t_plot):
        # initial and boundary
        scheme = 'Euler_implicit'
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])

        # define coefficient matrix
        c = self.c
        co_matrx = np.zeros([len(self.x), len(self.x)])
        # line 1
        co_matrx[0, -1] = - c / 2
        co_matrx[0, 1] = c / 2
        co_matrx[0, 0] = 1
        # line 2
        co_matrx[-1, 0] = c / 2
        co_matrx[-1, -2] = - c /2
        co_matrx[-1, -1] = 1
        # others
        for j in range(len(self.x) - 2):
            co_matrx[j + 1, j + 1] = 1
            co_matrx[j + 1, j] = - c / 2
            co_matrx[j + 1, j + 2] = c / 2

        # compute
        for i in range(1, len(self.t)):
            rhs = matrx[:, i - 1].copy()
            matrx[:, i] = np.linalg.solve(co_matrx, rhs)
        # plot
        for time in t_plot: self._plot_cfl(matrx[:, int(time / self.dt)], time, scheme)
        print(f'case: {self.name}, scheme: FTCS')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def _1d_1vec_static(self, co_matrx, scheme: str, t_plot):
        # initial and boundary
        matrx = np.zeros(len(self.x))
        matrx = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        matrx_f = matrx.copy()
        # compute and plot
        for i in range(1, len(self.t)):
            matrx = co_matrx @ matrx_f.T
            matrx_f = matrx.copy()
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_cfl(matrx, time, scheme)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def _1d_3vec_eulerian(self, matrx_ini, F_gene: callable, scheme: str, t_plot, mesh = True, k_2 = None, k_4 = None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_half = F_gene(matrx_f)
            matrx = matrx_f - t_x * (F_half[1:] - F_half[:-1])
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_cfl(matrx, time, scheme, cfl = False, mesh = mesh, k_2 = k_2, k_4 = k_4)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return 0

    def _1d_eulerian_u_a(self, matrx_f):
        gamma = self.gamma
        def det_a_matrx(matrx_f):
            a_matrx = np.zeros(len(self.x) + 2)
            a_matrx[1: -1] = ((gamma * (gamma - 1) * (matrx_f[:, 2] - matrx_f[:, 1] ** 2 / (2 * matrx_f[:, 1])))
                        / matrx_f[:, 0]) ** 0.5
            a_matrx[0] = a_matrx[1]
            a_matrx[-1] = a_matrx[-2]
            return a_matrx
        a_matrx = det_a_matrx(matrx_f)
        # velocity
        def det_u_matrx(matrx_f):
            u_matrx = np.zeros(len(self.x) + 2)
            u_matrx[1: -1] = matrx_f[:, 1] / matrx_f[:, 0]
            u_matrx[0] = u_matrx[1]
            u_matrx[-1] = u_matrx[-2]
            return u_matrx
        u_matrx = det_u_matrx(matrx_f)
        return u_matrx, a_matrx

    def _1d_eulerian_A(self, matrx_f):
        gamma = self.gamma
        A = np.zeros([matrx_f.shape[0] + 2, 3, 3])
        A[1:-1, 0, 0] = A[:, 0, 2] = 0
        A[1:-1, 0, 1] = 1
        A[1:-1, 1, 0] = (gamma - 3) * 0.5 * (matrx_f[:, 1] ** 2 / matrx_f[:, 0] ** 2)
        A[1:-1, 1, 1] = (3 - gamma) * matrx_f[:, 1] / matrx_f[:, 0]
        A[1:-1, 1, 2] = gamma - 1
        A[1:-1, 2, 0] = (gamma - 1) * (matrx_f[:, 1] / matrx_f[:, 0]) ** 2 - gamma * matrx_f[:, 1] * matrx_f[:, 2] / matrx_f[:,
                                                                                                          0] ** 2
        A[1:-1, 2, 1] = -1.5 * (gamma - 1) * (matrx_f[:, 1] / matrx_f[:, 0]) ** 2 + gamma * matrx_f[:, 2] / matrx_f[:, 0]
        A[1:-1, 2, 2] = gamma * matrx_f[:, 1] / matrx_f[:, 0]
        A[0] = A[1]
        A[-1] = A[-2]
        return A

    def lax_wendroff(self, t_plot):
        # scheme parameters
        scheme = 'Lax-Wendroff'
        c = self.c
        # define coefficient matrix
        co_matrx = np.zeros([len(self.x), len(self.x)])
        # line 1
        co_matrx[0, -1] = 0.5 * (c ** 2 + c)
        co_matrx[0, 0] = 1 - c ** 2
        co_matrx[0, 1] = 0.5 * (c ** 2 - c)
        # line 2
        co_matrx[-1, 0] = 0.5 * (c ** 2 - c)
        co_matrx[-1, -1] = 1 - c ** 2
        co_matrx[-1, -2] = 0.5 * (c ** 2 + c)
        # others
        for i in range(1, len(self.x) - 1):
            co_matrx[i, i] = 1 - c ** 2
            co_matrx[i, i - 1] = 0.5 * (c ** 2 + c)
            co_matrx[i, i + 1] = 0.5 * (c ** 2 - c)
        # compute and plot
        self._1d_1vec_static(co_matrx, scheme, t_plot)
        return co_matrx

    def warming_beam(self, t_plot):
        # scheme parameters
        scheme = 'Warming-Beam'
        c = self.c
        # define coefficient matrix
        co_matrx = np.zeros([len(self.x), len(self.x)])
        # line 1
        co_matrx[0, -1] = 2 * c - c ** 2
        co_matrx[0, 0] = 1 - 1.5 * c + 0.5 * c ** 2
        co_matrx[0, -2] = - 0.5 * (c - c ** 2)
        # line 2
        co_matrx[1, 0] = 2 * c - c ** 2
        co_matrx[1, 1] = 1 - 1.5 * c + 0.5 * c ** 2
        co_matrx[1, -1] = - 0.5 * (c - c ** 2)
        # others
        for i in range(2, len(self.x)):
            co_matrx[i, i] = 1 - 1.5 * c + 0.5 * c ** 2
            co_matrx[i, i - 1] = 2 * c - c ** 2
            co_matrx[i, i - 2] = - 0.5 * (c - c ** 2)
        # compute and plot
        self._1d_1vec_static(co_matrx, scheme, t_plot)
        return co_matrx

    def obtained_in_work_5(self, t_plot):
        # scheme parameters
        scheme = 'Obtained'
        c = self.c
        # define coefficient matrix
        co_matrx = np.zeros([len(self.x), len(self.x)])
        for i in range(0, len(self.x)):
            s1 = (i + 1) if i <= (len(self.x) - 2) else (len(self.x) - i - 1)
            co_matrx[i, i] = (2 - c - 2 * c ** 2 + c ** 3) / 2
            co_matrx[i, i - 1] = c * (2 + c - c ** 2) / 2
            co_matrx[i, i - 2] = (c * (c ** 2 - 1)) / 6
            co_matrx[i, s1] = -c * (2 - 3 * c + c ** 2) / 6
        # compute and plot
        self._1d_1vec_static(co_matrx, scheme, t_plot)
        return 0

    def rusanov(self, t_plot):
        # initial and boundary
        scheme = 'Rusanov'
        gamma = self.gamma
        matrx = np.zeros([len(self.x), 3])
        rho_u_p = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        matrx[:, 0] = rho_u_p[:, 0]
        matrx[:, 1] = rho_u_p[:, 0] * rho_u_p[:, 1]
        matrx[:, 2] = (rho_u_p[:, 2] / (gamma - 1)) + 0.5 * rho_u_p[:, 0] * rho_u_p[:, 1] ** 2
        matrx_f = matrx.copy()

        # Define Jacobian A ${\part F\over\part U}$
        A = self._1d_eulerian_A(matrx_f)

        # Flux generator
        def F_gene(matrx_f):
            # u and a (speed of sound) array
            u_matrx, a_matrx = self._1d_eulerian_u_a(matrx_f)
            # The basic flux
            F_matrx = np.zeros([len(self.x) + 2, 3])
            print(f'A shape: {A.shape}')
            print(f'u_matrx shape: {u_matrx.shape}')
            F_matrx[1:-1] = np.einsum('ijk, ik-> ij', A, u_matrx)
            F_matrx[0] = F_matrx[1]
            F_matrx[-1] = F_matrx[-2]
            # basic half-node flux
            F_half = 0.5 * (F_matrx[:-1] + F_matrx[1:])
            # artificial viscosity added
            u_abs = np.abs(u_matrx)
            a_abs = np.abs(a_matrx)
            vis_matrx = -0.25 * (u_abs[:-1] + a_abs[:-1] + u_abs[1:] + a_abs[1:]) * (u_matrx[1:] - u_matrx[:-1])
            # final half-node flux
            F_half += vis_matrx
            return F_half

        # compute and plot
        self._1d_3vec_eulerian(matrx, F_gene, scheme, t_plot)
        return 0

    def jameson(self, t_plot, k_2 = 0.55, k_4 = 1 / 100):
        # initial and boundary
        scheme = 'Jameson'
        gamma = self.gamma
        matrx = np.zeros([len(self.x), 3])
        rho_u_p = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        matrx[:, 0] = rho_u_p[:, 0]
        matrx[:, 1] = rho_u_p[:, 0] * rho_u_p[:, 1]
        matrx[:, 2] = (rho_u_p[:, 2] / (gamma - 1)) + 0.5 * rho_u_p[:, 0] * rho_u_p[:, 1] ** 2
        matrx_f = matrx.copy()
        matrx_ini = matrx.copy()

        # Define Jacobian A ${\part F\over\part U}$
        A = self._1d_eulerian_A(matrx_f)

        # Flux generator
        def F_gene(matrx_f):
            # u (velocity) and a (speed of sound) array
            u_matrx, a_matrx = self._1d_eulerian_u_a(matrx_f)
            # The basic flux
            F_matrx = np.zeros([len(self.x) + 2, 3])
            F_matrx[1:-1] = np.einsum('ijk, ik-> ij', A, u_matrx)
            F_matrx[0] = F_matrx[1]
            F_matrx[-1] = F_matrx[-2]
            # basic half-node flux
            F_half = 0.5 * (F_matrx[:-1] + F_matrx[1:])
            # artificial viscosity added
            u_abs = np.abs(u_matrx)
            a_abs = np.abs(a_matrx)
            p_matrx = np.zeros(len(self.x) + 5)
            p_matrx[3:-3] = (gamma - 1) * (matrx_f[:, 3] - 0.5 * matrx_f[:, 1] ** 2 / matrx_f[:, 1])
            p_matrx[0] = p_matrx[1] = p_matrx[2]
            p_matrx[-1] = p_matrx[-2] = p_matrx[-3] = p_matrx[-4]
            # viscous parameter $\varepsilon_2$
            nu_matrx = np.abs(p_matrx[2:] - 2 * p_matrx[1:-1] + p_matrx[:-2]) / np.abs(p_matrx[2:] + 2 * p_matrx[1:-1] + p_matrx[:-2])
            windows = np.lib.stride_tricks.sliding_window_view(nu_matrx, window_shape=4)
            e2_matrx = k_2 * np.max(windows, axis=1)
            e4_matrx = np.maximum(0, k_4 - e2_matrx)
            # expanded matrx
            matrx_expand = np.zeros(len(self.x) + 3)
            matrx_expand[1:-2] = matrx_f
            matrx_expand[0] = matrx_expand[1]
            matrx_expand[-1] = matrx_expand[-2] = matrx_expand[-3]
            lambda_max = 0.5 * (u_abs[:-1] + a_abs[:-1] + u_abs[1:] + a_abs[1:])
            vis_matrx_2 = -lambda_max * e2_matrx * (matrx_expand[2:-1] - matrx_expand[1: -2])
            vis_matrx_4 = lambda_max * e4_matrx * (matrx_expand[3:] - 3 * matrx_expand[2:-1] + 3 * matrx_expand[1:-2] - matrx_expand[:-3])
            # final half-node flux
            F_half += (vis_matrx_2 + vis_matrx_4)
            return F_half

        # compute and plot
        self._1d_3vec_eulerian(matrx, F_gene, scheme, t_plot, k_2 = k_2, k_4 = k_4)
        return 0
