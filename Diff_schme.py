import numpy as np
import matplotlib.pyplot as plt
# import time, sys
import os

from botocore.httpsession import mask_proxy_url


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
        self.a = self.c * self.dx / self.dt
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

    def _plot_cfl(self, result, time, scheme,
                  mesh = None, cfl = True, k_2 = None, k_4 = None, ylim = None):
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
                file_subfolder = os.path.join(file_subfolder, f'@k_2 = {k_2}@k_4 = {k_4}')
            if not os.path.exists(file_subfolder):
                os.makedirs(file_subfolder)
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Temperature')
        plt.title(f"Solution at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel("Temperature")
        if ylim is not None:
            plt.ylim(ylim)
        else:
            plt.ylim(0, 3.3)
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(file_subfolder, f'Solution at {time:.3f}.png')
        plt.savefig(file_path)
        plt.close()

    def _plot_1d_3vec(self, result, time, scheme,
                      mesh = None, cfl = True, k_2 = None, k_4 = None, entropy_fix = None):
        gamma = self.gamma
        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)
        file_folder = self.file_folder
        x = self.x
        y = result
        rho_result = y[:, 0]
        u_result = y[:, 1] / y[:, 0]
        p_result = (gamma - 1) * (y[:, 2] - 0.5 * y[:, 1] ** 2 / y[:, 0])
        file_subfolder = file_folder
        if cfl:
            file_subfolder = os.path.join(file_folder, f'{self.name}@{scheme}@CFL = {self.c}')
            if not os.path.exists(file_subfolder):
                os.makedirs(file_subfolder)
        elif mesh is not None:
            file_subfolder = os.path.join(file_folder, f'{self.name}@{scheme}@mesh = {len(self.x)}')
            if k_2 is not None:
                file_subfolder = os.path.join(file_subfolder, f'@k_2 = {k_2}@k_4 = {k_4}')
            elif entropy_fix is not None:
                file_subfolder = os.path.join(file_subfolder, f'@entropy_fix = {entropy_fix}')
        if not os.path.exists(file_subfolder):
            os.makedirs(file_subfolder)

        rho_subfolder = os.path.join(file_subfolder, 'rho')
        u_subfolder = os.path.join(file_subfolder, 'u')
        p_subfolder = os.path.join(file_subfolder, 'p')
        if not os.path.exists(rho_subfolder):
            os.makedirs(rho_subfolder)
        if not os.path.exists(u_subfolder):
            os.makedirs(u_subfolder)
        if not os.path.exists(p_subfolder):
            os.makedirs(p_subfolder)

        plt.figure(figsize=(8, 6))
        plt.plot(x, rho_result, marker='o', linestyle='-', color='b', label=r'$\rho$')
        plt.title(fr"Solution of $\rho$ at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel(r"Density $\rho$")
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(rho_subfolder, f'Solution@rho at {time:.3f}.png')
        plt.savefig(file_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(x, u_result, marker='o', linestyle='-', color='r', label='Velocity')
        plt.title(fr"Solution of $u$ at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel(r"Velocity $u$")
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(u_subfolder, f'Solution@u at {time:.3f}.png')
        plt.savefig(file_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(x, p_result, marker='o', linestyle='-', color='g', label='Pressure')
        plt.title(fr"Solution of $p$ at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel(r"Velocity $p$")
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(p_subfolder, f'Solution@p at {time:.3f}.png')
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

    def _1d_1vec_conserv_rk4(self,
                             matrx_ini,
                             F_gene: callable,
                             scheme: str,
                             t_plot,
                             mesh=True,
                             k_2=None,
                             k_4=None,
                             ylim=None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx

        # discrete \part f\over\part x
        def F_part_gene(matrx_gene):
            F_half = F_gene(matrx_gene)
            F_part = F_half[1:] - F_half[:-1]
            return F_part

        self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot,
                          mesh = mesh, k_2 = k_2, k_4 = k_4, ylim = ylim)
        return 0

    def _1d_1vec_rk4(self,
                     matrx_ini,
                     F_part_gene: callable,
                     scheme: str,
                     t_plot,
                     mesh=True,
                     k_2=None,
                     k_4=None,
                     ylim=None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_part = F_part_gene(matrx_f)  # -1 to l-1
            matrx_1 = matrx_f - 0.25 * t_x * F_part
            F_part_1 = F_part_gene(matrx_1)
            matrx_2 = matrx_f - t_x * F_part_1 / 3
            F_part_2 = F_part_gene(matrx_2)
            matrx_3 = matrx_f - 0.5 * t_x * F_part_2
            F_part_3 = F_part_gene(matrx_3)
            matrx = matrx_f - t_x * F_part_3
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_cfl(matrx, time, scheme,
                                   cfl=False, mesh=mesh, k_2=k_2, k_4=k_4, ylim=ylim)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        print('\n')
        return 0


    def _1d_3vec_eulerian_explicit(self,
                                   matrx_ini,
                                   F_gene: callable,
                                   scheme: str,
                                   t_plot,
                                   mesh = True,
                                   k_2 = None,
                                   k_4 = None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_half = F_gene(matrx_f)
            matrx = matrx_f - t_x * (F_half[1:] - F_half[:-1])
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_1d_3vec(matrx, time, scheme, cfl = False, mesh = mesh, k_2 = k_2, k_4 = k_4)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return 0

    def _1d_3vec_eulerian_rk4(self,
                              matrx_ini,
                              F_gene: callable,
                              scheme: str,
                              t_plot,
                              mesh = True,
                              k_2 = None,
                              k_4 = None,
                              entropy_fix = None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_half = F_gene(matrx_f)
            matrx_1 = matrx_f - 0.25 * t_x * (F_half[1:] - F_half[:-1])
            F_half_1 = F_gene(matrx_1)
            matrx_2 = matrx_f - t_x * (F_half_1[1:] - F_half_1[:-1]) / 3
            F_half_2 = F_gene(matrx_2)
            matrx_3 = matrx_f - 0.5 * t_x * (F_half_2[1:] - F_half_2[:-1])
            F_half_3 = F_gene(matrx_3)
            matrx = matrx_f - t_x * (F_half_3[1:] - F_half_3[:-1])
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_1d_3vec(matrx, time, scheme,
                                       cfl=False, mesh=mesh, k_2=k_2, k_4=k_4, entropy_fix = entropy_fix)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return 0

    def _1d_eulerian_u_a(self, matrx_f):
        gamma = self.gamma
        # speed of sound
        def det_a_matrx(matrx_f):
            a_matrx = np.zeros(len(self.x) + 2) # -1 to l
            a_matrx[1: -1] = (((gamma * (gamma - 1) * (matrx_f[:, 2] - 0.5 * matrx_f[:, 1] ** 2 / matrx_f[:, 0]))
                        / matrx_f[:, 0]) ** 0.5)
            a_matrx[0] = a_matrx[1]
            a_matrx[-1] = a_matrx[-2]
            a_matrx = a_matrx[:, np.newaxis]
            return a_matrx
        a_matrx = det_a_matrx(matrx_f)
        # velocity
        def det_u_matrx(matrx_f):
            u_matrx = np.zeros(len(self.x) + 2) # -1 to l
            u_matrx[1: -1] = matrx_f[:, 1] / matrx_f[:, 0]
            u_matrx[0] = u_matrx[1]
            u_matrx[-1] = u_matrx[-2]
            u_matrx = u_matrx[:, np.newaxis]
            return u_matrx
        u_matrx = det_u_matrx(matrx_f)
        return u_matrx, a_matrx

    def _1d_eulerian_A(self, matrx_f):
        gamma = self.gamma
        A = np.zeros([matrx_f.shape[0], 3, 3])
        A[:, 0, 0] = A[:, 0, 2] = 0
        A[:, 0, 1] = 1
        A[:, 1, 0] = (gamma - 3) * 0.5 * (matrx_f[:, 1] ** 2 / matrx_f[:, 0] ** 2)
        A[:, 1, 1] = (3 - gamma) * matrx_f[:, 1] / matrx_f[:, 0]
        A[:, 1, 2] = gamma - 1
        A[:, 2, 0] = (gamma - 1) * (matrx_f[:, 1] / matrx_f[:, 0]) ** 3 - gamma * matrx_f[:, 1] * matrx_f[:, 2] / matrx_f[:,
                                                                                                          0] ** 2
        A[:, 2, 1] = -1.5 * (gamma - 1) * (matrx_f[:, 1] / matrx_f[:, 0]) ** 2 + gamma * matrx_f[:, 2] / matrx_f[:, 0]
        A[:, 2, 2] = gamma * matrx_f[:, 1] / matrx_f[:, 0]
        return A

    def _get_3d_flux_basic(self, matrx_f_gene):
        gamma = self.gamma
        rho_matrx = matrx_f_gene[:, 0]
        m_matrx = matrx_f_gene[:, 1]
        epsilon_matrx = matrx_f_gene[:, 2]
        F_matrx = np.zeros([len(self.x) + 2, 3])
        F_matrx[1:-1, 0] = matrx_f_gene[:, 1]
        F_matrx[1:-1, 1] = m_matrx ** 2 / rho_matrx + (gamma - 1) * (epsilon_matrx - m_matrx ** 2 / rho_matrx)
        F_matrx[1:-1, 2] = (m_matrx / rho_matrx) * (epsilon_matrx + (gamma - 1) * (epsilon_matrx - m_matrx ** 2 / rho_matrx))
        F_matrx[0] = F_matrx[1]
        F_matrx[-1] = F_matrx[-2]
        return F_matrx

    def _get_1d_flux_basic(self, matrx_f_gene):
        F_matrx = self.a * matrx_f_gene
        return F_matrx

    def _roe_average_values(self, ul_matrx, ur_matrx):
        # ul, ur with shape (3,)
        gamma = self.gamma
        # return \rho, u, H, a
        u_aver = np.zeros((len(self.x) + 1, 4))
        # rho
        u_aver[:, 0] = np.sqrt(ul_matrx[:, 0] * ur_matrx[:, 0])
        # u
        u_aver[:, 1] = ((ul_matrx[:, 1] / np.sqrt(ur_matrx[:, 0])) + (ur_matrx[:, 1] / np.sqrt(ur_matrx[:, 0]))
                     / (np.sqrt(ul_matrx[:, 0]) + np.sqrt(ur_matrx[:, 0])))
        # Hl and Hr
        Hl = (ul_matrx[:, 2] + (gamma - 1) * (ul_matrx[:, 2] - ul_matrx[:, 1] ** 2 / ul_matrx[:, 0])) / ul_matrx[:, 0]
        Hr = (ur_matrx[:, 2] + (gamma - 1) * (ur_matrx[:, 2] - ur_matrx[:, 1] ** 2 / ur_matrx[:, 0])) / ur_matrx[:, 0]
        # H
        u_aver[:, 2] = (Hl * ul_matrx[:, 0] + Hr * ur_matrx[:, 0]) / (np.sqrt(ul_matrx[:, 0]) + np.sqrt(ur_matrx[:, 0]))
        # a
        u_aver[:, 3] = (gamma - 1) * (u_aver[:, 2] - 0.5 * u_aver[:, 1] ** 2)

        return u_aver

    def _roe_R_matrix(self, u_aver):
        u = u_aver[:, 1]
        H = u_aver[:, 2]
        a = u_aver[:, 3]
        R = np.zeros((len(self.x) + 1, 3, 3))
        R[:, 0, :] = 1
        R[:, 1, 0] = u - a
        R[:, 2, 0] = H - u * a
        R[:, 1, 1] = u
        R[:, 2, 1] = 0.5 * u ** 2
        R[:, 1, 2] = u + a
        R[:, 2, 2] = H + u * a
        return R

    def _roe_lambda_at_L_at_delta_U(self, u_aver, ul_matrx, ur_matrx, entropy_fix = None):
        gamma = self.gamma
        # roe average values
        rho = u_aver[:, 0]
        u = u_aver[:, 1]
        a = u_aver[:, 3]
        # rho, u, p, of the field
        rho_matrx_l = ul_matrx[:, 0]
        rho_matrx_r = ur_matrx[:, 0]
        u_matrx_l = ul_matrx[:, 1] / ul_matrx[:, 0]
        u_matrx_r = ur_matrx[:, 1] / ur_matrx[:, 0]
        p_matrx_l = (gamma - 1) * (ul_matrx[:, 2] - ul_matrx[:, 1] ** 2 / ul_matrx[:, 0])
        p_matrx_r = (gamma - 1) * (ur_matrx[:, 2] - ur_matrx[:, 1] ** 2 / ur_matrx[:, 0])
        delta_rho = rho_matrx_r - rho_matrx_l
        delta_u = u_matrx_r - u_matrx_l
        delta_p = p_matrx_r - p_matrx_l
        # matrix |\Lambda|
        if entropy_fix is not None:
            e = entropy_fix
            lambda_matrx_abs = np.zeros((len(self.x) + 1, 3))
            lambda_matrx_abs[:, 0] = np.abs(u - a)
            lambda_matrx_abs[:, 1] = np.abs(u)
            lambda_matrx_abs[:, 2] = np.abs(u + a)
            lambda_another = (lambda_matrx_abs ** 2 + e ** 2) / (2 * e)
            lambda_matrx_abs = np.maximum(lambda_matrx_abs, lambda_another)
        else:
            lambda_matrx_abs = np.zeros((len(self.x) + 1, 3))
            lambda_matrx_abs[:, 0] = np.abs(u - a)
            lambda_matrx_abs[:, 1] = np.abs(u)
            lambda_matrx_abs[:, 2] = np.abs(u + a)
        # column matrix |\Lambda| @ L @ \Delta U
        lambda_L_U = np.zeros((len(self.x) + 1, 3))
        lambda_L_U[:, 0] = lambda_matrx_abs[:, 0] * (delta_p - rho * a * delta_u) / (2 * a ** 2)
        lambda_L_U[:, 1] = lambda_matrx_abs[:, 1] * (delta_rho - delta_p / a ** 2)
        lambda_L_U[:, 2] = lambda_matrx_abs[:, 2] * (delta_p + rho * a * delta_u) / (2 * a ** 2)

        return lambda_L_U

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

        # Flux generator
        def F_gene(matrx_f_gene):
            # u and a (speed of sound) array
            u_matrx, a_matrx = self._1d_eulerian_u_a(matrx_f_gene)
            # The basic flux
            F_matrx = self._get_3d_flux_basic(matrx_f_gene)
            # basic half-node flux
            F_half = 0.5 * (F_matrx[:-1] + F_matrx[1:])
            # artificial viscosity added
            u_abs = np.abs(u_matrx)
            a_abs = np.abs(a_matrx)
            # expanded matrx
            matrx_expand = np.zeros([len(self.x) + 2, 3])
            matrx_expand[1:-1] = matrx_f_gene
            matrx_expand[0] = matrx_expand[1]
            matrx_expand[-1] = matrx_expand[-2]
            # artificial viscosity
            lambda_max = np.maximum(u_abs[:-1] + a_abs[:-1], u_abs[1:] + a_abs[1:])
            vis_matrx = -0.5 * lambda_max * (matrx_expand[1:] - matrx_expand[:-1])
            # final half-node flux
            F_half += vis_matrx
            return F_half

        # compute and plot
        self._1d_3vec_eulerian_explicit(matrx, F_gene, scheme, t_plot)
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

        # Flux generator
        def F_gene(matrx_f_gene):
            # u (velocity) and a (speed of sound) array
            u_matrx, a_matrx = self._1d_eulerian_u_a(matrx_f_gene)   # -1 to l
            # The basic flux
            F_matrx = self._get_3d_flux_basic(matrx_f_gene)
            # basic half-node flux
            F_half = 0.5 * (F_matrx[:-1] + F_matrx[1:]) # -1 to l-1
            # artificial viscosity added
            u_abs = np.abs(u_matrx)
            a_abs = np.abs(a_matrx) # -1 to l
            p_matrx = np.zeros(len(self.x) + 6) # -3 to l+2
            p_matrx[3:-3] = (gamma - 1) * (matrx_f_gene[:, 2] - 0.5 * matrx_f_gene[:, 1] ** 2 / matrx_f_gene[:, 0])
            p_matrx[0] = p_matrx[1] = p_matrx[2] = p_matrx[3]
            p_matrx[-1] = p_matrx[-2] = p_matrx[-3] = p_matrx[-4]
            # viscous parameter $\varepsilon_2$ (-2 to l+1)
            nu_matrx = np.abs(p_matrx[2:] - 2 * p_matrx[1:-1] + p_matrx[:-2]) / np.abs(p_matrx[2:] + 2 * p_matrx[1:-1] + p_matrx[:-2])
            # print(f'nu_matrx shape = {nu_matrx.shape}')
            windows = np.lib.stride_tricks.sliding_window_view(nu_matrx, window_shape=4)
            e2_matrx = k_2 * np.max(windows, axis=1)    # -1 to l-1
            e2_matrx = e2_matrx[:, np.newaxis]
            e4_matrx = np.maximum(0, k_4 - e2_matrx)
            # expanded matrx (from -2 to l+1)
            matrx_expand = np.zeros([len(self.x) + 4, 3])
            matrx_expand[2:-2] = matrx_f_gene
            matrx_expand[0] = matrx_expand[1] = matrx_expand[2]
            matrx_expand[-1] = matrx_expand[-2] = matrx_expand[-3]
            # lambda_max and artificial viscosity
            # lambda_max = 0.5 * (u_abs[:-1] + a_abs[:-1] + u_abs[1:] + a_abs[1:])    # -1 to l-1
            lambda_max = np.maximum(u_abs[:-1] + a_abs[:-1], u_abs[1:] + a_abs[1:])
            vis_matrx_2 = -e2_matrx * lambda_max * (matrx_expand[2:-1] - matrx_expand[1:-2])    # -1 to l-1
            vis_matrx_4 = e4_matrx * lambda_max * (matrx_expand[3:] - 3 * matrx_expand[2:-1] + 3 * matrx_expand[1:-2] - matrx_expand[:-3])  # -1 to l-1
            # final half-node flux
            F_half += (vis_matrx_2 + vis_matrx_4)
            return F_half

        # compute and plot
        self._1d_3vec_eulerian_explicit(matrx, F_gene, scheme, t_plot, k_2 = k_2, k_4 = k_4)
        return 0

    def roe(self, t_plot, entropy_fix = None):
        scheme = 'Roe'
        gamma = self.gamma
        matrx = np.zeros([len(self.x), 3])
        rho_u_p = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        matrx[:, 0] = rho_u_p[:, 0]
        matrx[:, 1] = rho_u_p[:, 0] * rho_u_p[:, 1]
        matrx[:, 2] = (rho_u_p[:, 2] / (gamma - 1)) + 0.5 * rho_u_p[:, 0] * rho_u_p[:, 1] ** 2

        # Flux generator
        def F_gene(matrx_f_gene):
            # The basic flux
            F_matrx = self._get_3d_flux_basic(matrx_f_gene)
            # basic half-node flux
            F_half = 0.5 * (F_matrx[:-1] + F_matrx[1:])  # -1 to l-1
            # expanded matrx (from -1 to l)
            matrx_expand = np.zeros([len(self.x) + 2, 3])
            matrx_expand[1:-1] = matrx_f_gene
            matrx_expand[0] = matrx_expand[1]
            matrx_expand[-1] = matrx_expand[-2]
            # ul and ur
            ul_matrx = matrx_expand[:-1]    # -1 to l-1
            ur_matrx = matrx_expand[1:]     # 0 to l
            # roe numerical flux
            roe_aver_values = self._roe_average_values(ul_matrx, ur_matrx)
            R = self._roe_R_matrix(roe_aver_values)
            lam_L_U = self._roe_lambda_at_L_at_delta_U(roe_aver_values, ul_matrx, ur_matrx,
                                                       entropy_fix = entropy_fix)
            F_roe = - 0.5 * np.einsum('ijk,ik->ij', R, lam_L_U)
            # final half-node flux
            F_half += F_roe
            return F_half   # -1 to l-1

        # compute and plot
        self._1d_3vec_eulerian_rk4(matrx, F_gene, scheme, t_plot, entropy_fix = entropy_fix)
        return 0

    def drp(self, t_plot, ylim = None):
        scheme = 'DRP'
        matrx = np.zeros(len(self.x))
        matrx = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        l = len(matrx)
        # global parameters
        a0 = 0
        a1 = 0.79926643
        a2 = -0.18941314
        a3 = 0.02651995

        # Flux generator
        def F_part_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # expanded basic flux (-3 to l+2)
            F_expand = np.zeros(len(self.x) + 6)
            F_expand[3:-3] = F_matrx
            for i in range(3):
                F_expand[i] = F_matrx[l-3+i]
                F_expand[-i-1] = F_matrx[2-i]

            # discretized \part f\over\part x
            F_part = a3 * (F_expand[: -6] + F_expand[6:])\
                     + a2 * (F_expand[1: -5] + F_expand[5:-1])\
                     + a1 * (F_expand[2: -4] + F_expand[4:-2])\
                     + a0 * F_expand[3: -3]

            return F_part

        # compute and plot
        self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim)
        return 0

    def drp_m(self, t_plot, ylim = None, Re_a = 0.2):
        scheme = 'DRP-M'
        matrx = np.zeros(len(self.x))
        matrx = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        l = len(matrx)
        # global parameters
        a0 = 0
        a1 = 0.79926643
        a2 = -0.18941314
        a3 = 0.02651995
        nu_a_x = self.a / Re_a
        c0 = 0.327698660846
        c1 = -0.235718815308
        c2 = 0.086150669577
        c3 = -0.014281184692

        # Flux generator
        def F_part_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # expanded basic flux (-3 to l+2)
            F_expand = np.zeros(len(self.x) + 6)
            F_expand[3:-3] = F_matrx
            for i in range(3):
                F_expand[i] = F_matrx[l - 3 + i]
                F_expand[-i-1] = F_matrx[2 - i]
            # discretized \part f\over\part x
            F_part = a3 * (F_expand[: -6] + F_expand[6:]) \
                     + a2 * (F_expand[1: -5] + F_expand[5:-1]) \
                     + a1 * (F_expand[2: -4] + F_expand[4:-2]) \
                     + a0 * F_expand[3: -3]
            # viscous term
            F_vis = nu_a_x * (c3 * (F_expand[: -6] + F_expand[6:])
                     + c2 * (F_expand[1: -5] + F_expand[5:-1])
                     + c1 * (F_expand[2: -4] + F_expand[4:-2])
                     + c0 * F_expand[3: -3])
            # fianl
            F_part += F_vis
            return F_part

        # compute and plot
        self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim)
        return 0

    def mdcd(self, t_plot, ylim = None):
        scheme = 'MDCD'
        matrx = np.zeros(len(self.x))
        matrx = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        l = len(matrx)

        # Flux generator
        def F_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # g_disp (-1 to l-1)
            g_disp = 0.0463783
            # g_diss (-1 to l-1)
            g_diss = 0.012

            # expanded basic flux (-3 to l+2)
            F_expand = np.zeros(len(self.x) + 6)
            F_expand[3:-3] = F_matrx
            for i in range(3):
                F_expand[i] = F_matrx[l - 3 + i]
                F_expand[-i-1] = F_matrx[2 - i]
            # half_node flux (-1 to l-1)
            F_half = (0.5 * (g_diss + g_disp) * F_expand[:-5]
                      + (-1.5 * g_disp - 2.5 * g_diss - 1 / 12) * F_expand[1: -4]
                      + (g_disp + 5 * g_diss + 7 / 12) * F_expand[2: -3]
                      + (g_disp - 5 * g_diss + 7 / 12) * F_expand[3: -2]
                      + (-1.5 * g_disp + 2.5 * g_diss - 1 / 12) * F_expand[4: -1]
                      + (0.5 * g_disp - 0.5 * g_diss) * F_expand[5:])
            return F_half  # -1 to l-1

        # compute and plot
        self._1d_1vec_conserv_rk4(matrx, F_gene, scheme, t_plot, ylim=ylim)
        return 0

    def sadrp(self, t_plot, ylim = None):
        scheme = 'SA-DRP'
        matrx = np.zeros(len(self.x))
        matrx = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        l = len(matrx)

        # The coefficient matrix of S1 to C2
        co_matrx_S1 = np.zeros([len(self.x), len(self.x)])
        co_matrx_S2 = np.zeros([len(self.x), len(self.x)])
        co_matrx_S3 = np.zeros([len(self.x), len(self.x)])
        co_matrx_S4 = np.zeros([len(self.x), len(self.x)])
        co_matrx_C1 = np.zeros([len(self.x), len(self.x)])
        co_matrx_C2 = np.zeros([len(self.x), len(self.x)])
        # S1
        for i in range(len(self.x)):
            co_matrx_S1[i, i] = -2
            co_matrx_S1[i, i+1 - ((i+1) // l) * l] = 1
            co_matrx_S1[i, i - 1] = 1
        # S2
        for i in range(len(self.x)):
            co_matrx_S2[i, i] = -2
            co_matrx_S2[i, i+2 - ((i+2) // l) * l] = 1
            co_matrx_S2[i, i - 2] = 1
        co_matrx_S2 = co_matrx_S2 * 0.25
        # S3
        for i in range(len(self.x)):
            co_matrx_S3[i, i] = 1
            co_matrx_S3[i, i+1 - ((i+1) // l) * l] = -2
            co_matrx_S3[i, i+2 - ((i+2) // l) * l] = 1
        # S4
        for i in range(len(self.x)):
            co_matrx_S4[i, i - 1] = 1
            co_matrx_S4[i, i+1 - ((i+1) // l) * l] = -2
            co_matrx_S4[i, i+3 - ((i+3) // l) * l] = 1
        co_matrx_S4 = co_matrx_S4 * 0.25
        # C1
        for i in range(len(self.x)):
            co_matrx_C1[i, i] = -1
            co_matrx_C1[i, i+1 - ((i+1) // l) * l] = 1
        # C2
        for i in range(len(self.x)):
            co_matrx_C2[i, i+2 - ((i+2) // l) * l] = 1
            co_matrx_C2[i, i - 1] = -1
        co_matrx_C2 = co_matrx_C2 / 3

        # Flux generator
        def F_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # 0 to l-1
            S1 = np.einsum('ij, j->i', co_matrx_S1, F_matrx)
            S2 = np.einsum('ij, j->i', co_matrx_S2, F_matrx)
            S3 = np.einsum('ij, j->i', co_matrx_S3, F_matrx)
            S4 = np.einsum('ij, j->i', co_matrx_S4, F_matrx)
            C1 = np.einsum('ij, j->i', co_matrx_C1, F_matrx)
            C2 = np.einsum('ij, j->i', co_matrx_C2, F_matrx)
            # k_esw (0 to l-1)
            e = 1e-8
            expr = (np.abs(np.abs(S1 + S2) - np.abs(S1 - S2))
                    + np.abs(np.abs(S3 + S4) - np.abs(S3 - S4))
                    + np.abs(np.abs(C1 + C2) - 0.5 * np.abs(C1 - C2))
                    + 2 * e) / (np.abs(S1 + S2)
                                + np.abs(S1 - S2)
                                + np.abs(S3 + S4)
                                + np.abs(S3 - S4)
                                + np.abs(C1 + C2)
                                + np.abs(C1 - C2) + e)
            k_esw = np.arccos(2 * (np.minimum(expr, 1)) - 1)
            # g_disp (-1 to l-1)
            mask_p0 = (0 <= k_esw) & (k_esw < 0.01)
            mask_p1 = (0.01 <= k_esw) & (k_esw < 2.5)
            g_disp_ = 0.1985842 * np.ones(len(self.x))
            expr_disp = (k_esw
                         + np.sin(2 * k_esw) / 6
                         - 4 * np.sin(k_esw) / 3) / (np.sin(3 * k_esw) - 4 * np.sin(2 * k_esw) + 5 * np.sin(k_esw))
            g_disp_[mask_p0] = 1 / 30
            g_disp_[mask_p1] = expr_disp[mask_p1]
            g_disp = np.zeros(l + 1)
            g_disp[1:] = g_disp_
            g_disp[0] = g_disp_[-1]
            # g_diss (-1 to l-1)
            mask_s0 = (0 <= k_esw) & (k_esw <= 1)
            g_diss_ = 0.001 * np.ones(len(self.x))
            expr_diss = np.minimum(0.012,
                               0.001 + 0.011 * np.sqrt((k_esw[~mask_s0] - 1) / (np.pi - 1)))
            g_diss_[~mask_s0] = expr_diss
            g_diss = np.zeros(l + 1)
            g_diss[1:] = g_diss_
            g_diss[0] = g_diss_[-1]

            # expanded basic flux (-3 to l+2)
            F_expand = np.zeros(len(self.x) + 6)
            F_expand[3:-3] = F_matrx
            for i in range(3):
                F_expand[i] = F_matrx[l - 3 + i]
                F_expand[-i - 1] = F_matrx[2 - i]
            # half_node flux (-1 to l-1)
            F_half = (0.5 * (g_diss + g_disp) * F_expand[:-5]
                      + (-1.5 * g_disp - 2.5 * g_diss - 1/12) * F_expand[1: -4]
                      + (g_disp + 5 * g_diss + 7/12) * F_expand[2: -3]
                      + (g_disp - 5 * g_diss + 7/12) * F_expand[3: -2]
                      + (-1.5 * g_disp + 2.5 * g_diss - 1/12) * F_expand[4: -1]
                      + (0.5 * g_disp - 0.5 * g_diss) * F_expand[5:])
            return F_half  # -1 to l-1

        # compute and plot
        self._1d_1vec_conserv_rk4(matrx, F_gene, scheme, t_plot, ylim = ylim)
        return 0