import numpy as np
import matplotlib.pyplot as plt
# import time, sys
import os
import cupy as cp


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

    # plot functions

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
                  mesh = None, cfl = True, k_2 = None, k_4 = None, ylim = None, m = None):
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
            if m is not None:
                file_subfolder = os.path.join(file_subfolder, f'@m = {m}')
            if not os.path.exists(file_subfolder):
                os.makedirs(file_subfolder)
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Velocity')
        plt.title(f"Solution at Time={time:.3f},step = {int(time / self.dt)}")
        plt.xlabel("x")
        plt.ylabel("Velocity")
        if ylim is not None:
            plt.ylim(ylim)
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

    # boundary condition
    def _periodic_BDC_initialize_1Dscalar(self):
        matrx = np.zeros(len(self.x))
        matrx[:-1] = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0] - 1)])
        matrx[-1] = matrx[0]
        return matrx

    # computing functions (and time discretization)

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
                             ylim=None,
                             m = None):
        matrx = matrx_ini.copy()

        # discrete \part f\over\part x
        def F_part_gene(matrx_gene):
            F_half = F_gene(matrx_gene) # -1 to l-1
            F_part = F_half[1:] - F_half[:-1]
            return F_part

        matrx_re = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot,
                          mesh = mesh, k_2 = k_2, k_4 = k_4, ylim = ylim, m = m)
        return matrx_re

    def _1d_1vec_rk4(self,
                     matrx_ini,
                     F_part_gene: callable,
                     scheme: str,
                     t_plot,
                     mesh=True,
                     k_2=None,
                     k_4=None,
                     ylim=None,
                     m = None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_part = F_part_gene(matrx_f)  # 0 to l-1
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
                                   cfl=False, mesh=mesh, k_2=k_2, k_4=k_4, ylim=ylim, m = m)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        print('\n')
        return matrx

    def _1d_1vec_conserv_rk4_cp(self,
                             matrx_ini,
                             F_gene: callable,
                             scheme: str,
                             t_plot,
                             mesh=True,
                             k_2=None,
                             k_4=None,
                             ylim=None,
                             m = None):
        matrx = matrx_ini.copy()

        # discrete \part f\over\part x
        def F_part_gene(matrx_gene):
            F_half = F_gene(matrx_gene) # -1 to l-1
            F_part = F_half[1:] - F_half[:-1]
            return F_part

        matrx_re = self._1d_1vec_rk4_cp(matrx, F_part_gene, scheme, t_plot,
                          mesh = mesh, k_2 = k_2, k_4 = k_4, ylim = ylim, m = m)
        return matrx_re

    def _1d_1vec_rk4_cp(self,
                        matrx_ini,
                        F_part_gene: callable,
                        scheme: str,
                        t_plot,
                        mesh=True,
                        k_2=None,
                        k_4=None,
                        ylim=None,
                        m=None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_part = F_part_gene(matrx_f)  # 0 to l-1
            matrx_1 = matrx_f - 0.25 * t_x * F_part
            F_part_1 = F_part_gene(matrx_1)
            matrx_2 = matrx_f - t_x * F_part_1 / 3
            F_part_2 = F_part_gene(matrx_2)
            matrx_3 = matrx_f - 0.5 * t_x * F_part_2
            F_part_3 = F_part_gene(matrx_3)
            matrx = matrx_f - t_x * F_part_3
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    matrx_n = matrx.get()
                    self._plot_cfl(matrx_n, time, scheme,
                                   cfl=False, mesh=mesh, k_2=k_2, k_4=k_4, ylim=ylim, m=m)
        matrx = matrx.get()
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        print('\n')
        return matrx

    def _1d_3vec_tvd_rk3(self,
                         matrx_ini,
                         F_gene: callable,
                         scheme: str,
                         t_plot,
                         mesh=True,
                         k_2=None,
                         k_4=None,
                         entropy_fix=None):
        matrx = matrx_ini.copy()
        t_x = self.dt / self.dx
        for i in range(1, len(self.t)):
            matrx_f = matrx.copy()
            F_half = F_gene(matrx_f)
            matrx_1 = matrx_f - t_x * (F_half[1:] - F_half[:-1])
            F_half_1 = F_gene(matrx_1)
            matrx_2 = 0.75 * matrx_f + 0.25 * (matrx_1 - t_x * (F_half_1[1:] - F_half_1[:-1]))
            F_half_2 = F_gene(matrx_2)
            matrx_3 = matrx_f / 3 + 2 * (matrx_2 - t_x * (F_half_2[1:] - F_half_2[:-1])) / 3
            matrx = matrx_3
            for time in t_plot:
                if self.t[i] <= time < self.t[i + 1]:
                    self._plot_1d_3vec(matrx, time, scheme,
                                       cfl=False, mesh=mesh, k_2=k_2, k_4=k_4, entropy_fix=entropy_fix)
        print(f'case: {self.name}, scheme: {scheme}')
        print(f'Space range: from {self.left_x} to {self.right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
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


    # private methods

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

    def _get_3d_flux_basic_uniform(self, matrx_f_gene):
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
        return F_matrx  # -1 to l

    def _get_3d_flux_basic_local(self, matrx_f_gene):
        gamma = self.gamma
        rho_matrx = matrx_f_gene[:, 0]
        m_matrx = matrx_f_gene[:, 1]
        epsilon_matrx = matrx_f_gene[:, 2]
        F_matrx = np.zeros([len(matrx_f_gene), 3])
        F_matrx[:, 0] = matrx_f_gene[:, 1]
        F_matrx[:, 1] = m_matrx ** 2 / rho_matrx + (gamma - 1) * (epsilon_matrx - m_matrx ** 2 / rho_matrx)
        F_matrx[:, 2] = (m_matrx / rho_matrx) * (
                    epsilon_matrx + (gamma - 1) * (epsilon_matrx - m_matrx ** 2 / rho_matrx))
        return F_matrx  # same shape with matrx_f_gene

    def _get_1d_flux_basic(self, matrx_f_gene):
        F_matrx = self.a * matrx_f_gene
        return F_matrx

    def _roe_average_values(self, ul_matrx, ur_matrx):
        # ul_matrx, ur_matrx with shape (n,3)
        gamma = self.gamma
        # return \rho, u, H, a
        u_aver = np.zeros((len(ul_matrx), 4))
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

        return u_aver   # same length with ul_matrx, ur_matrx

    def _roe_R_matrix(self, u_aver):
        u = u_aver[:, 1]
        H = u_aver[:, 2]
        a = u_aver[:, 3]
        R = np.zeros((len(u_aver), 3, 3))
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

    # wrong expression of L
    def _roe_L_matrix(self, u_aver):
        gamma = self.gamma
        u = u_aver[:, 1]
        a = u_aver[:, 3]
        L = np.zeros((len(self.x) + 2, 3, 3))
        L[:, 0, 0] = 0.5 * (gamma - 1) * u ** 2 / a ** 2 + u / a
        L[:, 0, 1] = -((gamma - 1) * u / a ** 2 + 1 / a)
        L[:, 0, 2] = (gamma - 1) / a ** 2
        L[:, 1, 0] = 2 - (gamma - 1) * u ** 2 / a ** 2
        L[:, 1, 1] = 2 * (gamma - 1) * u / a ** 2
        L[:, 1, 2] = -2 * (gamma - 1) / a ** 2
        L[:, 2, 0] = 0.5 * (gamma - 1) * u ** 2 / a ** 2 - u / a
        L[:, 2, 1] = -((gamma - 1) * u / a ** 2 - 1 / a)
        L[:, 2, 2] = (gamma - 1) / a ** 2
        L = 0.5 * L
        return L    # with shape (len(u_aver), 3, 3)

    def _roe_lambda_matrix(self, u_aver):
        u = u_aver[:, 1]
        a = u_aver[:, 3]
        lambda_matrx = np.zeros((len(u_aver), 3, 3))
        lambda_matrx[:, 0, 0] = u - a
        lambda_matrx[:, 1, 1] = u
        lambda_matrx[:, 2, 2] = u + a
        return lambda_matrx

    def _minmod(self, matrx1, matrx2):
        return 0.5 * (np.sign(matrx1) + np.sign(matrx2))\
            * np.minimum(np.abs(matrx1), np.abs(matrx2))

    def _tvd_delta_x_Dj(self, u_aver, ul_matrx, ur_matrx, uc_matrx):
        gamma = self.gamma
        # roe average values
        rho = u_aver[:, 0]
        u = u_aver[:, 1]
        a = u_aver[:, 3]
        # rho, u, p, of the field
        rho_matrx_l = ul_matrx[:, 0]
        rho_matrx_r = ur_matrx[:, 0]
        rho_matrx_c = uc_matrx[:, 0]
        u_matrx_l = ul_matrx[:, 1] / ul_matrx[:, 0]
        u_matrx_r = ur_matrx[:, 1] / ur_matrx[:, 0]
        u_matrx_c = uc_matrx[:, 1] / uc_matrx[:, 0]
        p_matrx_l = (gamma - 1) * (ul_matrx[:, 2] - ul_matrx[:, 1] ** 2 / ul_matrx[:, 0])
        p_matrx_r = (gamma - 1) * (ur_matrx[:, 2] - ur_matrx[:, 1] ** 2 / ur_matrx[:, 0])
        p_matrx_c = (gamma - 1) * (uc_matrx[:, 2] - uc_matrx[:, 1] ** 2 / uc_matrx[:, 0])
        delta_rho = rho_matrx_r - rho_matrx_l
        delta_u = u_matrx_r - u_matrx_l
        delta_p = p_matrx_r - p_matrx_l
        # matrix |\Lambda|
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

    # scheme programs

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
            F_matrx = self._get_3d_flux_basic_uniform(matrx_f_gene)
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
            F_matrx = self._get_3d_flux_basic_uniform(matrx_f_gene)
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
            F_matrx = self._get_3d_flux_basic_uniform(matrx_f_gene)
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
            roe_aver_values = self._roe_average_values(ul_matrx, ur_matrx)  # -1 to l-1
            R = self._roe_R_matrix(roe_aver_values) # -1 to l-1
            lam_L_U = self._roe_lambda_at_L_at_delta_U(roe_aver_values, ul_matrx, ur_matrx,
                                                       entropy_fix = entropy_fix)
            F_roe = - 0.5 * np.einsum('ijk,ik->ij', R, lam_L_U)
            # final half-node flux
            F_half += F_roe
            return F_half   # -1 to l-1

        # compute and plot
        self._1d_3vec_eulerian_rk4(matrx, F_gene, scheme, t_plot, entropy_fix = entropy_fix)
        return 0

    # The periodic BDC to be reconstructed
    def drp(self, t_plot, ylim = None, m = None):
        scheme = 'DRP'
        matrx = self._periodic_BDC_initialize_1Dscalar()
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
            F_expand[:3] = F_matrx[-4:-1]
            F_expand[-4:] = F_matrx[:4]

            # discretized \part f\over\part x
            F_part = a3 * (-F_expand[: -6] + F_expand[6:])\
                     + a2 * (-F_expand[1: -5] + F_expand[5:-1])\
                     + a1 * (-F_expand[2: -4] + F_expand[4:-2])\
                     + a0 * F_expand[3: -3]

            return F_part

        # compute and plot
        result = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim, m = m)
        return result

    def drp_m(self, t_plot, ylim = None, Re_a = 1, m = None):
        scheme = 'DRP-M'
        matrx = self._periodic_BDC_initialize_1Dscalar()
        l = len(matrx)
        # global parameters
        a0 = 0
        a1 = 0.770882380518
        a2 = -0.166705904415
        a3 = 0.020843142770

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
            F_expand[:3] = F_matrx[-4:-1]
            F_expand[-4:] = F_matrx[:4]
            # discretized \part f\over\part x
            F_part = a3 * (-F_expand[: -6] + F_expand[6:]) \
                     + a2 * (-F_expand[1: -5] + F_expand[5:-1]) \
                     + a1 * (-F_expand[2: -4] + F_expand[4:-2]) \
                     + a0 * F_expand[3: -3]
            # viscous term
            F_vis = (1 / Re_a) * (c3 * (F_expand[: -6] + F_expand[6:])
                     + c2 * (F_expand[1: -5] + F_expand[5:-1])
                     + c1 * (F_expand[2: -4] + F_expand[4:-2])
                     + c0 * F_expand[3: -3])
            # fianl
            F_part += F_vis
            return F_part

        # compute and plot
        result = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim, m = m)
        return result

    def mdcd(self, t_plot, ylim = None, m = None):
        scheme = 'MDCD'
        matrx = self._periodic_BDC_initialize_1Dscalar()
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
            F_expand[:3] = F_matrx[-4:-1]
            F_expand[-4:] = F_matrx[:4]
            # half_node flux (-1 to l-1)
            F_half = (0.5 * (g_diss + g_disp) * F_expand[:-5]
                      + (-1.5 * g_disp - 2.5 * g_diss - 1 / 12) * F_expand[1: -4]
                      + (g_disp + 5 * g_diss + 7 / 12) * F_expand[2: -3]
                      + (g_disp - 5 * g_diss + 7 / 12) * F_expand[3: -2]
                      + (-1.5 * g_disp + 2.5 * g_diss - 1 / 12) * F_expand[4: -1]
                      + (0.5 * g_disp - 0.5 * g_diss) * F_expand[5:])
            return F_half  # -1 to l-1

        # compute and plot
        result = self._1d_1vec_conserv_rk4(matrx, F_gene, scheme, t_plot, ylim=ylim, m = m)
        return result

    def sadrp(self, t_plot, ylim = None, m = None):
        scheme = 'SA-DRP'
        matrx = self._periodic_BDC_initialize_1Dscalar()
        l = len(matrx)
        l_ = l-1

        # The coefficient matrix of S1 to C2 (0 to l-2)
        co_matrx_S1 = np.zeros([len(self.x)-1, len(self.x)-1])
        co_matrx_S2 = np.zeros([len(self.x)-1, len(self.x)-1])
        co_matrx_S3 = np.zeros([len(self.x)-1, len(self.x)-1])
        co_matrx_S4 = np.zeros([len(self.x)-1, len(self.x)-1])
        co_matrx_C1 = np.zeros([len(self.x)-1, len(self.x)-1])
        co_matrx_C2 = np.zeros([len(self.x)-1, len(self.x)-1])
        # S1
        for i in range(l_):
            co_matrx_S1[i, i] = -2
            co_matrx_S1[i, i+1 - ((i+1) // l_) * l_] = 1
            co_matrx_S1[i, i - 1] = 1
        # S2
        for i in range(l_):
            co_matrx_S2[i, i] = -2
            co_matrx_S2[i, i+2 - ((i+2) // l_) * l_] = 1
            co_matrx_S2[i, i - 2] = 1
        co_matrx_S2 = co_matrx_S2 * 0.25
        # S3
        for i in range(l_):
            co_matrx_S3[i, i] = 1
            co_matrx_S3[i, i+1 - ((i+1) // l_) * l_] = -2
            co_matrx_S3[i, i+2 - ((i+2) // l_) * l_] = 1
        # S4
        for i in range(l_):
            co_matrx_S4[i, i - 1] = 1
            co_matrx_S4[i, i+1 - ((i+1) // l_) * l_] = -2
            co_matrx_S4[i, i+3 - ((i+3) // l_) * l_] = 1
        co_matrx_S4 = co_matrx_S4 * 0.25
        # C1
        for i in range(l_):
            co_matrx_C1[i, i] = -1
            co_matrx_C1[i, i+1 - ((i+1) // l_) * l_] = 1
        # C2
        for i in range(l_):
            co_matrx_C2[i, i+2 - ((i+2) // l_) * l_] = 1
            co_matrx_C2[i, i - 1] = -1
        co_matrx_C2 = co_matrx_C2 / 3

        # Flux generator
        def F_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # 0 to l-1
            F_short = F_matrx[:-1]
            S1 = np.einsum('ij,j->i',co_matrx_S1, F_short, optimize = "optimal")
            S2 = np.einsum('ij,j->i',co_matrx_S2, F_short, optimize = "optimal")
            S3 = np.einsum('ij,j->i',co_matrx_S3, F_short, optimize = "optimal")
            S4 = np.einsum('ij,j->i',co_matrx_S4, F_short, optimize = "optimal")
            C1 = np.einsum('ij,j->i',co_matrx_C1, F_short, optimize = "optimal")
            C2 = np.einsum('ij,j->i',co_matrx_C2, F_short, optimize = "optimal")
            S1 = np.concatenate((S1, [S1[0]]), axis = 0)
            S2 = np.concatenate((S2, [S2[0]]), axis = 0)
            S3 = np.concatenate((S3, [S3[0]]), axis = 0)
            S4 = np.concatenate((S4, [S4[0]]), axis = 0)
            C1 = np.concatenate((C1, [C1[0]]), axis = 0)
            C2 = np.concatenate((C2, [C2[0]]), axis = 0)
            # sum and diffs
            s1_p_s2 = S1 + S2
            s1_m_s2 = S1 - S2
            s3_p_s4 = S3 + S4
            s3_m_s4 = S3 - S4
            c1_p_c2 = C1 + C2
            c1_m_c2 = C1 - C2
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
            F_expand[:3] = F_matrx[-4:-1]
            F_expand[-4:] = F_matrx[:4]
            # half_node flux (-1 to l-1)
            F_half = (0.5 * (g_diss + g_disp) * F_expand[:-5]
                      + (-1.5 * g_disp - 2.5 * g_diss - 1/12) * F_expand[1: -4]
                      + (g_disp + 5 * g_diss + 7/12) * F_expand[2: -3]
                      + (g_disp - 5 * g_diss + 7/12) * F_expand[3: -2]
                      + (-1.5 * g_disp + 2.5 * g_diss - 1/12) * F_expand[4: -1]
                      + (0.5 * g_disp - 0.5 * g_diss) * F_expand[5:])
            return F_half  # -1 to l-1

        # compute and plot
        result = self._1d_1vec_conserv_rk4(matrx, F_gene, scheme, t_plot, ylim = ylim, m = m)
        return result

    def sadrp_cp(self, t_plot, ylim=None, m=None):
        scheme = 'SA-DRP'
        matrx = self._periodic_BDC_initialize_1Dscalar()
        matrx_cp = cp.asarray(matrx)
        l = len(matrx)
        l_ = l - 1

        # The coefficient matrix of S1 to C2 (0 to l-2) on GPU
        # S1
        co_matrx_S1 = cp.diag(cp.full(l_, -2)) + cp.diag(cp.ones(l_ - 1), k=1) + cp.diag(cp.ones(l_ - 1), k=-1)
        co_matrx_S1[0, l_ - 1] = 1
        co_matrx_S1[l_ - 1, 0] = 1
        # S2
        co_matrx_S2 = cp.diag(cp.full(l_, -2)) + cp.diag(cp.ones(l_ - 2), k=2) + cp.diag(cp.ones(l_ - 2), k=-2)
        co_matrx_S2[0, l_ - 2] = 1
        co_matrx_S2[1, l_ - 1] = 1
        co_matrx_S2[l_ - 2, 0] = 1
        co_matrx_S2[l_ - 1, 1] = 1
        co_matrx_S2 *= 0.25
        # S3
        co_matrx_S3 = cp.diag(cp.full(l_, 1)) + cp.diag(-2 * cp.ones(l_ - 1), k=1) + cp.diag(cp.ones(l_ - 2), k=-2)
        co_matrx_S1[l_ - 1, 0] = -2
        co_matrx_S2[l_ - 2, 0] = 1
        co_matrx_S2[l_ - 1, 1] = 1
        # S4
        co_matrx_S4 = cp.diag(cp.ones(l_ - 1), k=-1) + cp.diag(-2 * cp.ones(l_ - 1), k=1) + cp.diag(cp.ones(l_ - 3),
                                                                                                    k=3)
        co_matrx_S4[0, l_ - 1] = 1
        co_matrx_S4[l_ - 1, 0] = -2
        co_matrx_S4[l_ - 3, 0] = 1
        co_matrx_S4[l_ - 2, 1] = 1
        co_matrx_S4[l_ - 1, 2] = 3
        co_matrx_S4 = co_matrx_S4 * 0.25
        # C1
        co_matrx_C1 = cp.diag(cp.full(l_, -1)) + cp.diag(cp.ones(l_ - 1), k=1)
        co_matrx_C1[l_ - 1, 0] = 1
        # C2
        co_matrx_C2 = cp.diag(cp.ones(l_ - 2), k=2) + cp.diag(-1 * cp.ones(l_ - 1), k=-1)
        co_matrx_C2[l_ - 1, 1] = 1
        co_matrx_C2[l_ - 2, 0] = 1
        co_matrx_C2[0, l_ - 1] = -1
        co_matrx_C2 = co_matrx_C2 / 3

        # Flux generator
        def F_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # by cupy
            F_cp = cp.asarray(F_matrx)
            # 0 to l-1
            S1 = cp.einsum('ij, j->i', co_matrx_S1, F_cp[:-1])
            S2 = cp.einsum('ij, j->i', co_matrx_S2, F_cp[:-1])
            S3 = cp.einsum('ij, j->i', co_matrx_S3, F_cp[:-1])
            S4 = cp.einsum('ij, j->i', co_matrx_S4, F_cp[:-1])
            C1 = cp.einsum('ij, j->i', co_matrx_C1, F_cp[:-1])
            C2 = cp.einsum('ij, j->i', co_matrx_C2, F_cp[:-1])
            S1_ele = cp.expand_dims(S1[0], axis=0)
            S2_ele = cp.expand_dims(S2[0], axis=0)
            S3_ele = cp.expand_dims(S3[0], axis=0)
            S4_ele = cp.expand_dims(S4[0], axis=0)
            C1_ele = cp.expand_dims(C1[0], axis=0)
            C2_ele = cp.expand_dims(C2[0], axis=0)
            S1 = cp.concatenate((S1, S1_ele), axis=0)
            S2 = cp.concatenate((S2, S2_ele), axis=0)
            S3 = cp.concatenate((S3, S3_ele), axis=0)
            S4 = cp.concatenate((S4, S4_ele), axis=0)
            C1 = cp.concatenate((C1, C1_ele), axis=0)
            C2 = cp.concatenate((C2, C2_ele), axis=0)
            # k_esw (0 to l-1)
            e = 1e-8
            expr = (cp.abs(cp.abs(S1 + S2) - cp.abs(S1 - S2))
                    + cp.abs(cp.abs(S3 + S4) - cp.abs(S3 - S4))
                    + cp.abs(cp.abs(C1 + C2) - 0.5 * cp.abs(C1 - C2))
                    + 2 * e) / (cp.abs(S1 + S2)
                                + cp.abs(S1 - S2)
                                + cp.abs(S3 + S4)
                                + cp.abs(S3 - S4)
                                + cp.abs(C1 + C2)
                                + cp.abs(C1 - C2) + e)
            k_esw = cp.arccos(2 * (cp.minimum(expr, 1)) - 1)
            # g_disp (-1 to l-1)
            mask_p0 = (0 <= k_esw) & (k_esw < 0.01)
            mask_p1 = (0.01 <= k_esw) & (k_esw < 2.5)
            g_disp_ = 0.1985842 * cp.ones(len(self.x))
            expr_disp = (k_esw
                         + cp.sin(2 * k_esw) / 6
                         - 4 * cp.sin(k_esw) / 3) / (cp.sin(3 * k_esw) - 4 * cp.sin(2 * k_esw) + 5 * cp.sin(k_esw))
            g_disp_[mask_p0] = 1 / 30
            g_disp_[mask_p1] = expr_disp[mask_p1]
            g_disp = cp.zeros(l + 1)
            g_disp[1:] = g_disp_
            g_disp[0] = g_disp_[-1]
            # g_diss (-1 to l-1)
            mask_s0 = (0 <= k_esw) & (k_esw <= 1)
            g_diss_ = 0.001 * cp.ones(len(self.x))
            expr_diss = cp.minimum(0.012,
                                   0.001 + 0.011 * cp.sqrt((k_esw[~mask_s0] - 1) / (cp.pi - 1)))
            g_diss_[~mask_s0] = expr_diss
            g_diss = cp.zeros(l + 1)
            g_diss[1:] = g_diss_
            g_diss[0] = g_diss_[-1]

            # expanded basic flux (-3 to l+2)
            F_expand = cp.zeros(len(self.x) + 6)
            F_expand[3:-3] = F_cp
            F_expand[:3] = F_cp[-4:-1]
            F_expand[-4:] = F_cp[:4]
            # half_node flux (-1 to l-1)
            F_half = (0.5 * (g_diss + g_disp) * F_expand[:-5]
                      + (-1.5 * g_disp - 2.5 * g_diss - 1 / 12) * F_expand[1: -4]
                      + (g_disp + 5 * g_diss + 7 / 12) * F_expand[2: -3]
                      + (g_disp - 5 * g_diss + 7 / 12) * F_expand[3: -2]
                      + (-1.5 * g_disp + 2.5 * g_diss - 1 / 12) * F_expand[4: -1]
                      + (0.5 * g_disp - 0.5 * g_diss) * F_expand[5:])
            # release Video memory
            '''del S1, S2, S3, S4, C1, C2, S1_ele, S2_ele, S3_ele, S4_ele, C1_ele, C2_ele
            del g_diss, g_disp, g_disp_, k_esw, mask_p1, mask_p0, mask_s0, expr_disp, expr_diss, F_expand
            cp.get_default_memory_pool().free_all_blocks()'''
            return F_half  # -1 to l-1

        # compute and plot
        result = self._1d_1vec_conserv_rk4_cp(matrx_cp, F_gene, scheme, t_plot, ylim=ylim, m=m)
        '''del co_matrx_S1, co_matrx_S2, co_matrx_S3, co_matrx_S4, co_matrx_C1, co_matrx_C2
        cp.get_default_memory_pool().free_all_blocks()'''
        return result

    def upwind1(self, t_plot, ylim = None, m = None):
        scheme = '1_UPWIMD'
        matrx = self._periodic_BDC_initialize_1Dscalar()

        def F_part_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # expanded basic flux (-1 to l-1)
            F_expand = np.zeros(len(self.x) + 1)
            F_expand[1:-1] = F_matrx[:-1]
            F_expand[-1] = F_matrx[0]
            F_expand[0] = F_matrx[-2]

            # discretized \part f\over\part x
            F_part = F_matrx - F_expand[:-1]
            return F_part

        # compute and plot
        result = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim, m = m)
        return result

    def upwind2(self, t_plot, ylim = None, m = None):
        scheme = '2_UPWIMD'
        matrx = self._periodic_BDC_initialize_1Dscalar()

        def F_part_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # expanded basic flux (-2 to l-1)
            F_expand = np.zeros(len(self.x) + 2)
            F_expand[2:] = F_matrx
            F_expand[:2] = F_matrx[-3:-1]

            # discretized \part f\over\part x
            F_part = 0.5 * (3*F_matrx - 4*F_expand[1:-1] + F_expand[:-2])
            return F_part

        # compute and plot
        result = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim, m=m)
        return result

    def upwind3(self, t_plot, ylim = None, m = None):
        scheme = '3_UPWIMD'
        matrx = self._periodic_BDC_initialize_1Dscalar()

        def F_part_gene(matrx_f_gene):
            # The basic flux (0 to l-1)
            F_matrx = self._get_1d_flux_basic(matrx_f_gene)
            # expanded basic flux (-2 to l)
            F_expand = np.zeros(len(self.x) + 3)
            F_expand[2:-1] = F_matrx
            F_expand[-1] = F_matrx[1]
            F_expand[:2] = F_matrx[-3:-1]

            # discretized \part f\over\part x
            F_part = (2 * F_expand[3:] + 3 * F_matrx - 6 * F_expand[1:-2] + F_expand[:-3]) / 6
            return F_part

        # compute and plot
        result = self._1d_1vec_rk4(matrx, F_part_gene, scheme, t_plot, ylim=ylim, m=m)
        return result

    def tvd_minmod(self, t_plot, y_lim = None, entropy_fix = None):
        scheme = 'TVD'
        gamma = self.gamma
        t_x = self.dt / self.dx
        matrx = np.zeros([len(self.x), 3])
        rho_u_p = np.array([self.init_condition(self.x[0] + i * self.dx) for i in range(matrx.shape[0])])
        matrx[:, 0] = rho_u_p[:, 0]
        matrx[:, 1] = rho_u_p[:, 0] * rho_u_p[:, 1]
        matrx[:, 2] = (rho_u_p[:, 2] / (gamma - 1)) + 0.5 * rho_u_p[:, 0] * rho_u_p[:, 1] ** 2

        # Flux generator
        def F_gene(matrx_f_gene):
            # matrx_f_gene: 0 to l-1
            # expanded matrx (from -2 to l+1)
            matrx_expand = np.zeros([len(self.x) + 4, 3])
            matrx_expand[2:-2] = matrx_f_gene
            matrx_expand[0] = matrx_expand[1] = matrx_expand[2]
            matrx_expand[-1] = matrx_expand[-2] = matrx_expand[-3]
            # ul and ur for roe linearization
            ul_matrx = matrx_expand[1: -1]  # -1 to l
            ur_matrx = matrx_expand[2:]  # 0 to l+1
            # roe numerical flux
            roe_aver_values = self._roe_average_values(ul_matrx, ur_matrx)  # -1/2 to l + 1/2
            R = self._roe_R_matrix(roe_aver_values)[:-1] # -1/2 to l - 1/2
            L = self._roe_L_matrix(roe_aver_values) # -1/2 to l + 1/2
            Lambda = self._roe_lambda_matrix(roe_aver_values)[:-1]   # -1/2 to l - 1/2
            # U_j+1 - U_j (j: -1 to l)
            diff_p = ur_matrx - ul_matrx
            # U_j - U_j-1 (j: -1 to l)
            diff_m = ul_matrx - matrx_expand[:-2]
            # D_j * \Delta x (-1 to l)
            D_matrx = self._minmod(diff_p, diff_m)
            uL_matrx = ul_matrx[:-1] + 0.5 * D_matrx[:-1]
            uR_matrx = ur_matrx[:-1] - 0.5 * D_matrx[1:]
            '''# D_j * \Delta x  (-1 to l)
            W_p = np.einsum('ijk,ik->ij', L, diff_p)
            W_m = np.einsum('ijk,ik->ij', L, diff_m)
            D_matrx = self._minmod(W_p, W_m)   # limiter
            # W^L and W^R (-1 to l-1)
            L_cut = L[:-1]  # -1 to l-1
            iden = np.eye(3)
            iden = np.tile(iden, (len(L_cut), 1, 1))    # identity matrix
            wl_matrx = np.einsum('ijk,ik->ij', L_cut, ul_matrx[:-1]) \
                       + 0.5 * np.einsum('ijk,ik->ij', iden - t_x * Lambda, D_matrx[:-1])
            wr_matrx = np.einsum('ijk,ik->ij', L_cut, ur_matrx[:-1]) \
                       - 0.5 * np.einsum('ijk,ik->ij', iden + t_x * Lambda, D_matrx[1:])
            # u^L and u^R for flux (-1 to l-1)
            uL_matrx = np.einsum('ijk,ik->ij', R, wl_matrx)
            uR_matrx = np.einsum('ijk,ik->ij', R, wr_matrx)'''
            # half_node matrx with minmod (-1 to l-1)
            F_half = 0.5 * (self._get_3d_flux_basic_local(uL_matrx) + self._get_3d_flux_basic_local(uR_matrx))
            # second part by direct computation (-1 to l-1)
            L_cut = L[:-1]  # -1 to l-1
            if entropy_fix is not None:
                e = entropy_fix
                Lambda_another = (np.abs(Lambda) ** 2 + e ** 2) / (2 * e)
                lambda_matrx_abs = np.maximum(np.abs(Lambda), Lambda_another)
                R_lambda_abs = np.einsum('ijk,ikl->ijl', R, lambda_matrx_abs)
            else:
                R_lambda_abs = np.einsum('ijk,ikl->ijl', R, np.abs(Lambda))
            L_U_r_l = np.einsum('ijk,ik->ij', L_cut, uR_matrx - uL_matrx)
            F_roe = -0.5 * np.einsum('ijk,ik->ij', R_lambda_abs, L_U_r_l)
            # final numerical flux
            F_half += F_roe
            return F_half  # -1 to l-1

        # compute and plot
        self._1d_3vec_tvd_rk3(matrx, F_gene, scheme, t_plot)
        return 0