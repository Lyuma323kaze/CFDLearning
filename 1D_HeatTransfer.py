"""
The generated plots are saved in the folder WorkWeek3, and the names of the subfolders
shows the schemes, boundary conditions used and the sigmas of the cases.
"""

import numpy as np
import matplotlib.pyplot as plt
# import time, sys
import os

# Saving folder
if not os.path.exists('WorkWeek3'):
    os.makedirs('WorkWeek3')
file_folder = 'WorkWeek3'

# OOP
class DiffSchemes:
    def __init__(self, name, dt, dx, sigma, x, t, init_condition, bound_condition):
        self.dt = dt
        self.dx = dx
        self.sigma = sigma
        self.x = x
        self.t = t
        self.init_condition = init_condition
        self.bound_condition = bound_condition
        self.name = name

    def plot(self, result, time, scheme):
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

    def ftcs(self, t_plot):
        # initial and boundary
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(i * self.dx) for i in range(matrx.shape[0])])
        bd_vec = self.bound_condition
        # compute
        for i in range(1, len(self.t)):
            rhs = matrx[:, i - 1].copy()
            rhs[0], rhs[-1] = bd_vec(i * self.dt)
            matrx[:, i - 1] = rhs
            for j in range(1, len(self.x) - 1):
                matrx[j, i] = self.sigma * matrx[j + 1, i - 1] + (1 - 2 * self.sigma) * matrx[j, i - 1] + self.sigma * matrx[j - 1, i - 1]
        # plot
        for time in t_plot:
            self.plot(matrx[:, int(time / self.dt)], time, 'FTCS')
        print(f'case: {self.name}, scheme: FTCS')
        print(f'Space range: from {left_x} to {right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def btcs(self, t_plot):
        # initial and boundary
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(i * self.dx) for i in range(matrx.shape[0])])
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
            self.plot(matrx[:, int(time / self.dt)], time, 'BTCS')
        print(f'case: {self.name}, scheme: BTCS')
        print(f'Space range: from {left_x} to {right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx

    def ctcs(self, t_plot):
        # initial and boundary
        matrx = np.zeros([len(self.x), len(self.t)])
        matrx[:, 0] = np.array([self.init_condition(i * self.dx) for i in range(matrx.shape[0])])
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
            self.plot(matrx[:, int(time / self.dt)], time, 'CTCS')
        print(f'case: {self.name}, scheme: CTCS')
        print(f'Space range: from {left_x} to {right_x}.')
        print('time interval:', self.dt)
        print('space interval:', self.dx)
        return matrx


# domain parameters
left_x = 0
right_x = 1
len_x = right_x - left_x
t_terminate1 = 0.4
t_terminate2 = 1


# mesh parameters
mx = 121    # mesh point number
gamma = 1   # heat transfer coefficient
sigma = 1.0 # difference coefficient
dx = len_x / (mx - 1)
dt = sigma * dx ** 2 / gamma


# set of mesh points and plot points
x_range = np.arange(left_x, right_x + len_x / (mx - 1), dx)
t_range1 = np.arange(0, t_terminate1, dt)
t_range2 = np.arange(0, t_terminate2, dt)
t_plot1 = np.arange(0, t_terminate1, 0.01)
t_plot2 = np.arange(0, t_terminate2, 0.01)

# initial condition
def ini_condition(x):
    if 0 <= x < 0.3:
        return 0
    elif 0.3 <= x <= 0.6:
        return 1
    elif 0.6 < x:
        return 1 + 2.5 * (x - 0.6)

# boundary condition 1
def bound_condition_1(t):
    return 0, 2

# boundary condition 2
def bound_condition_2(t):
    return (0, 2+np.sin(10 * t))

# define cases
case1 = DiffSchemes('case1', dt, dx, sigma, x_range, t_range1, ini_condition, bound_condition_1)
case2 = DiffSchemes('case2', dt, dx, sigma, x_range, t_range2, ini_condition, bound_condition_2)

# FTCS
case1.ftcs(t_plot1)
case2.ftcs(t_plot2)

# BTCS
case1.btcs(t_plot1)
case2.btcs(t_plot2)


# CTCS
case1.ctcs(t_plot1)
case2.ctcs(t_plot2)











