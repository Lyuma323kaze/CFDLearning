from Diff_schme import DiffSchemes
import numpy as np
import matplotlib.pyplot as plt

# name and folder of the case
name = 'wave_packet'
folder = 'Proj1'

# domain parameters
left_x = 0
right_x = 1

t_terminate = 10


# mesh parameters
mx = 256    # mesh point number
c = 0.3 # CFL number
a = 1 # convective wave speed
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a
m = 20
m_ls = [20, 40, 60, 80, 100, 120]
m_ls_single = [20]


# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range = np.arange(0, t_terminate + dt, dt)
t_plot = [0, 1, 5, 10.]
t_plot_refined = np.linspace(0, 0.1, 10)

# plot parameter
# y_lim = (0, 1.2)
y_lim = None