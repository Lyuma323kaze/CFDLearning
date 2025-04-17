from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name = 'linear_wave_new'
folder = 'WorkWeek5'

# domain parameters
left_x = -0.5
right_x = 0.5

t_terminate = 12

# mesh parameters
mx = 100    # mesh point number
c = 0.5 # CFL number
a = 1 # convective wave speed
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a
gamma = 1.4

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range = np.arange(0, t_terminate, dt)
t_plot = np.array([0.1, 1.0, 10.0])

# initial condition
def ini_condition(x):
    if -0.5 < x < -0.25:
        return 0
    elif -0.25 <= x <= 0.25:
        return 1
    else:
        return 0

item = DiffSchemes(name, dt, dx, x_range, t_range, gamma = gamma, c = c, ini_condi = ini_condition, folder = folder)

item.lax_wendroff(t_plot)
item.warming_beam(t_plot)
item.obtained_in_work_5(t_plot)