from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name = 'linear_wave'
folder = 'WorkWeek4'

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

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range1 = np.arange(0, t_terminate, dt)
t_plot = np.array([0.1, 1.0, 10.0])

# initial condition
def ini_condition(x):
    if -0.5 < x < -0.25:
        return 0
    elif -0.25 <= x <= 0.25:
        return 1
    else:
        return 0

# print(x_range)

item = DiffSchemes(name, dt, dx, x_range, t_range1, c = c, ini_condi = ini_condition, folder = folder)

# Euler implicit
item.euler_implicit(t_plot)

# One-order upwind
item.one_upwind(t_plot)