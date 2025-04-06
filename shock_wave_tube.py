from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name = 'shock_wave_tube'
folder = 'WorkWeek6'

# domain parameters
left_x = -0.5
right_x = 0.5

t_terminate = 0.25

# mesh parameters
mx = 100    # mesh point number
c = 0.5 # CFL number
a = 1 # convective wave speed
gamma = 1.4 # isentropic ratio
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range1 = np.arange(0, t_terminate, dt)
t_plot = np.arange(0, t_terminate, 0.01)

# initial condition
def ini_condition(x):
    if x <= 0:
        return np.array([1, 0.75, 1])
    else:
        return np.array([0.125, 0, 0.1])

item = DiffSchemes(name, dt, dx, x_range, t_range1, c = c, ini_condi = ini_condition, folder = folder)
