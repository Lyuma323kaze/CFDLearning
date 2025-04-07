from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name = 'shock_wave_tube'
folder = 'WorkWeek7'

# domain parameters
left_x = -0.5
right_x = 0.5

t_terminate = 0.25

# mesh parameters
mx = 80    # mesh point number
c = 0.005 # CFL number
a = 1 # convective wave speed
gamma = 1.4 # isentropic ratio
dx = (right_x - left_x) / (mx - 2)
dt = c * dx / a

# artificial viscosity parameters
k2 = 20
k4 = 5e-2

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

item = DiffSchemes(name, dt, dx, x_range, t_range1, gamma = gamma, c = c, ini_condi = ini_condition, folder = folder)

item.rusanov(t_plot)
item.jameson(t_plot, k_2=k2, k_4=k4)