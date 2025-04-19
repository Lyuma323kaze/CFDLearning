from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name = 'SOD_TVD'
folder = 'WorkWeek9'

# domain parameters
left_x = 0
right_x = 1

t_terminate = 0.2

# mesh parameters
mx = 200    # mesh point number
c = 0.2 # CFL number
a = 1 # convective wave speed
gamma = 1.4 # isentropic ratio
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a
epsilon = None

# artificial viscosity parameters
k2 = 3
k4 = 0.1

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range1 = np.arange(0, t_terminate, dt)
t_plot = np.arange(0, t_terminate, 0.01)

# initial condition
def ini_condition(x):
    if x <= 0.5:
        return np.array([1, 0, 1])
    else:
        return np.array([0.125, 0, 0.1])

item = DiffSchemes(name, dt, dx, x_range, t_range1, gamma = gamma, c = c, ini_condi = ini_condition, folder = folder)

item.tvd_minmod(t_plot, entropy_fix = epsilon)