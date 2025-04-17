from Diff_schme import DiffSchemes
import numpy as np

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
m = 5

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range = np.arange(0, t_terminate + dt, dt)
t_plot = [0, 1, 5, 10.]

# plot parameter
# y_lim = (0, 1.2)
y_lim = None

# initial condition
def ini_condition(x, m = m):
    value = 0
    for i in range(m):
        value += np.sin(2 * np.pi * (i+1) * x)
    value /= m
    return value

item = DiffSchemes(name, dt, dx, x_range, t_range, c = c, ini_condi = ini_condition, folder = folder)

item.drp(t_plot, y_lim)
item.drp_m(t_plot, y_lim)
item.sadrp(t_plot, y_lim)
item.mdcd(t_plot, y_lim)