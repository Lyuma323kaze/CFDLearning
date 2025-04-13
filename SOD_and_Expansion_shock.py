from Diff_schme import DiffSchemes
import numpy as np

# name and folder of the case
name1 = 'SOD'
name2 = 'Expansion_shock'
folder = 'WorkWeek8'

# domain parameters
left_x = 0
right_x = 1

t_terminate = 0.2

# mesh parameters
mx = 200    # mesh point number
c = 1e-3 # CFL number
a = 1 # convective wave speed
gamma = 1.4 # isentropic ratio
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a
epsilon = 1.8



# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range1 = np.arange(0, t_terminate, dt)
t_plot = np.arange(0, t_terminate, 0.01)

# initial condition
def ini_condition_SOD(x):
    if x < 0.5:
        return np.array([1, 0, 1])
    else:
        return np.array([0.125, 0, 0.1])

def ini_condition_Expansion_shock(x):
    if x < 0.5:
        return np.array([5, np.sqrt(1.4), 29])
    else:
        return np.array([1, 5 * np.sqrt(1.4), 1])


item_SDO = DiffSchemes(name1, dt, dx, x_range, t_range1,
                       gamma = gamma, c = c, ini_condi = ini_condition_SOD, folder = folder)
item_expansion = DiffSchemes(name2, dt, dx, x_range, t_range1,
                             gamma = gamma, c = c, ini_condi = ini_condition_Expansion_shock,
                             folder = folder)

# item_SDO.roe(t_plot)
# item_expansion.roe(t_plot)
item_expansion.roe(t_plot, entropy_fix = epsilon)