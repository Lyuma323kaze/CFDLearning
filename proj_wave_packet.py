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
m = 20

# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range = np.arange(0, t_terminate + dt, dt)
t_plot = [0, 1, 5, 10.]
t_plot_refined = np.linspace(0, 0.1, 10)

# plot parameter
# y_lim = (0, 1.2)
y_lim = None
m_ls = [5, 10, 15, 20]

# initial condition
def ini_condition(x, m = m):
    value = 0
    for i in range(m):
        value += np.sin(2 * np.pi * (i+1) * x)
    value /= m
    return value

def analytical_solution(x, t, m = m):
    value = 0
    for i in range(m):
        value += np.sin(2 * np.pi * (i+1) * (x - t))
    value /= m
    return value

def compute_l2_loss(result, solution):
    diff = solution - result
    loss = np.linalg.norm(diff)
    return loss

item = DiffSchemes(name, dt, dx, x_range, t_range, c = c, ini_condi = ini_condition, folder = folder)

result_drp = item.drp(t_plot, y_lim, m = m)
result_drp_m = item.drp_m(t_plot, y_lim, m = m, Re_a = 20)
result_sadrp = item.sadrp(t_plot, y_lim, m = m)
result_mdcd = item.mdcd(t_plot, y_lim, m = m)

ana_solution = analytical_solution(x_range, t_terminate, m = m)

loss_drp = compute_l2_loss(result_drp, ana_solution)
loss_drp_m = compute_l2_loss(result_drp_m, ana_solution)
loss_sadrp = compute_l2_loss(result_sadrp, ana_solution)
loss_mdcd = compute_l2_loss(result_mdcd, ana_solution)
