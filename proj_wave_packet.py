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
        value += np.sin(2 * np.pi * (i+1) * (x + t+0.038))  # transition fix
        # value += np.sin(2 * np.pi * (i+1) * (x + t))      # original solution
    value /= m
    return value

def accurate_postprocess(results, solution):
    colors = ['r', 'brown', 'b', 'g']
    markers = ['o', 'v', 'o', '^']
    labels = ['DRP', 'DRP-M', 'SA-DRP', 'MDCD']
    results = results[0]
    plt.figure(figsize=(8, 6))
    mask = x_range >= 0.8
    for i in range(len(results)):
        result = results[i]
        color = colors[i]
        marker = markers[i]
        label = labels[i]
        plt.plot(x_range[mask], result[mask], marker=marker, linestyle='-', color=color, label=label)
    plt.plot(x_range[mask], solution[mask], linestyle='-', color='k', label='Solution')
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend()
    plt.savefig('Proj1\Result at t = 10.png')

def compute_l2_loss(result, solution, delta_x):
    diff = (solution - result)
    loss = np.linalg.norm(diff)
    return loss

def get_results(m_ls, mx = None):
    results = []
    for m in m_ls:
        def ini_condit(x):
            return ini_condition(x, m)
        if mx is None:
            item = DiffSchemes(name, dt, dx, x_range, t_range, c=c, ini_condi=ini_condit, folder=folder)
        else:
            dx_ = (right_x - left_x) / (mx - 1)
            dt_ = c * dx / a
            item = DiffSchemes(name, dt_, dx_, x_range, t_range, c=c, ini_condi=ini_condit, folder=folder)
        result_drp = item.drp(t_plot, y_lim, m=m)
        result_drp_m = item.drp_m(t_plot, y_lim, m=m, Re_a=20)
        result_sadrp = item.sadrp(t_plot, y_lim, m=m)
        result_mdcd = item.mdcd(t_plot, y_lim, m=m)
        result_subls = [result_drp, result_drp_m, result_sadrp, result_mdcd]
        results.append(result_subls)
    return results

def plot_loss(results, solution, m_ls, mx = 1000):
    colors = ['r', 'brown', 'b', 'g']
    markers = ['o', 'v', 'o', '^']
    labels = ['DRP', 'DRP-M', 'SA-DRP', 'MDCD']
    losses = np.zeros((4, len(m_ls)))
    for i in range(len(m_ls)):
        result_ls = results[i]
        for j in range(len(result_ls)):
            loss = compute_l2_loss(result_ls[j], solution, (right_x - left_x) / (mx - 1))
            losses[j, i] = loss
    for i in range(4):
        plt.plot(m_ls, losses[i], linestyle='-',marker = markers[i], color=colors[i], label=labels[i])
    plt.xlabel("x")
    plt.ylabel("L2 loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('Proj1\Loss at t = 10.png')



results = np.array(get_results(m_ls, mx = 1000))
# result_20 = get_results(m_ls_single, mx)

ana_solution = analytical_solution(x_range, t_terminate, m = m)

# accurate_postprocess(result_20, ana_solution)

plot_loss(results, ana_solution, m_ls)