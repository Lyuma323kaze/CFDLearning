from Diff_schme import DiffSchemes
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os



# name and folder of the case
name = 'mesh_wave'
folder = 'Proj1'

# domain parameters
left_x = 0
right_x = 1

t_terminate = 1


# mesh parameters
mx_ls = [3250]
mx_single = [256]
mx = 256    # mesh point number
c = 0.2 # CFL number
a = 1 # convective wave speed
dx = (right_x - left_x) / (mx - 1)
dt = c * dx / a


# set of mesh points and plot points
x_range = np.arange(left_x, right_x + dx, dx)
t_range = np.arange(0, t_terminate + 2*dt, dt)
t_plot = [0, 0.5, 1]

# plot parameter
# y_lim = (0, 1.2)
y_lim = None

def ini_condition(x, m = 64):
    np.random.seed(941)
    value = 1
    addition = 0
    epsilon = 0.1
    k0 = 24
    np.random.seed(941)
    def ek(k, k0):
        return (k/k0)**4 * np.exp(-2*(k/k0)**2)
    for k in range(m):
        psi = np.random.random()
        addition += np.sqrt(ek(k+1, k0)) * np.sin(2*np.pi*(x+psi)*(k+1))
    addition *= epsilon
    addition += value
    return addition

def accu_solution(x, t):
    return ini_condition(x - t)

def get_results(mx_ls):
    results = []
    def ini_condit(x):
        return ini_condition(x)
    for mx_ in mx_ls:
        dx_ = (right_x - left_x) / (mx_ - 1)
        dt_ = c * dx_ / a
        x_range_ = np.arange(left_x, right_x + dx_, dx_)
        t_range_ = np.arange(0, t_terminate + 2 * dt_, dt_)
        item = DiffSchemes(name, dt_, dx_, x_range_, t_range_, c=c, ini_condi=ini_condit, folder=folder)
        result_drp = item.drp(t_plot, y_lim)
        result_drp_m = item.drp_m(t_plot, y_lim, Re_a=40)
        result_sadrp = item.sadrp(t_plot, y_lim)
        result_mdcd = item.mdcd(t_plot, y_lim)
        result_subls = [result_drp, result_drp_m, result_sadrp, result_mdcd]
        results.append(result_subls)
    return results     # (len(mx_ls), 4)

def get_results_upwind(mx_ls):
    results = []
    def ini_condit(x):
        return ini_condition(x)
    for mx_ in mx_ls:
        dx_ = (right_x - left_x) / (mx_ - 1)
        dt_ = c * dx_ / a
        x_range_ = np.arange(left_x, right_x + dx_, dx_)
        t_range_ = np.arange(0, t_terminate + 2 * dt_, dt_)
        item = DiffSchemes(name, dt_, dx_, x_range_, t_range_, c=c, ini_condi=ini_condit, folder=folder)
        result_up1 = item.upwind1(t_plot, y_lim)
        result_up2 = item.upwind2(t_plot, y_lim)
        result_up3 = item.upwind3(t_plot, y_lim)
        result_subls = [result_up1, result_up2, result_up3]
        results.append(result_subls)
    return results  # (len(mx_ls), 3)

def postprocess(results, solution):
    colors = ['r', 'brown', 'b', 'g']
    markers = ['o', 'v', 'o', '^']
    labels = ['DRP', 'DRP-M', 'SA-DRP', 'MDCD']
    results = results[0]
    plt.figure(figsize=(8, 6))
    for i in range(len(results)):
        result = results[i]
        color = colors[i]
        marker = markers[i]
        label = labels[i]
        plt.plot(x_range, result, marker=marker, linestyle='-', color=color, label=label)
    plt.plot(x_range, solution, linestyle='-', color='k', label='Solution')
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend()
    plt.savefig('Proj1/Fig_mesh_wave/Result at t = 10.png')

def compute_l1_loss(result, acc_solu):
    # result.shape = (4,)
    loss = np.zeros(len(result))
    for i in range(len(result)):
        diff = result[i] - acc_solu
        N_ = len(result[0])
        loss[i] = (np.abs(diff) / N_).sum()
    return loss

def compute_all_l1_loss(results, mx_ls_):
    losses = np.zeros((len(results), len(results[0])))
    for i in range(len(results)):
        dx_ = (right_x - left_x) / (mx_ls_[i] - 1)
        x_range_ = np.arange(left_x, right_x + dx_, dx_)
        acc_solu = accu_solution(x_range_, t_terminate)
        loss = compute_l1_loss(results[i], acc_solu)
        losses[i] = loss
    return losses

def evaluate_precision(losses, mx_ls_):
    log_values = np.zeros_like(losses)
    orders = []
    for i in range(len(mx_ls_)):
        log_values[i] = np.log(losses[i])
    for j in range(len(mx_ls_)-1):
        orders.append((log_values[j+1] - log_values[j]) / np.log(mx_ls_[j] / mx_ls_[j+1]))
    return np.array(orders)

'''results_single = np.array(get_results(mx_single))
accurate_solution = accu_solution(x_range, t_terminate)
postprocess(results_single, accurate_solution)
loss_single = compute_l1_loss(results_single[0], accurate_solution)
print(loss_single)'''



recompute = True

if (not os.path.exists('Proj1/mesh_wave_results.pkl')) or recompute:
    results = get_results(mx_ls)
    with open('Proj1/mesh_wave_results.pkl', 'wb') as fw:
        pickle.dump(results, fw)
else:
    with open('Proj1/mesh_wave_results.pkl', 'rb') as fr:
        results = pickle.load(fr)


accurate_solution = accu_solution(x_range, t_terminate)

losses = compute_all_l1_loss(results, mx_ls)
print(losses)
'''orders = evaluate_precision(losses, mx_ls)
print(orders)'''

'''results_upwind =get_results_upwind(mx_ls)
losses_upwind = compute_all_l1_loss(results_upwind, mx_ls)
print(losses_upwind)
orders_upwind = evaluate_precision(losses_upwind, mx_ls)
print(orders_upwind)'''