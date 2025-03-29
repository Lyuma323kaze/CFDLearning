import numpy as np
import matplotlib.pyplot as plt
import os

# define folder
folder = 'WorkWeek6'
if not os.path.exists(folder):
    os.makedirs(folder)

# to compute z
def z_func(k, c, mu):
    k = np.asarray(k)
    c = np.asarray(c)
    c = np.atleast_1d(c)
    mu = np.asarray(mu)
    mu = np.atleast_1d(mu)
    k_reshaped = k[:, np.newaxis, np.newaxis]
    c_reshaped = c[np.newaxis, :, np.newaxis]
    mu_reshaped = mu[np.newaxis, np.newaxis, :]
    term = 4 * mu_reshaped * (1 - np.cos(k_reshaped)) ** 2 + 1j * np.sin(k_reshaped)
    return - c_reshaped * term

# give messages for stability at c
def compute_stability_certain_c(k_range, c_ls, mu, delta):
    for c in c_ls:
        plot_at_c(k_range, c, mu, delta, plot=False)
    return 0

# return $|G|$ and computed z matrix for the parameters
def compute_over_k(k_range, c, mu):
    # compute data over k at certain c
    z_all = z_func(k_range, c, mu)
    if len(np.asarray(c).shape) == 0:
        z = z_all[:, 0, 0]
    else:
        z = z_all
    G_range = 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
    coef_range = np.abs(G_range)
    return coef_range, z_all

# print the instability range, and return the two critical k values
def compute_critical_k(k_range, c, mu, delta, prnt = None):
    coef_range = compute_over_k(k_range, c, mu)[0]
    if not np.any(coef_range >= 1 + delta):
        if prnt is None: print(f"at c={c}, for all k ∈ (0, π), the solution is stable")
        return None
    else:
        valid_k = k_range[coef_range >= 1 + delta]
        if prnt is None: print(f"at c={c}, for k in range: ({valid_k.min():.6f}, {valid_k.max():.6f}), the solution is unstable")
        return valid_k.min(), valid_k.max()

# plot at c
def plot_at_c(k_range, c, mu, delta, plot = True):
    coef_range = compute_over_k(k_range, c, mu)[0]
    sym = None  # whether there's unstable region
    # determine if instability range exists
    if not np.any(coef_range >= 1 + delta):
        print(f"at c = {c}, for all k ∈ (0, π), the solution is stable")
    else:
        valid_k = k_range[coef_range >= 1 + delta]
        mask = coef_range >= 1 + delta  # 生成布尔掩码
        k_filtered = k_range[mask]
        sym = 1
        print(f"at c = {c}, for k in range: ({valid_k.min():.6f}, {valid_k.max():.6f}), the solution is unstable")

    # plot
    if plot == True:
        path = f'Stability_plot@c={c:.2f}.png'
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, coef_range, color='blue', linestyle='-', linewidth=2, label='$|G|$ vs k')
        # The critical region
        if sym is not None:
            plt.scatter(k_filtered, coef_range[mask], color='red', s=40, zorder=3,
                        label=fr'$|G| \geq$ {1 + delta}')
        plt.title(f'Stability plot at c = {c:.2f}', fontsize=12)
        plt.xlabel('k', fontsize=10)
        plt.ylabel('$|G|$', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        file_path = os.path.join(folder, path)
        plt.savefig(file_path)
        plt.close()
    return 0

# plot the instability gap length against c, return the critical c
def compute_over_c(k_range, c_range, mu, delta, plot = True):
    gap = np.zeros(len(c_range))
    first_non_zero_c = None  # 初始化第一个非零的c值
    first_non_zero_index = -1
    for i, c in enumerate(c_range):
        # 计算临界 k 范围
        minmax = compute_critical_k(k_range, c, mu, delta, prnt = True)
        if minmax is not None:
            gap[i] = minmax[1] - minmax[0]
            if first_non_zero_c is None:
                first_non_zero_c = c_range[np.where(c_range == c)[0]]
                first_non_zero_c = first_non_zero_c[0]
                first_non_zero_index = np.where(c_range == first_non_zero_c)[0]
        else:
            gap[i] = 0
    if plot == True:
        path = f'Stability_over_c.png'
        plt.plot(c_range, gap)
        if first_non_zero_c is not None:
            plt.scatter(
                first_non_zero_c,
                0,
                color="red",
                label=f"First non-zero gap at c={first_non_zero_c:.4f}"
            )
            plt.legend()

        plt.xlabel("c")
        plt.ylabel("Unstable k Range (max - min)")
        plt.title("Unstable Region Size vs. c")
        plt.grid(True)
        file_path = os.path.join(folder, path)
        plt.savefig(file_path)
        plt.close()
        print(f'The critical point is: c={first_non_zero_c:.4f}') if first_non_zero_c is not None else None
    return first_non_zero_c

# plot the critical c values against mu, and return the c array
def compute_critical_c_over_mu(k_range, c_range, mu_range, delta, plot = True):
    criticals = np.zeros(len(mu_range))
    for i, mu in enumerate(mu_range):
        nonzero_c = compute_over_c(k_range, c_range, mu, delta, plot = False)
        criticals[i] = nonzero_c
    if plot == True:
        path = f'Critical_c_over_mu.png'
        plt.plot(mu_range, criticals)

        plt.xlabel(r"$\mu$")
        plt.ylabel("Critical c values")
        plt.title(r"Critical c values vs. $\mu$")
        plt.grid(True)
        file_path = os.path.join(folder, path)
        plt.savefig(file_path)
        plt.close()
    return criticals

# parameters
mu = 0.01
mu_range = np.linspace(0, 0.5, 1000)
k_range = np.linspace(0, np.pi, 200)
delta = 0.001
c_range = np.linspace(0, 3, 1200)
c_ls = [0.1, 0.5, 1, 2, 2.5, 2.7, 2.8, 2.85, 2.9, 3]


compute_stability_certain_c(k_range, c_ls, mu, delta)
compute_over_c(c_range, c_range, mu, delta)
compute_critical_c_over_mu(k_range, c_range, mu_range, delta)