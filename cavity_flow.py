from Simple import CavitySIMPLE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

# domain and computational parameters
nx, ny = 128, 128
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
cfl = 0.5
dt = cfl * min(dx, dy)  # time step
Re = 1000 # Reynolds number 
U_top = 1
alpha_u = 0.99     # velocity relaxation factor
alpha_v = 0.99
alpha_p = 1   # pressure relaxation factor
max_iter = 10000
tol = 1e-5
tune = False

# name and folder of the case
name = 'cavity_flow'
folder = 'Proj2\\SIMPLE'
if not os.path.exists(folder):
    os.makedirs(folder)
file_path = os.path.join(folder, f'{name}@Re={Re}.png')

# solver definition
cavity = CavitySIMPLE(
    name="CavityFlow",
    dt=dt,
    dx=dx,
    x=x,
    t=np.arange(0, 10, dt),
    dy=dy,
    y=y,
    Re=Re,
    U_top=U_top,
    alpha_u=alpha_u,
    alpha_v=alpha_v,  
    alpha_p=alpha_p,  
    max_iter=max_iter,
    tol=tol
)

# solve
cavity.solve(tune=tune)

# get results
u, v, p = cavity.get_center_velocity()

# print(cavity.u)

x = np.linspace(0.5/(nx), 1-0.5/(nx), nx)  # x of principle nodes
y = np.linspace(0.5/(ny), 1-0.5/(ny), ny)  # y of principle nodes
X, Y = np.meshgrid(x, y, indexing='ij')  # mesh
speed = np.sqrt(u**2 + v**2)


plt.figure(figsize=(12, 5))

# left: stream plot
plt.subplot(1, 2, 1)
# plot streamline
stream = plt.streamplot(X.T, Y.T, u.T, v.T, 
               density=3, color=speed.T, linewidth=1, arrowsize=1,cmap='jet')
plt.title('Streamlines')
cbar = plt.colorbar(stream.lines)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')  # ensure same proportion of axes

# right: pressure
plt.subplot(1, 2, 2)
# pressure contour, 20 the density
contour = plt.contourf(X, Y, p, 20, cmap='coolwarm')
plt.colorbar(contour, label='Pressure')
plt.title('Pressure Contour')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')  # ensure same proportion of axes

plt.tight_layout()
plt.savefig(file_path)