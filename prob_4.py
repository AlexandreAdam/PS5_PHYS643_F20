"""
4. Advection-diffusion equation
We solve the advection-diffusion equation and compare 2 different diffusion coefficient
"""
import numpy as np
import matplotlib.pyplot as plt
from prob_3 import lf_advection_update

########## Parameters ##########
Ngrid = 50      # Number of spatial cells
Nsteps = 5000   # Number of time steps
dt = 1          # time step (in seconds)
dx = 1          # spacing of the spatial grid (in meters)

u = -0.1        # bulk flow (meters per seconds)
D1 = 0.1        # Diffusion coefficients (meters^2 per seconds)
D2 = 2

implicit = True # wether to use the implicit method or not
################################

def diffusion_update(D, dt, dx, Ngrid, implicit=True):
    """
    D: Diffusion coefficient
    dt: time step between updates
    dx: spacing of the regular spatial grid

    """
    # Adimensional coefficient
    beta = D * dt / dx**2

    # Implicit update operator
    A = np.eye(Ngrid) * (1 + 2 * beta) - np.eye(Ngrid, k=1) * beta 
    A -= np.eye(Ngrid, k=-1) * beta

    # boundary conditions (no-slip for both)
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0

    if implicit:
        def update(f):
            return np.linalg.solve(A, f) 
    else:
        def update(f):
            f[1:-1] += beta * (f[:-2] + f[2:] - 2 * f[1:-1]) 
            return f

    return update

def main():
    x = np.arange(Ngrid) * dx   # spatial grid

    # setup update equations with the right coefficients
    d1_update = diffusion_update(D1, dt, dx, Ngrid)
    d2_update = diffusion_update(D2, dt, dx, Ngrid)
    advection_update = lf_advection_update(u, dt, dx)

    # Initial conditions
    f1 = np.copy(x)
    f2 = np.copy(x)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title(r"D = %.1f m$^2$ s$^{-1}$" % D1)
    ax1.set_ylabel("f [m]")
    ax1.set_xlabel("x [m]")
    x1, = ax1.plot(x, f1, "or")

    ax2.set_title(r"D = %.1f m$^2$ s$^{-1}$" % D2)
    ax2.set_xlabel("x [m]")
    x2, = ax2.plot(x, f2, "or")

    fig.canvas.draw()

    # Simulation
    for _ in range(Nsteps):
        f1 = d1_update(f1)
        f1 = advection_update(f1)
        f2 = d2_update(f2)
        f2 = advection_update(f2)
        x1.set_ydata(f1)
        x2.set_ydata(f2)
        fig.canvas.draw()
        plt.pause(0.01)

if __name__ == "__main__":
    main()
