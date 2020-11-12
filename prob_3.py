"""
3. Advection equation
This script compares the FCTS solution to the advection equation with 
the Lax-Friedrichs solution
"""
import numpy as np
import matplotlib.pyplot as plt

########## Parameters ##########
Ngrid = 50      # Number of spatial cells
Nsteps = 5000   # Number of time steps
dt = 1          # time step (in seconds)
dx = 1          # spacing of the spatial grid (in meters)

v = -0.1        # bulk flow (in meters per seconds)
################################

def fcts_advection_update(f, alpha):
    """
    FCTS: Forward-Time Central-Space is a first order finite differencing method. 
        This is the update to the advection equation 
        (without diffusion, pressure, gravity or other source 
        term).
    alpha: v * dt / (2 * dx)
    """
    f[1:-1] = f[1:-1] - alpha * (f[2:] - f[:-2])
    return f

def lf_advection_update(f, alpha):
    """
    LF: Lax-friedrichs, this is a first order, but stable, 
        method to solve the advection equation
    alpha: v * dt / (2 * dx)
    """
    f[1:-1] = 0.5 * (f[2:] + f[:-2]) - alpha * (f[2:] - f[:-2])
    return f


def main():
    alpha = v * dt / 2. / dx
    x = np.arange(Ngrid) * dx    # spatial grid

    # Initial conditions (f(t=0, x) = x)
    f1 = np.copy(x) * 1 / Ngrid  # for FCTS
    f2 = np.copy(f1)             # for Lax-Friedrichs

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title("FCTS")
    ax1.set_ylabel("f [m]")
    ax1.set_xlabel("x [m]")
    x1, = ax1.plot(x, f1, "or")

    ax2.set_title("Lax-Friedrichs")
    ax2.set_xlabel("x [m]")
    x2, = ax2.plot(x, f2, "or")

    fig.canvas.draw()

    # Simulation
    for _ in range(Nsteps):
        f1 = fcts_advection_update(f1, alpha)
        f2 = lf_advection_update(f2, alpha)

        x1.set_ydata(f1)
        x2.set_ydata(f2)
        fig.canvas.draw()
        plt.pause(0.01)

if __name__ == "__main__":
    main()

