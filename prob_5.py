"""
5. 1-Dimensional Hydro Solver
This script explore the motion of sound waves through a uniform density gas 
without gravity starting from a gaussian perturbation with different amplitudes.

We use the Donor-Cell Advection framework (or finite volume element)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

########## Parameters ##########
Ngrid = 51      # Number of spatial cells (must be odd)
Nsteps = 500    # Number of time steps
dt = 1          # time step (in seconds)
dx = 1          # spacing of the spatial grid (in meters)
cs = 0.1        # speed of sound (meters per seconds)

# parameters of the perturbation
A1 = 0.1
A2 = 0.5
A3 = 2
x0 = Ngrid * dx /2
################################


def current(f, u):
    """
        u: velocity at the interface of each cells (except rightmost and leftmost interface)
        inflow: information received from (right/left) neighbor to the cell
        outflow: information transmitted to (right/left) neighbor
        J_right: Current of right interface (except boundaries)
        J_left: Current for left interface
    """
    # we add a ghost boundary for each current vector
    J_right = np.zeros(f.size)
    J_left = np.zeros(f.size)

    # Right interface of each cell
    outflow = u > 0
    inflow = u < 0
    J_right[:-1][outflow] = f[:-1][outflow] * u[outflow]
    J_right[:-1][inflow] = f[1:][inflow] * u[inflow]

    # left interface of each cell
    outflow = u < 0
    inflow = u > 0
    J_left[1:][outflow] = f[1:][outflow] * u[outflow]
    J_left[1:][inflow] = f[:-1][inflow] * u[inflow]
    return J_left, J_right
        


def donor_cell_advection(dt, dx, cs, Ngrid):
    """
    f1 = rho, density of the fluid (kilograms per meters^3)
    f2 = rho * u, matter current at the center of each cell 
    cs: sound speed (meters per seconds)
    P = rho * cs^2, pressure in the center of each cells (in Pascals)
    J: matter current at the boundary of each cells, (Ngrid+1)-vector
    """
    alpha = dt/dx
    beta = alpha * cs**2
    ri = slice(1, None, 2) # right interface 
    li = slice(0, None, 2) # left interface 
    def update(f1, f2):
        u = 0.5 * (f2[:-1] / f1[:-1] + f2[1:] / f1[1:]) 
        J1_left, J1_right = current(f1, u)
        J2_left, J2_right = current(f2, u)

        # update before source term
        f1 -= alpha * (J1_right - J1_left)
        f2 -= alpha * (J2_right - J2_left)

        # update with source term (in Euler equation)
        f2[1:-1] -= beta * (f1[2:] - f1[:-2])

        # Reflective boundary conditions
        f1[0] -= alpha * J1_right[0]
        f1[-1] += alpha * J1_left[-1]
        f2[0] -= alpha * J2_right[0]
        f2[-1] += alpha * J2_left[-1]

        return f1, f2
    return update

def main(save):
    def P(f1):
        return cs**2 * f1
    # Initialize update functions
    update = donor_cell_advection(dt, dx, cs, Ngrid)

    x = np.arange(Ngrid) * dx    # spatial grid

    # Initial conditions 
    f1 = [A * np.exp(-0.5 * (x - x0)**2) + 0.1 for A in [A1, A2, A3]]
    f2 = [np.zeros_like(x, np.float32) for _ in range(3)]

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.set_title(f"Density, Amplitude = {A1} [kg/m$^3$]")
    ax1.set_ylabel(r"$\rho$ [kg m$^{-3}$]")
    ax1.set_xlabel("x [m]")
    ax1.set_ylim(0, 1)
    x1, = ax1.plot(x, f1[0], "or")

    ax2.set_title(fr"Density, Amplitude = {A2} [kg/m$^3$]")
    ax2.set_ylabel(r"$\rho u$ [kg m$^{-2}$ s$^{-1}$]")
    ax2.set_xlabel("x [m]")
    ax2.set_ylim(0, 1)
    x2, = ax2.plot(x, f1[1], "or")

    ax3.set_title(fr"Density, Amplitude = {A3} [kg/m$^3$]")
    ax3.set_ylabel("P [Pa]")
    ax3.set_xlabel("x [m]")
    ax3.set_ylim(0, 1)
    x3, =ax3.plot(x, f1[2], "or")

    plt.subplots_adjust(left=0.1, wspace=0.5)
    fig.canvas.draw()
    lines = [x1, x2, x3]

    def update_plot(num, f1, f2, lines):
        for _f1, _f2, x in zip(f1, f2, lines):
            _f1, _f2 = update(_f1, _f2)
            x.set_ydata(_f1)
        return lines

    # Simulation
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='AlexandreAdam'), bitrate=1800)
        anim = animation.FuncAnimation(fig, update_plot, Nsteps, fargs=(f1, f2, lines ), \
                interval=50, blit=True)
        anim.save("sound_wave.mp4", writer=writer)
    else:
        for _ in range(Nsteps):
            update_plot(0, f1, f2, lines)
            fig.canvas.draw()
            plt.pause(0.01)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--save", action="store_true", help="Save the figure in mp4 format")
    args = parser.parse_args()

    main(args.save)
