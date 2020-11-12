import numpy as np
import matplotlib.pyplot as plt

Ngrid = 50
Nsteps = 5000
dt = 1
dx = 1

v = -0.1 # bulk flow
D = 0.5


def main1():
    alpha = v * dt / 2. / dx
    x = np.arange(Ngrid) * dx  # spatial grid

    f1 = np.copy(x) * 1 / Ngrid  #  FCTS
    f2 = np.copy(f1)            # Lax-Friedrichs

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].set_title("FCTS")
    x1, = axes[0].plot(x, f1, "or")
    axes[1].set_title("Lax-Friedrichs")
    x2, = axes[1].plot(x, f2, "or")
    
    fig.canvas.draw()

    for _ in range(Nsteps):
        # FCTS
        f1[1:-1] = f1[1:-1] - alpha * (f1[2:] - f1[:-2])
        # Lax-Friedrichs
        f2[1:-1] = 0.5 * (f2[2:] + f2[:-2]) - alpha * (f2[2:] - f2[:-2])

        x1.set_ydata(f1)
        x2.set_ydata(f2)
        fig.canvas.draw()
        plt.pause(0.01)

def main2(explicit=True):
    alpha = v * dt /dx
    beta = D * dt / dx**2
    A = np.eye(Ngrid) * (1 + 2 * beta) - np.eye(Ngrid, k=1) * beta 
    A -= np.eye(Ngrid, k=-1) * beta
    # fixed boundary conditions
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0

    x = np.arange(Ngrid) * dx

    T = np.zeros_like(x)
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title("Advection/diffusion")
    x, = ax.plot(x, T, "or")

    if explicit: 
        for _ in range(Nsteps):
            T[1:-1] += beta * (T[:-2] + T[2:] - 2 * T[1:-1]) # diffusion
            T[1:-1] = 0.5 * (T[:-2] + T[2:]) - 0.5 * alpha * (-T[:-2] + T[2:]) #advection

            x.set_ydata(T)
            fig.canvas.draw()
            plt.pause(0.01)
    else:
        for _ in range(Nsteps):
            T = np.linalg.solve(A, T) # implicit diffusion
            T[1:-1] = 0.5 * (T[:-2] + T[2:]) - 0.5 * alpha * (-T[:-2] + T[2:]) #advection
            x.set_ydata(T)
            fig.canvas.draw()
            plt.pause(0.01)


def main3():
    cs = 1 # sound speed
    x = np.arange(Ngrid) * dx  # spatial grid

    f1 = np.copy(x) * 1 / Ngrid  
    f2 = np.copy(f1)            
    J1 = np.zeros(Ngrid)
    J2 = np.zeros(Ngrid)

    plt.ion()
    fig, ax = plt.subplot(1, 1, figsize=(5, 5))

    for _ in range(Nsteps):

        u = 0.5 * (f2[:-1] / f1[:-1] + f2[1:] / f1[1:])
        # rightmost interface
        outflow = u[1::2] > 0  
        inflow = ~outflow
        J1[1::2][outflow] = f[::2][outflow] * u[1::2][outflow]
        J1[1::2][inflow] = f[1::2][inflow] * u[1::2][inflow]
        
        # leftmost interface
        inflow = u[::2] > 0
        outflow = ~inflow
        J1[::2][inflow] = f[::2][inflow] * u[::2][inflow] 
        J1[::2][outflow] = f[1::2][outflow] * u[::2][outflow]


        





if __name__ == "__main__":
    main1()
    # main2(False)
