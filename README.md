# PS5

**Name**: Alexandre Adam 

**Python version**: 3.7.6

**Repo stucture**
- prob_1.py
- prob_2.py
- prob_3.py
- README.md

## prob_1.py (problem 1)
This script gives a comparison between the Forward-Time Central-Space (FCTS) method 
and the Lax-Friedrichs method to solve the advection equation.

## prob_2.py (problem 2)
This script solves the advection-diffusion equation by splitting the advection 
update and the diffusion update. For the advection update, we import the Lax-Friedrichs 
method defined in prob_1.py. The diffusion update is done implicitely by solving 
a linear equation. 

The script purpose is to compare the effect of different diffusion coefficients.

## prob_3.py (problem 3)
We solve the continuity equation and the Euler equation with a single source term 
(the pressure gradient). We use the Donor Cell Advection Scheme.

The script plots 3 the density profile for 3 different initial conditions. 
In each cases, we start with a static 
medium (f2(t = 0, x) = 0), and place a gaussian overdensity in the middle of the medium. The density amplitude 
differ in each plot (see title of each axes).

There are 2 things that change as we increase the amplitude of the perturbation:

1. The speed of the wave increases;
2. The shape of the wave changes: it becomes steeper where it meets the unperturbed medium 
        when the amplitude increases.

A shock wave is defined as a step function in a pressure versus time diagram. It is a 
feature of supersonic flow. This is because information cannot propagate faster than 
the speed of sound. The particles will get packed on the shock boundary, creating a 
very thin, step like, density/pressure wave.

Looking at our simulation, we notice that the edge of the wave becoming steeper is akin to 
a shock wave, altough the thickness is not of 1 cell (which we would observe if the wave 
was a step function). Notice that the first time the 2 waves recombine (after a reflexion) 
is the moment when most of the density is concentrated in a single cell (for the rightmost 
plot).

The width of the shock is mainly due to the difference in density between the pre-shock 
medium and the post-shock medium. As we increase the amplitude of the overdensity, we 
increase this difference. The edge of the wave will get steeper as it tries to smear 
out the overdensity (effect of the pressure term) while 
being limited by the speed of sound in the medium.


