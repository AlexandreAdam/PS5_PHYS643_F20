# PS5

name: Alexandre Adam 
python version: 3.7.6

Repo stucture
|-prob_1.py
|-prob_2.py
|-prob_3.py
|-README.md

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

