# Flatness-based MPC using B-splines for UAV
The use of B-splines in trajectory generation for a tailsitter drone is compared to the Direct Multiple Shooting (DMS) method. A Model Predictive Controller (MPC) is constructed based on the B-spline approach in order to exploit the flatness property of the tailsitter. This code is part of the Master's Dissertation of Emile Bovyn, Emile.Bovyn@UGent.be (2024, Ghent University). 

For each approach, a dedicated class is constructed to set up the optimal control problem using the CasADi library for nonlinear optimisation (control_DMS_Ts.py & control_Bsplines.py). DMS is used for regular trajectory optimisation (TrajOpt_DMS.py), while the flatness-based B-spline approach is used for both trajectory optimisation and MPC (TrajOpt_MPC_Bsplines.py).

## Model
The tailsitter model used has analytical expresssions for both the forward and inverse dynamics stored in 'optimal_control/model'.

## Tools
The folder 'optimal_control/tools' contains a dedicated file for spline construction (spline.py) on which the B-spline frame is built (Bspline.py). Functions like Euler and Runge-Kutta 4 integration, Dijkstra and rotation matrices are also stored in this folder.

## Flights
Some arbitrary flights in the Euclidian space are defined for simulation and/or validation purposes via analytical expressions. Refer to 'optimal_control/flights'.

## Plots
Numerical results (processing times, iterations) of experiments can be visualised by dedicated histograms.
