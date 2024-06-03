import copy
from typing import Callable, List

import casadi
import numpy as np
import matplotlib.pyplot as plt

from tools.Bspline import Bspline
from tools.Timer import Timer

__copyright__ = "Copyright 2024, Ghent University"

"""
This code was created by Emile Bovyn <emile.bovyn@ugent.be> which is largely based on the code for optimal control by Thomas Neve <thomas.neve@ugent.be>

"""

class Ocp_Bspline:
    def __init__(self,
                nx,
                nu,
                Ts: float,
                N: int,
                int_knots: int,
                order: int,
                invdyn: Callable,
                generation: bool,
                tracking: bool,
                flat_order,
                X_ref = None,
                wind = None,
                terminal_constraint = False,
                show_execution_time = True,
                store_intermediate = False,
                s_opts = {"max_iter": 300, "constr_viol_tol": 0.05, "acceptable_tol": 0.05, "print_level": 5, "sb": "yes"},
                threads = 1):
        """
        Optimal control abstraction for B-splines, symbolic problem statement using CasADi. The goal is to define the variables (states x and actions u),
        while also incorporating the initial state X0 and goal state XN. The total cost function is defined and passed to the optimizer 
        as the quantity to be minimized. Furthermore, a customized solver is created with CasADi and giving to the optimizer.

        Args:
            nx (int): state size
            nu (int): action size
            spline (object): spline built up with B-splines
            invdyn (callable): continuous dynamics (x, u) = g(y,dy,ddy,dddy,ddddy)
            flat_order (int): the highest order derivative of the flat output y needed for the inverse dynamics
            N (int): number of control intervals
            Ts (float): time horizon, Defaults to None
            max_iter (int): maximum number of iterations
            terminal_constraint (bool, optional): set the goal state as a terminal constraint. Defaults to False.
            silent (bool, optional): print optimisation steps. Defaults to False.
            show_execution_time (bool, optional): print execution time. Defaults to True.
            store_intermediate (bool, optional): store intermediate solutions of the optimisation problem. Defaults to False.
        """
        scaling_pos = 100
        scaling_ang = 3
        scaling_thr = 10
        scaling_del = 0.0003

        if (wind is not None):
            self.wind = np.array(wind)
        else:
            self.wind = None

        self.opti = casadi.Opti()

        self.Ts = Ts

        # Situational quantities
        self.nx = nx
        self.nu = nu
        self.N = N
        
        # Dynamcis
        self.invdyn = invdyn
        self.flat_output_size = nu

        # Kind of problem
        if generation*tracking + (generation + tracking ) == 1:
            self.generation = generation
            self.tracking = tracking
        else:
            raise Exception("Optimization not specified: path generation or tracking?")
        
        print("---------------------------")
        print("Generation :     {}".format(self.generation))
        print("Tracking :       {}".format(self.tracking))
        print("---------------------------")
        
        if tracking and (X_ref is not None):
            if X_ref.shape != (nx, N+1):
                raise Exception("Wrong shape X_ref for tracking: needed ({},{}), given {}".format(nx, N+1, X_ref.shape))
            self.X_ref = X_ref

        # Spline
        kk = np.linspace(0, Ts, int_knots)  
        spline = Bspline(kk, order, Ts=Ts, N=N, flat_order=flat_order)
        self.spline = spline
        self.flat_order = flat_order       

        # Create optimization problem (y_coeff, X0, XN are casadi.MX)
        # Note: spline.K == number of control points; each consists of 1 flat output vector
        self.y_coeff = casadi.repmat(casadi.horzcat(scaling_pos, scaling_pos, scaling_pos, scaling_ang), spline.K, 1)*self.opti.variable(spline.K, self.flat_output_size)
        #self.y_coeff = self.opti.variable(spline.K, self.flat_output_size)

        self.X0 = self.opti.parameter(self.nx) # x0_param
        self.XN = self.opti.parameter(self.nx) # xT_param

        self.T_param = self.opti.parameter()

        # Symbolic expressions to define cost functions (x, u, xN are casadi.SX)
        self.x = casadi.SX.sym("symbolic_x", self.nx)
        self.u = casadi.SX.sym("symbolic_u", self.nu)
        self.x_ref = casadi.SX.sym("symbolic_x_ref", self.nx) # xN

        self.X, self.U = self.get_state_action()
        self.sideslip = self.get_sideslip()
        self.angle_of_attack = self.get_angle_of_attack()

        #self.opti.subject_to(self.X[:, 0] == self.X0)

        #if terminal_constraint:
            #self.opti.subject_to(self.X[:, N] == self.XN)

        # Predefine the cost functions
        self.cost_fun = None
        self.terminal_cost_fun = None
        self.cost = {"run": 0, "terminal": 0, "total": 0}

        # Create customized solver that uses IPOPT
        self.set_solver(print_time=show_execution_time, s_opts=s_opts)

        # Create empty lists in case the intermediate solutions need to be stored
        self.Y_sol_intermediate = []
        self.store_intermediate_flag = False
        if store_intermediate:
            self.store_intermediate()

        # Create the timer object
        self.timer = Timer(verbose=False)
    
    def get_sideslip(self, X=None):
        if X is None:
            X, U = self.get_state_action()
        psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]
        
        if self.wind is None:
            #va_xbyb_xb = -(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)
            #va_xbyb_yb = -(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot + casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)
            sin_beta = ((casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta))*((casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot) + casadi.cos(phi)*casadi.sin(theta)*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot)) + (casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta))*(-(casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot) - casadi.cos(phi)*casadi.sin(theta)*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot)) + casadi.cos(phi)*casadi.cos(theta)*((casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot) - (casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot)))/casadi.sqrt((-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot)**2 + (-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot)**2 + (-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot)**2 + 1e-6)
            
            sideslip = casadi.arcsin(sin_beta) #+ casadi.pi*((va_xbyb_yb > 0)/2 - (va_xbyb_yb < 0)/2 + (va_xbyb_yb == 0))*(va_xbyb_xb < 0)
            return sideslip
        
        wx, wy, wz = self.wind[0], self.wind[1], self.wind[2]
        va_xbyb_xb = -(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot + (wx*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + wy*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + wz*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) - wx
        va_xbyb_yb = -(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot + (wx*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + wy*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + wz*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) - wy
        
        sin_beta_wind = ((casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta))*((casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot) + casadi.cos(phi)*casadi.sin(theta)*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot)) + (casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta))*(-(casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot) - casadi.cos(phi)*casadi.sin(theta)*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot)) + casadi.cos(phi)*casadi.cos(theta)*((casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot) - (casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot)))/casadi.sqrt((-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + xdot + (wx*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + wy*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + wz*casadi.cos(phi)*casadi.cos(theta))*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) - wx)**2 + (-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot + (wx*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + wy*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + wz*casadi.cos(phi)*casadi.cos(theta))*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) - wy)**2 + (-(xdot*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + ydot*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + zdot*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) + zdot + (wx*(casadi.cos(psi)*casadi.sin(theta) + casadi.sin(psi)*casadi.sin(phi)*casadi.cos(theta)) + wy*(casadi.sin(psi)*casadi.sin(theta) - casadi.cos(psi)*casadi.sin(phi)*casadi.cos(theta)) + wz*casadi.cos(phi)*casadi.cos(theta))*casadi.cos(phi)*casadi.cos(theta) - wz)**2 + 1e-6)
        
        sideslip = casadi.arcsin(sin_beta_wind) + casadi.pi*((va_xbyb_yb > 0)/2 - (va_xbyb_yb < 0)/2 + (va_xbyb_yb == 0))*(va_xbyb_xb < 0)
        return sideslip
    
    def get_angle_of_attack(self, X=None):
        if X is None:
            X, U = self.get_state_action()
        psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]

        if self.wind is None:
            #va_xbzb_xb = (-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot
            #va_xbzb_zb = -(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot
 
            sin_alpha = (-casadi.sin(psi)*casadi.cos(phi)*((casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot) + casadi.cos(phi)*casadi.sin(theta)*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot)) + casadi.cos(psi)*casadi.cos(phi)*(-(casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot) - casadi.cos(phi)*casadi.sin(theta)*((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot)) + casadi.sin(phi)*((casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot) - (casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot)))/casadi.sqrt(((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot)**2 + (-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot)**2 + (-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot)**2 + 1e-6)
            
            angle_of_attack = casadi.arcsin(-sin_alpha) #+ casadi.pi*((va_xbzb_zb > 0)/2 - (va_xbzb_zb < 0)/2 + (va_xbzb_zb == 0))*(va_xbzb_xb < 0)
            return angle_of_attack
        
        wx, wy, wz = self.wind[0], self.wind[1], self.wind[2]
        va_xbzb_xb = (-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot - (-wx*casadi.sin(psi)*casadi.cos(phi) + wy*casadi.cos(psi)*casadi.cos(phi) + wz*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) - wx
        va_xbzb_zb = -(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot + (-wx*casadi.sin(psi)*casadi.cos(phi) + wy*casadi.cos(psi)*casadi.cos(phi) + wz*casadi.sin(phi))*casadi.sin(phi) - wz
        
        sin_alpha_wind = (-casadi.sin(psi)*casadi.cos(phi)*((casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot) + casadi.cos(phi)*casadi.sin(theta)*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot)) + casadi.cos(psi)*casadi.cos(phi)*(-(casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot) - casadi.cos(phi)*casadi.sin(theta)*((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot)) + casadi.sin(phi)*((casadi.cos(psi)*casadi.cos(theta) - casadi.sin(psi)*casadi.sin(phi)*casadi.sin(theta))*(-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot) - (casadi.sin(psi)*casadi.cos(theta) + casadi.cos(psi)*casadi.sin(phi)*casadi.sin(theta))*((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot)))/casadi.sqrt(((-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) + xdot - (-wx*casadi.sin(psi)*casadi.cos(phi) + wy*casadi.cos(psi)*casadi.cos(phi) + wz*casadi.sin(phi))*casadi.sin(psi)*casadi.cos(phi) - wx)**2 + (-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) + ydot + (-wx*casadi.sin(psi)*casadi.cos(phi) + wy*casadi.cos(psi)*casadi.cos(phi) + wz*casadi.sin(phi))*casadi.cos(psi)*casadi.cos(phi) - wy)**2 + (-(-xdot*casadi.sin(psi)*casadi.cos(phi) + ydot*casadi.cos(psi)*casadi.cos(phi) + zdot*casadi.sin(phi))*casadi.sin(phi) + zdot + (-wx*casadi.sin(psi)*casadi.cos(phi) + wy*casadi.cos(psi)*casadi.cos(phi) + wz*casadi.sin(phi))*casadi.sin(phi) - wz)**2 + 1e-6)
        
        angle_of_attack = casadi.arcsin(-sin_alpha_wind) + casadi.pi*((va_xbzb_zb > 0)/2 - (va_xbzb_zb < 0)/2 + (va_xbzb_zb == 0))*(va_xbzb_xb < 0)
        return angle_of_attack

    def get_sideslip_TRY_OUT(self, X=None):
        if X is None:
            X, U = self.get_state_action()

        psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]
        
        if self.wind is None:
            SIN_beta_NW = casadi.sqrt(((-xdot**2 + ydot**2)*casadi.cos(psi)**2 - 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot + xdot**2 - zdot**2)*casadi.cos(phi)**2 - 2*zdot*casadi.sin(phi)*(-ydot*casadi.cos(psi) + casadi.sin(psi)*xdot)*casadi.cos(phi) + zdot**2) / casadi.sqrt((-(casadi.cos(phi)**2 - 2)*(xdot - ydot)*(xdot + ydot)*casadi.cos(psi)**2 - 2*ydot*(casadi.sin(psi)*xdot*casadi.cos(phi)**2 - casadi.sin(phi)*zdot*casadi.cos(phi) - 2*casadi.sin(psi)*xdot)*casadi.cos(psi) + (xdot**2 - zdot**2)*casadi.cos(phi)**2 - 2*casadi.sin(psi)*casadi.sin(phi)*casadi.cos(phi)*xdot*zdot - xdot**2 + ydot**2)*casadi.cos(theta)**2 - 2*(-2*casadi.sin(phi)*ydot*xdot*casadi.cos(psi)**2 + (zdot*xdot*casadi.cos(phi) + casadi.sin(psi)*casadi.sin(phi)*(xdot - ydot)*(xdot + ydot))*casadi.cos(psi) + ydot*(casadi.sin(psi)*zdot*casadi.cos(phi) + casadi.sin(phi)*xdot))*casadi.sin(theta)*casadi.cos(theta) + (-xdot**2 + ydot**2)*casadi.cos(psi)**2 - 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot + xdot**2 + zdot**2 + 1e-6)

            COS_beta_NW = ((casadi.sin(phi)*ydot*casadi.cos(psi) - casadi.sin(psi)*xdot*casadi.sin(phi) - casadi.cos(phi)*zdot)*casadi.sin(theta) + casadi.cos(theta)*(casadi.sin(psi)*ydot + xdot*casadi.cos(psi))) / casadi.sqrt((-(casadi.cos(phi)**2 - 2)*(xdot - ydot)*(xdot + ydot)*casadi.cos(psi)**2 - 2*ydot*(casadi.sin(psi)*xdot*casadi.cos(phi)**2 - casadi.sin(phi)*zdot*casadi.cos(phi) - 2*casadi.sin(psi)*xdot)*casadi.cos(psi) + (xdot**2 - zdot**2)*casadi.cos(phi)**2 - 2*casadi.sin(psi)*casadi.sin(phi)*casadi.cos(phi)*xdot*zdot - xdot**2 + ydot**2)*casadi.cos(theta)**2 - 2*(-2*casadi.sin(phi)*ydot*xdot*casadi.cos(psi)**2 + (zdot*xdot*casadi.cos(phi) + casadi.sin(psi)*casadi.sin(phi)*(xdot - ydot)*(xdot + ydot))*casadi.cos(psi) + ydot*(casadi.sin(psi)*zdot*casadi.cos(phi) + casadi.sin(phi)*xdot))*casadi.sin(theta)*casadi.cos(theta) + (-xdot**2 + ydot**2)*casadi.cos(psi)**2 - 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot + xdot**2 + zdot**2 + 1e-6)

            SIGN_choice_NW = ((ydot*casadi.cos(psi) - casadi.sin(psi)*xdot)*casadi.cos(phi) + casadi.sin(phi)*zdot)*casadi.cos(phi)*casadi.cos(theta)
            
            try:
                if SIGN_choice_NW >= 0:
                    return casadi.atan2(SIN_beta_NW, COS_beta_NW)
                else:
                    return casadi.atan2(-SIN_beta_NW, COS_beta_NW)
            except:
                return casadi.atan2(SIN_beta_NW, COS_beta_NW)

        wx, wy, wz = self.wind[0], self.wind[1], self.wind[2]

        SIN_beta = casadi.sqrt((-(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 - 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) + (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot))*casadi.cos(phi)**2 - 2*casadi.sin(phi)*(wz - zdot)*((ydot - wy)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot))*casadi.cos(phi) + (wz - zdot)**2) / casadi.sqrt((-(casadi.cos(phi)**2 - 2)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 - 2*(casadi.sin(psi)*(wx - xdot)*casadi.cos(phi)**2 - casadi.sin(phi)*(wz - zdot)*casadi.cos(phi) - 2*casadi.sin(psi)*(wx - xdot))*(wy - ydot)*casadi.cos(psi) + (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot)*casadi.cos(phi)**2 - 2*casadi.sin(phi)*casadi.sin(psi)*(wz - zdot)*(wx - xdot)*casadi.cos(phi) - (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(theta)**2 - 2*(-2*casadi.sin(phi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi)**2 + ((wz - zdot)*(wx - xdot)*casadi.cos(phi) + casadi.sin(phi)*casadi.sin(psi)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(psi) + (casadi.sin(psi)*(wz - zdot)*casadi.cos(phi) + casadi.sin(phi)*(wx - xdot))*(wy - ydot))*casadi.sin(theta)*casadi.cos(theta) - (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 - 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) + xdot**2 - 2*wx*xdot + wx**2 + (wz - zdot)**2 + 1e-6)

        COS_beta = ((-casadi.sin(phi)*(wy - ydot)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot)*casadi.sin(phi) + casadi.cos(phi)*(wz - zdot))*casadi.sin(theta) - casadi.cos(theta)*((wx - xdot)*casadi.cos(psi) + casadi.sin(psi)*(wy - ydot))) / casadi.sqrt((-(casadi.cos(phi)**2 - 2)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 - 2*(casadi.sin(psi)*(wx - xdot)*casadi.cos(phi)**2 - casadi.sin(phi)*(wz - zdot)*casadi.cos(phi) - 2*casadi.sin(psi)*(wx - xdot))*(wy - ydot)*casadi.cos(psi) + (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot)*casadi.cos(phi)**2 - 2*casadi.sin(phi)*casadi.sin(psi)*(wz - zdot)*(wx - xdot)*casadi.cos(phi) - (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(theta)**2 - 2*(-2*casadi.sin(phi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi)**2 + ((wz - zdot)*(wx - xdot)*casadi.cos(phi) + casadi.sin(phi)*casadi.sin(psi)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(psi) + (casadi.sin(psi)*(wz - zdot)*casadi.cos(phi) + casadi.sin(phi)*(wx - xdot))*(wy - ydot))*casadi.sin(theta)*casadi.cos(theta) - (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 - 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) + xdot**2 - 2*wx*xdot + wx**2 + (wz - zdot)**2 + 1e-6)

        SIGN_choice = (((ydot - wy)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot))*casadi.cos(phi) - casadi.sin(phi)*(wz - zdot))*casadi.cos(phi)*casadi.cos(theta)
        
        try:
            if SIGN_choice >= 0:
                return casadi.atan2(SIN_beta, COS_beta)
            else:
                return casadi.atan2(-SIN_beta, COS_beta)
        except:
                return casadi.atan2(SIN_beta, COS_beta)

    def get_angle_of_attack_TRY_OUT(self, X=None):
        if X is None:
            X, U = self.get_state_action()

        psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]

        if self.wind is None:
            COS_alpha_NW = ((casadi.sin(phi)*ydot*casadi.cos(psi) - casadi.sin(psi)*xdot*casadi.sin(phi) - casadi.cos(phi)*zdot)*casadi.sin(theta) + casadi.cos(theta)*(casadi.sin(psi)*ydot + xdot*casadi.cos(psi)))/casadi.sqrt(((xdot**2 - ydot**2)*casadi.cos(psi)**2 + 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot - xdot**2 + zdot**2)*casadi.cos(phi)**2 + 2*zdot*casadi.sin(phi)*(-ydot*casadi.cos(psi) + casadi.sin(psi)*xdot)*casadi.cos(phi) + xdot**2 + ydot**2)

            SIN_alpha_NW = casadi.sqrt(((casadi.cos(phi)**2 - 2)*(xdot - ydot)*(xdot + ydot)*casadi.cos(psi)**2 + 2*ydot*(casadi.sin(psi)*xdot*casadi.cos(phi)**2 - casadi.sin(phi)*zdot*casadi.cos(phi) - 2*casadi.sin(psi)*xdot)*casadi.cos(psi) + (-xdot**2 + zdot**2)*casadi.cos(phi)**2 + 2*casadi.sin(psi)*casadi.cos(phi)*casadi.sin(phi)*xdot*zdot + xdot**2 - ydot**2)*casadi.cos(theta)**2 + 2*casadi.sin(theta)*(-2*casadi.sin(phi)*ydot*xdot*casadi.cos(psi)**2 + (zdot*xdot*casadi.cos(phi) + casadi.sin(psi)*casadi.sin(phi)*(xdot - ydot)*(xdot + ydot))*casadi.cos(psi) + ydot*(casadi.sin(psi)*zdot*casadi.cos(phi) + casadi.sin(phi)*xdot))*casadi.cos(theta) + (xdot**2 - ydot**2)*casadi.cos(psi)**2 + 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot + ydot**2)/casadi.sqrt(((xdot**2 - ydot**2)*casadi.cos(psi)**2 + 2*casadi.cos(psi)*casadi.sin(psi)*xdot*ydot - xdot**2 + zdot**2)*casadi.cos(phi)**2 + 2*zdot*casadi.sin(phi)*(-ydot*casadi.cos(psi) + casadi.sin(psi)*xdot)*casadi.cos(phi) + xdot**2 + ydot**2)

            SIGN_choice_NW = -casadi.cos(psi)*casadi.cos(phi)*((casadi.sin(psi)*xdot*casadi.sin(phi) - casadi.sin(phi)*ydot*casadi.cos(psi) + casadi.cos(phi)*zdot)*casadi.cos(theta) + casadi.sin(theta)*(casadi.sin(psi)*ydot + xdot*casadi.cos(psi)))

            try:
                if SIGN_choice_NW <= 0:
                    return casadi.atan2(SIN_alpha_NW, COS_alpha_NW)
                else:
                    return casadi.atan2(-SIN_alpha_NW, COS_alpha_NW)
            except:
                return casadi.atan2(SIN_alpha_NW, COS_alpha_NW)

        wx, wy, wz = self.wind[0], self.wind[1], self.wind[2]

        SIN_alpha = casadi.sqrt(((casadi.cos(phi)**2 - 2)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 + 2*(casadi.sin(psi)*(wx - xdot)*casadi.cos(phi)**2 - casadi.sin(phi)*(wz - zdot)*casadi.cos(phi) - 2*casadi.sin(psi)*(wx - xdot))*(wy - ydot)*casadi.cos(psi) - (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot)*casadi.cos(phi)**2 + 2*casadi.sin(phi)*casadi.sin(psi)*(wz - zdot)*(wx - xdot)*casadi.cos(phi) + (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(theta)**2 + 2*(-2*casadi.sin(phi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi)**2 + ((wz - zdot)*(wx - xdot)*casadi.cos(phi) + casadi.sin(phi)*casadi.sin(psi)*(wx + wy - xdot - ydot)*(wx - wy - xdot + ydot))*casadi.cos(psi) + (casadi.sin(psi)*(wz - zdot)*casadi.cos(phi) + casadi.sin(phi)*(wx - xdot))*(wy - ydot))*casadi.sin(theta)*casadi.cos(theta) + (wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 + 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) + (wy - ydot)**2)/casadi.sqrt(((wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 + 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) - (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot))*casadi.cos(phi)**2 + 2*((ydot - wy)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot))*(wz - zdot)*casadi.sin(phi)*casadi.cos(phi) + xdot**2 - 2*wx*xdot + wx**2 + (wy - ydot)**2)

        COS_alpha = ((-casadi.sin(phi)*(wy - ydot)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot)*casadi.sin(phi) + casadi.cos(phi)*(wz - zdot))*casadi.sin(theta) - casadi.cos(theta)*((wx - xdot)*casadi.cos(psi) + casadi.sin(psi)*(wy - ydot)))/casadi.sqrt(((wx + wy - xdot - ydot)*(wx - wy - xdot + ydot)*casadi.cos(psi)**2 + 2*casadi.sin(psi)*(wy - ydot)*(wx - xdot)*casadi.cos(psi) - (wx + wz - xdot - zdot)*(wx - wz - xdot + zdot))*casadi.cos(phi)**2 + 2*((ydot - wy)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot))*(wz - zdot)*casadi.sin(phi)*casadi.cos(phi) + xdot**2 - 2*wx*xdot + wx**2 + (wy - ydot)**2)

        SIGN_choice = ((-casadi.sin(phi)*(wy - ydot)*casadi.cos(psi) + casadi.sin(psi)*(wx - xdot)*casadi.sin(phi) + casadi.cos(phi)*(wz - zdot))*casadi.cos(theta) + ((wx - xdot)*casadi.cos(psi) + casadi.sin(psi)*(wy - ydot))*casadi.sin(theta))*casadi.cos(phi)*casadi.cos(psi)

        try:
            if SIGN_choice <= 0:
                return casadi.atan2(SIN_alpha, COS_alpha)
            else:
                return casadi.atan2(-SIN_alpha, COS_alpha)
        except:
                return casadi.atan2(SIN_alpha, COS_alpha)
        
    def get_state_action(self, y_coeff=None, N=None):
        # Note: y is a list of the flat output and all relevant derivates
        y = self.get_flat_output(y_coeff=y_coeff, N=N)
        XU = self.invdyn(y)

        if isinstance(XU, tuple):
            XU = casadi.horzcat(*XU)

        X, U = casadi.horzsplit(XU, [0, XU.shape[1] - self.nu, XU.shape[1]])

        #print('Type X, U in get_state_action : {}, {}'.format(type(X), type(U)))

        return X.T, U.T
    
    def get_flat_output(self, y_coeff=None, N=None):
        if y_coeff is None:
            y_coeff = self.y_coeff

        if N is None:
            N = self.N + 1

        y = self.spline.get_flat_output(y_coeff, order=self.flat_order, N=N)
        return y

    def eval_cost(self, X, U, goal_state, include_terminal_cost=True):
        assert X.shape[0] == self.nx
        N = X.shape[1] - 1

        if self.generation:
            cost_accum = self.cost_fun.map(N)(X[:, :-1], U, casadi.repmat(goal_state, 1, N))
        else:
            cost_accum = self.cost_fun.map(N)(X[:, :-1], U, self.X_ref[:, :-1])

        if include_terminal_cost:
            mayer_term = self.terminal_cost_fun(X[:, -1], goal_state)
        else:
            mayer_term = 0

        return casadi.sum2(cost_accum) + mayer_term
    
    def set_cost(self, cost_fun: Callable[[List[float], List[float], List[float]], List[float]] = None, terminal_cost_fun: Callable = None):
        if cost_fun is not None:
            self.cost_fun = cost_fun
            L_run = 0  # cost over the horizon
            if self.generation:
                for i in range(self.N):
                    L_run += cost_fun(self.X[:, i], self.U[:, i], self.XN)  # symbolic running cost
                self.cost["run"] = L_run
            else:
                for i in range(self.N):
                    L_run += cost_fun(self.X[:, i], self.U[:, i], self.X_ref[:, i])  # symbolic running cost
                self.cost["run"] = L_run

        if terminal_cost_fun is not None:
            self.terminal_cost_fun = terminal_cost_fun  # debugging purposes
            terminal_cost = terminal_cost_fun(self.X[:, self.N], self.XN)  # symbolic terminal cost
            self.cost["terminal"] = terminal_cost

        self.cost["total"] = self.cost["run"] + self.cost["terminal"]

        # Define the total cost as the quantity to minimize in the control problem
        self.opti.minimize(self.cost["total"])

    @property
    def running_cost(self):
        return self.cost["run"]

    @running_cost.setter
    def running_cost(self, symbolic_cost):
        cost_fun = casadi.Function("cost_fun", [self.x, self.u, self.x_ref], [symbolic_cost])
        self.set_cost(cost_fun=cost_fun)

    @property
    def terminal_cost(self):
        return self.cost["terminal"]

    @terminal_cost.setter
    def terminal_cost(self, symbolic_terminal_cost):
        terminal_cost_fun = casadi.Function("terminal_cost_fun", [self.x, self.x_ref], [symbolic_terminal_cost])
        self.set_cost(terminal_cost_fun=terminal_cost_fun)

    def set_solver(self, print_time=True, solver="ipopt", s_opts={}):
        """
        Set the solver to be used for the trajectory optimization

        Args:
            max_iter (float): limit on number of iterations performed
            silent(bool): print to the output console
            print_time (bool): print the execution time. Defaults to True.
            solver (str, optional): the solver to be used. Defaults to 'ipopt'.

        Raises:
            Exception: if the chosen solver is not recognized
        """
        if solver == "ipopt":
            p_opts, s_opts = self.solver_settings(print_time=print_time, solver=solver, s_opts=s_opts)
            
            self.opti.solver("ipopt", p_opts, s_opts)
            
        else:
            raise Exception("solver option not recognized")
        

    def solver_settings(self, print_time=True, solver="ipopt", s_opts={"print_level": 5}):
        assert solver == "ipopt"

        # Print level sets the verbosity level for console output. The larger this value the more detailed is the output. 
        # The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.

        #if silent:
          #  print_level = 0
        #else:
          #  print_level = 5

        p_opts = {"print_time": print_time}  # print information about execution time (if True, also stores it in sol stats)
        s_opts =  s_opts

        return p_opts, s_opts
    
    def store_intermediate(self):
        if self.store_intermediate_flag is True:
            return  # avoid adding duplicate callbacks
        else:
            self.store_intermediate_flag = True

        # The store function will be called each iteration
        self.opti.callback(lambda i: self.Y_sol_intermediate.append(self.opti.debug.value(self.y_coeff)))
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getattr__(self, name):
        return getattr(self.opti, name)

    def solve(self, init_state, goal_state, y0_guess=None, T=1, lin_interpol=False, show_exception=True):
        """
        Solve the optimal control problem for the initial state X0 and goal state XN

        Args:
            init_state (array): initial state
            goal_state (array): goal state
            X0_guess (array, optional): the initial state sequence guess for X. Defaults to None.
            U0_guess (array, optional): the initial action sequence guess for U. Defaults to None.
            lin_interpol (bool, optional): build a linear interpolation between the initial and goal state. Defaults to False.
            show_exception (bool, optional): in case of no convergence, show the result. Defaults to True.

        Returns:
            [tuple]: Y_sol [nu, N], info
        """
        # Set both parameters X0 and XN of the control problem
        self.opti.set_value(self.X0, init_state)
        self.opti.set_value(self.XN, goal_state)

        # Assign initial guesses to the states X 
        if lin_interpol and (y0_guess is None):
            y0_guess = np.linspace(init_state[0 : self.flat_output_size], goal_state[0 : self.flat_output_size], self.spline.K)
        elif lin_interpol and (y0_guess is not None):
            raise Exception("cannot pass initial state sequence and interpolate")
        
        if y0_guess is not None:
            self.opti.set_initial(self.y_coeff, y0_guess)
            print('initial_state = ', init_state)
            print('y0_guess = ', y0_guess)
            print('goal_state = ', goal_state)
        else:
            #self.opti.set_initial(self.y_coeff, np.zeros(self.y_coeff.shape))
            self.opti.set_initial(self.y_coeff, 1e-5*np.ones(self.y_coeff.shape))

        self.opti.set_value(self.T_param, T)
        # self.opti.set_value(self.objective_weights, weights)

        try:
            # Timer object is initialized and the CasADi solver is released on the constructed problem
            with self.timer:
                self.sol = self.opti.solve()

            # Get the converged solutions
            self.Y_sol = self.sol.value(self.y_coeff)
            self.Ts_sol = self.sol.value(self.Ts)
            success = self.sol.stats()["success"]
            if "t_proc_total" in self.sol.stats().keys():
                time = self.sol.stats()["t_proc_total"]
            else:
                print("using outer execution time")
                time = self.timer.time

        except Exception as e:
            if show_exception:
                print(e)

            # Get the unconverged solutions
            self.sol = self.opti.debug
            self.Y_sol = self.sol.value(self.y_coeff)
            self.Ts_sol = self.sol.value(self.Ts)
            success = False

            # Get the elapsed time
            time = self.timer.time

        info = {"success": success,
                "cost": self.sol.value(self.opti.f),
                "time": time,
                "time_hess": self.sol.stats()["t_proc_nlp_hess_l"],
                "iter_count": self.sol.stats()["iter_count"]}

        return (self.Y_sol, self.Ts_sol,
                info)
        # return self.U_sol.reshape((self.N, -1)), self.X_sol.reshape((self.N+1, -1)), info


    def convergence_analyis(self, fontsize=14, FSaxis=16):
        import matplotlib.pyplot as plt

        stats = self.sol.stats()["iterations"]
        iters = self.sol.stats()["iter_count"]

        fig, ax = plt.subplots()
        ax.plot(stats["inf_pr"], label='Primal')
        ax.plot(stats["inf_du"], label='Dual')

        ax.set_ylabel('Infeasibility', fontsize=FSaxis)
        ax.set_xlabel('Iterations', fontsize=FSaxis)
        ax.set_xlim((0, iters))
        ax.tick_params(axis='both', which='major', labelsize=FSaxis)
        ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
        ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.legend(fontsize=fontsize)
        ax.set_yscale("log")

        return fig 
