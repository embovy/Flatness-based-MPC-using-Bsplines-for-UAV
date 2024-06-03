import copy
from typing import Callable, List

import casadi
import numpy as np

from tools.Integrator import Euler, rk4
from tools.Timer import Timer

__copyright__ = "Copyright 2024, Ghent University"

"""
This code was created by Emile Bovyn <emile.bovyn@ugent.be> which is largely based on the code for optimal control by Thomas Neve <thomas.neve@ugent.be>

"""

class Ocp_DMS_Ts:
    def __init__(self, 
                nx: int, 
                nu: int,
                Ts: float = None, 
                N: int = None, 
                f: Callable = None, 
                F: Callable = None, 
                M = 1,
                wind = None,
                terminal_constraint = False,
                silent = False,
                show_execution_time = True,
                store_intermediate = False,
                integrator = "euler",
                s_opts = {"max_iter": 300, "constr_viol_tol": 0.05, "acceptable_tol": 0.05, "print_level": 5, "sb": "yes"},
                threads = 1):
        """
        Optimal control abstraction for DMS, symbolic problem statement using CasADi. The goal is to define the variables (states x and actions u),
        while also incorporating the initial state X0 and goal state XN. The total cost function is defined and passed to the optimizer 
        as the quantity to be minimized. Furthermore, a customized solver is created with CasADi and giving to the optimizer.

        Args:
            nx (int): state size
            nu (int): action size
            Ts (float): time horizon, Defaults to None
            N (int): number of control intervals
            f (callable): continuous dynamics x_dot=f(x, u) which returns the derivative of the state x_dot
            F (callable): discrete dynamics x_new=F(x,u)
            M (int, optional): number of integration intervals per control interval. Defaults to 1.
            terminal_constraint (bool, optional): set the goal state as a terminal constraint. Defaults to False.
            silent (bool, optional): print optimisation steps. Defaults to False.
            show_execution_time (bool, optional): print execution time. Defaults to True.
            store_intermediate (bool, optional): store intermediate solutions of the optimisation problem. Defaults to False.
            integrator (str, optional): the integrator used in the discretization. Supports Runge-Kutta and Euler. Defaults to 'euler'.
            max_iter (int): maximum number of iterations
        """
        scaling_pos = 100
        scaling_ang = 3
        scaling_thr = 10
        scaling_del = 0.0003

        if (wind is not None):
            self.wind = np.array(wind)
        else:
            print('WIND is NOT given')
            self.wind = None

        self.opti = casadi.Opti()
        if Ts is None:
            #self.Ts = 10*self.opti.variable()
            self.Ts = self.opti.variable()
        else:
            self.Ts = Ts

        # Situational quantities
        self.nx = nx
        self.nu = nu
        self.N = N

        # Dynamcis
        if f is not None:
            self.F = self.discretize(f, M, integrator=integrator)
        elif F is not None:
            self.F = F
        else:
            raise Exception("no dynamics are provided")
        
        # Create optimization problem (X, U, X0, XN are casadi.MX)   
        #self.X = casadi.repmat(casadi.vertcat(scaling_pos, scaling_pos, scaling_pos, scaling_ang, scaling_ang, scaling_ang), 2, N+1)*self.opti.variable(self.nx, N+1)
        #self.U = casadi.repmat(casadi.vertcat(scaling_thr, scaling_thr, scaling_del, scaling_del), 1, N)*self.opti.variable(self.nu, N)

        self.X = self.opti.variable(self.nx, N+1)
        self.U = self.opti.variable(self.nu, N)

        self.X0 = self.opti.parameter(self.nx)
        self.XN = self.opti.parameter(self.nx)   #._x_ref
                
        self.params = []  # additional parameters

        # Symbolic expressions to define cost functions (x, u, xN are casadi.SX)
        self.x = casadi.SX.sym("symbolic_x", self.nx)
        self.u = casadi.SX.sym("symbolic_u", self.nu)
        self.xN = casadi.SX.sym("symbolic_xN", self.nx)    #.x_ref

        self.sideslip = self.get_sideslip(self.X)
        self.angle_of_attack = self.get_angle_of_attack(self.X)

        if f is not None:
            self.xdot = f(self.x, self.u)
        else:
            self.xdot = None

        # Make sure the consecutive states are continuous under the dynamics
        self._set_continuity(threads)

        if terminal_constraint:
            self.opti.subject_to(self.X[:, N] == self.XN)

        # Predefine the cost functions
        self.cost_fun = None
        self.terminal_cost_fun = None
        self.cost = {"run": 0, "terminal": 0, "total": 0}

        # Create customized solver that uses IPOPT
        #self.set_solver(max_iter, silent, print_time=show_execution_time)
        self.set_solver(print_time=show_execution_time, s_opts=s_opts)
        
        # Create empty lists in case the intermediate solutions need to be stored
        self.X_sol_intermediate = []
        self.U_sol_intermediate = []
        self.store_intermediate_flag = False
        if store_intermediate:
            self.store_intermediate()

        # Create the timer object 
        self.timer = Timer(verbose=False)

    def get_sideslip(self, X):
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
    
    def get_angle_of_attack(self, X):
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


    def discretize(self, f, M, integrator='euler'):
        x = casadi.SX.sym("x", self.nx)
        u = casadi.SX.sym("u", self.nu)
        dT = casadi.SX.sym("dT", 1)

        if integrator == "rk4":
            x_new = rk4(f, x, u, dT, M)
        elif integrator == "euler":
            x_new = Euler(f, x, u, dT, M)
        else:
            raise Exception("integrator not recognized")

        F = casadi.Function("F", [x, u, dT], [x_new], ["x", "u", "dT"], ["x_new"])

        return F

    def _set_continuity(self, threads: int):
        #self.opti.subject_to(self.X[:, 0] == self.X0)

        if threads == 1:
            for i in range(self.N):
                x_next = self.F(self.X[:, i], self.U[:, i], self.Ts/self.N)

                if isinstance(x_next, np.ndarray):
                    x_next = casadi.vcat(x_next.reshape(self.nx))  # convert numpy array (nx,) to casadi vector

                self.opti.subject_to(self.X[:6, i + 1] == x_next[:6])
        else:
            X_next = self.F.map(self.N, "thread", threads)(self.X[:, :-1], self.U, self.Ts/self.N)
            self.opti.subject_to(self.X[:, 1:] == X_next)
    
    def eval_cost(self, X, U, goal_state, include_terminal_cost=True):
        assert X.shape[0] == self.nx
        N = X.shape[1] - 1

        cost_accum = self.cost_fun.map(N)(X[:, :-1], U, casadi.repmat(goal_state, 1, N))
        if include_terminal_cost:
            mayer_term = self.terminal_cost_fun(X[:, -1], goal_state)
        else:
            mayer_term = 0

        return casadi.sum2(cost_accum) + mayer_term

    def set_cost(self, cost_fun: Callable[[List[float], List[float], List[float]], List[float]] = None, terminal_cost_fun: Callable = None):
        if cost_fun is not None:
            self.cost_fun = cost_fun
            L_run = 0  # cost over the horizon
            for i in range(self.N):
                L_run += cost_fun(self.X[:, i], self.U[:, i], self.XN)  # symbolic running cost
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
        cost_fun = casadi.Function( "cost_fun", [self.x, self.u, self.xN], [symbolic_cost])
        self.set_cost(cost_fun=cost_fun)

    @property
    def terminal_cost(self):
        return self.cost["terminal"]

    @terminal_cost.setter
    def terminal_cost(self, symbolic_terminal_cost):
        terminal_cost_fun = casadi.Function("terminal_cost_fun", [self.x, self.xN], [symbolic_terminal_cost])
        self.set_cost(terminal_cost_fun=terminal_cost_fun)

    def set_solver(self, print_time=True, solver="ipopt", s_opts={}):
        """
        Set the solver to be used for the trajectory optimization

        Args:
            max_iter (float): limit on number of iterations performed
            silent (bool): print to the output console
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

        def store(i):
            x = self.opti.debug.value(self.X).reshape((-1, self.N + 1))
            u = np.array(self.opti.debug.value(self.U)).reshape((-1, self.N))

            self.X_sol_intermediate.append(x)
            self.U_sol_intermediate.append(u)
        
        # The store function will be called each iteration
        self.opti.callback(store) 

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getattr__(self, name):
        return getattr(self.opti, name)

    def solve(self, init_state, goal_state, X0_guess=None, U0_guess=None, lin_interpol=False, show_exception=True):
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
            [tuple]: U_sol [nu, N], X_sol [nx, N], info
        """
        # Set both parameters X0 and XN of the control problem
        self.opti.set_value(self.X0, init_state)
        self.opti.set_value(self.XN, goal_state)

        # Assign initial guesses to the states X and actions U
        if lin_interpol and (X0_guess is None):
            print('X0_guess via linear interpolation')
            X0_guess = np.linspace(init_state, goal_state, num=self.N + 1).T
        elif lin_interpol and (X0_guess is not None):
            raise Exception("cannot pass initial state sequence and interpolate")
        
        if X0_guess is not None:
            print('X0_guess is initialised')
            self.opti.set_initial(self.X, X0_guess)
        else:
            #self.opti.set_initial(self.X, np.zeros((self.nx, self.N + 1)))
            self.opti.set_initial(self.X, 1e-5*np.ones((self.nx, self.N + 1)))
        
        if U0_guess is not None:
            print('U0_guess is initialised')
            self.opti.set_initial(self.U, U0_guess)
        else:
            self.opti.set_initial(self.U, np.ones((self.nu, self.N)))
        

        try:
            # Timer object is initialized and the CasADi solver is released on the constructed problem 
            with self.timer:
                self.sol = self.opti.solve()

            # Get the converged solutions
            self.U_sol = np.array(self.sol.value(self.U))  # array in case of scalar control
            self.X_sol = self.sol.value(self.X)
            self.Ts_sol = self.sol.value(self.Ts)
            success = self.sol.stats()["success"]
            if "t_proc_total" in self.sol.stats().keys():
                time = self.sol.stats()["t_proc_total"]
            else:
                print("using outer execution time")
                time = self.timer.time
                # time = (
                #     self.sol.stats()["t_proc_nlp_f"]
                #     + self.sol.stats()["t_proc_nlp_g"]
                #     + self.sol.stats()["t_proc_nlp_grad"]
                #     + self.sol.stats()["t_proc_nlp_grad_f"]
                #     + self.sol.stats()["t_proc_nlp_hess_l"]
                #     + self.sol.stats()["t_proc_nlp_jac_g"]
                #     + self.sol.stats()["t_wall_nlp_f"]
                #     + self.sol.stats()["t_wall_nlp_g"]
                #     + self.sol.stats()["t_wall_nlp_grad"]
                #     + self.sol.stats()["t_wall_nlp_grad_f"]
                #     + self.sol.stats()["t_wall_nlp_hess_l"]
                #     + self.sol.stats()["t_wall_nlp_jac_g"]
                # )
        except Exception as e:
            if show_exception:
                print(e)

            # Get the unconverged solutions
            self.sol = self.opti.debug
            self.U_sol = self.sol.value(self.U)
            self.X_sol = self.sol.value(self.X)
            self.Ts_sol = self.sol.value(self.Ts)
            success = False

            # Get the elapsed time
            time = self.timer.time

        info = {"success": success,
                "cost": self.sol.value(self.opti.f),
                "time": time,
                "time_hess": self.sol.stats()["t_proc_nlp_hess_l"],
                "iter_count": self.sol.stats()["iter_count"]}

        return (self.U_sol.reshape((-1, self.N)),
                self.X_sol.reshape((-1, self.N + 1)),
                self.Ts_sol,
                info)
    

    def convergence_analyis(self):
        import matplotlib.pyplot as plt

        stats = self.sol.stats()["iterations"]

        fig, ax = plt.subplots()
        ax.plot(stats["inf_pr"], label="inf_pr")
        ax.plot(stats["inf_du"], label="inf_du")
        ax.legend()
        ax.set_yscale("log")

