# %%
import casadi as cd
import matplotlib.pyplot as plt
import numpy as np
import math 
import winsound
import os
import json

from model.Tailsitter import TailsitterModel, plot_trajectory, plot_final_vs_ref

from tools.Dijkstra_los_3D import Dijkstra_los_3D

from control_Bsplines import Ocp_Bspline

from tools.Bspline import Bspline

# %%
""" Functions """
def set_X0_XN_constraints(ocp: Ocp_Bspline, slack_X0, X0_fullyImposed, slack_XN, XN_fullyImposed, Full0=6, Partly0=4, sl0=1e-1, FullN=6, PartlyN=4, slN=1e-1):
    X = ocp.X

    print('Full0 = {}, Partly0 = {}'.format(Full0, Partly0))
    print('FullN = {}, PartlyN = {}'.format(FullN, PartlyN))

    # Initial and goal state
    if slack_X0:
        if X0_fullyImposed:
            print('X0 Fully + Slack')
            slack = sl0*np.ones(Full0)                          #np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
            ocp.opti.subject_to(ocp.bounded((ocp.X0[0:Full0] - slack), X[0:Full0, 0], (ocp.X0[0:Full0] + slack)))
        else:
            print('X0 Partly + Slack')
            slack = sl0*np.ones(Partly0)
            ocp.opti.subject_to(ocp.bounded((ocp.X0[0:Partly0] - slack), X[0:Partly0, 0], (ocp.X0[0:Partly0] + slack)))    

    else:
        if X0_fullyImposed:
            print('X0 Fully')
            ocp.opti.subject_to(X[0:Full0, 0] == ocp.X0[0:Full0])
        else:
            print('X0 Partly')
            ocp.opti.subject_to(X[0:Partly0, 0] == ocp.X0[0:Partly0])

    if slack_XN:
        if XN_fullyImposed:
            print('XN Fully + Slack')
            slack = slN*np.ones(FullN)
            ocp.opti.subject_to(ocp.bounded((ocp.XN[0:FullN] - slack), X[0:FullN, -1], (ocp.XN[0:FullN] + slack)))
        else:
            print('XN Partly + Slack')
            slack = slN*np.ones(PartlyN)
            ocp.opti.subject_to(ocp.bounded((ocp.XN[0:PartlyN] - slack), X[0:PartlyN, -1], (ocp.XN[0:PartlyN] + slack)))    
    else:
        if XN_fullyImposed:
            print('XN Fully')
            ocp.opti.subject_to(X[0:FullN, -1] == ocp.XN[0:FullN])
        else:
            print('XN Partly')
            ocp.opti.subject_to(X[0:PartlyN, -1] == ocp.XN[0:PartlyN])
       
def set_bound_variables(ocp: Ocp_Bspline, bounds_dict={}, scales={}): 
    """ 
    Add constraints:
    No limitations on position vector p, angle vector q
    
    No limitations on angular speeds assumed for ease
    (1) Limitation on thrust T 
    (2) Limitation on surface angle delta
    (3) Limitation on absolute speed v 
    (4) Limitation on angular speeeds psidot, phidot, thetadot
    (5) Limitation on sideslip beta
    (6) Limitation on angle of attack alpha

    Continuity is forced automatically by using a spline function as path parameterization!

    """
    X, U = ocp.X, ocp.U
    sideslip = ocp.sideslip
    angle_of_attack = ocp.angle_of_attack

    # Operating intervals input U
    if bounds_dict["thrust"][0]:
        print('Thrust_bound             OK')
        print('     MAX = {}, SCALE = {}'.format(bounds_dict["thrust"][1], scales["scaling_thr"]))
        #ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["thrust"][1]  / scales["scaling_thr"], U[0, :]  / scales["scaling_thr"], bounds_dict["thrust"][1]  / scales["scaling_thr"]))
        #ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["thrust"][1]  / scales["scaling_thr"], U[1, :]  / scales["scaling_thr"], bounds_dict["thrust"][1]  / scales["scaling_thr"]))

        ocp.opti.subject_to(ocp.opti.bounded(0, U[0, :], bounds_dict["thrust"][1]))
        ocp.opti.subject_to(ocp.opti.bounded(0, U[1, :], bounds_dict["thrust"][1]))

    if bounds_dict["delta"][0]:
        print('Delta_bound              OK')
        print('     MAX = {}, SCALE = {}'.format(bounds_dict["delta"][1], scales["scaling_del"]))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["delta"][1]  / scales["scaling_del"], U[2, :]  / scales["scaling_del"], bounds_dict["delta"][1]  / scales["scaling_del"]))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["delta"][1]  / scales["scaling_del"], U[3, :]  / scales["scaling_del"], bounds_dict["delta"][1]  / scales["scaling_del"]))

    # Operating intervals state X
    if bounds_dict["linear_velocity"][0]:
        print('Linear_velocity_bound    OK')
        print('     MAX = {}'.format(bounds_dict["linear_velocity"][1]))
        ocp.opti.subject_to(ocp.opti.bounded(0, X[6,:]**2 + X[7,:]**2 + X[8,:]**2, bounds_dict["linear_velocity"][1]**2))

    if bounds_dict["psidot"][0]:
        print('Psidot_bound   OK')
        print('     MAX = {}'.format(bounds_dict["psidot"][1]*180/np.pi))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["psidot"][1], X[9,:], bounds_dict["psidot"][1]))

    if bounds_dict["phidot"][0]: 
        print('Phidot_bound   OK')
        print('     MAX = {}'.format(bounds_dict["phidot"][1]*180/np.pi))   
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["phidot"][1], X[10,:], bounds_dict["phidot"][1]))

    if bounds_dict["thetadot"][0]: 
        print('Thetadot_bound   OK')
        print('     MAX = {}'.format(bounds_dict["thetadot"][1]*180/np.pi))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["thetadot"][1], X[11,:], bounds_dict["thetadot"][1]))

    # Sideslip
    if bounds_dict["sideslip"][0]:
        print('Sideslip_boud            OK')
        print('     MAX = {}'.format(bounds_dict["sideslip"][1]*180/np.pi))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["sideslip"][1], sideslip, bounds_dict["sideslip"][1]))

    # Angle of attack
    if bounds_dict["angle_of_attack"][0]:
        print('Angle_of_attack_bound    OK')
        print('     MAX = {}'.format(bounds_dict["angle_of_attack"][1]*180/np.pi))
        ocp.opti.subject_to(ocp.opti.bounded(-bounds_dict["angle_of_attack"][1], angle_of_attack, bounds_dict["angle_of_attack"][1]))

def set_obstacles(ocp: Ocp_Bspline, marge_spheres=2, marge_cylinders=2, spheres={}, cylinders={}):
    print('Obstacles initialised    OK')
    X = ocp.X 
    objects = {}

    if spheres:
        objects['spheres'] = spheres
        for obj in spheres.values():
            distance = cd.sum1((X[0:3, 1:] - obj['xyz']) ** 2)

            ocp.opti.subject_to(distance > (obj['r'] + marge_spheres) ** 2)
    else:
        objects['spheres'] = None

    if cylinders:
        objects['cylinders'] = cylinders
        for obj in cylinders.values():
            distance = cd.sum1((X[0:2, 1:] - obj['xy']) ** 2)

            ocp.opti.subject_to(distance > (obj['r'] + marge_cylinders) ** 2)
    else:
        objects['cylinders'] = None

    if spheres == {} and cylinders == {}:
        objects = None
    
    return objects

def set_cost_function(ocp: Ocp_Bspline, scaling, Q=None, R=None, P=None, regularization_weight=0, scales={}, terminal_cost=False, generation=True):
    """ 
    Create the cost function:
    Q -> state errors
    R -> input efforts
 
    x = [x, y, z, psi, phi, theta, xdot, ydot, zdot, psidot_b, phidot_b, thetadot_b].T
    u = [T1, T2, delta1, delta2].T  
    """
    if Q is None:
        if generation:
            Q = np.diag([0,0,0, 
                         0,0,0,
                         0,0,0,
                         0,0,0])
        else:
            Q = np.diag([1/2,1/2,1/2, 
                         5,5,5,
                         0,0,0,
                         0,0,0])
    if R is None:
        R = np.diag([5, 5, 
                    10, 10])
    if P is None and terminal_cost == True:
        if generation:
            P = np.diag([0,0,0, 
                         0,0,0,
                         0,0,0,
                         0,0,0])
        else:
            P = np.diag([10,10,10,
                         100,100,100,
                         0,0,0,
                         0,0,0])
    
    if scaling and scales != {}:
        print('SCALED cost function with regularization = {}'.format(regularization_weight))
        Q[0,:] = Q[0,:] / scales["scaling_pos"]*2
        Q[1,:] = Q[1,:] / scales["scaling_ang"]*2
        Q[2,:] = Q[2,:] / scales["scaling_posdot"]*2
        Q[3,:] = Q[3,:] / scales["scaling_angdot"]*2

        R[0,:] = R[0,:] / scales["scaling_thr"]*2
        R[1,:] = R[1,:] / scales["scaling_del"]*2

        flat_weight = (regularization_weight) * cd.sum1(cd.diff(ocp.y_coeff[:, 0]) ** 2 / scales["scaling_pos"]**2 + cd.diff(ocp.y_coeff[:, 1]) ** 2 / scales["scaling_pos"]**2 + cd.diff(ocp.y_coeff[:, 2]) ** 2 / scales["scaling_pos"]**2 + cd.diff(ocp.y_coeff[:, 3]) ** 2 / scales["scaling_ang"]**2) 
    else:
        print('UNSCALED cost function with regularization = {}'.format(regularization_weight))
        flat_weight = regularization_weight * cd.sum1(cd.diff(ocp.y_coeff[:, 0]) ** 2 + cd.diff(ocp.y_coeff[:, 1]) ** 2 + cd.diff(ocp.y_coeff[:, 2]) ** 2 + cd.diff(ocp.y_coeff[:, 3]) ** 2) 
 
    ocp.running_cost = ((ocp.x.T - ocp.x_ref.T)@Q@(ocp.x-ocp.x_ref) + ocp.u.T@R@ocp.u)

    if terminal_cost:
        print('Terminal cost added')
        if scaling and scales != {}:
            P[0,:] = P[0,:] / scales["scaling_pos"]*2
            P[1,:] = P[1,:] / scales["scaling_ang"]*2
            P[2,:] = P[2,:] / scales["scaling_posdot"]*2
            P[3,:] = P[3,:] / scales["scaling_angdot"]*2

        ocp.terminal_cost = (ocp.x.T - ocp.x_ref.T)@P@(ocp.x-ocp.x_ref)

    ocp.cost["total"] += flat_weight
    ocp.opti.minimize(ocp.cost["total"])



def polynomial3D(t, Ts, coeffx, coeffy, coeffz):
    """
    Function that evaluates a polynomial with given coefficients for x, y and z over a time interval [0, Ts] with t being (an array of ) timefraction(s)
    """
    if (coeffx.shape != coeffy.shape) + (coeffx.shape != coeffz.shape):
        raise Exception("Not all coefficient lists have the same dimensions: coeffx {}, coeffy {}, coeffz {}".format(coeffx.shape, coeffy.shape, coeffz.shape))
    
    terms = coeffx.shape[0]
    points = t.shape[0]
    x, y, z = np.zeros(points), np.zeros(points), np.zeros(points)
    
    for i in range(terms):
        x +=  coeffx[i]*(1-t/Ts)**(i) 
        y +=  coeffy[i]*(1-t/Ts)**(i) 
        z +=  coeffz[i]*(1-t/Ts)**(i) 

    return x, y, z

def polynomial_derivative3D(t, Ts, coeffx, coeffy, coeffz, o):
    """
    Function that evaluates the o-th derivative of a polynomial with given coefficients for x, y and z over a time interval [0, Ts] with t being (an array of ) timefraction(s)
    """
    if (coeffx.shape != coeffy.shape) + (coeffx.shape != coeffz.shape):
        raise Exception("Not all coefficient lists have the same dimensions: coeffx {}, coeffy {}, coeffz {}".format(coeffx.shape, coeffy.shape, coeffz.shape))
    if o < 0:
        raise Exception("Order o cannot be negative")

    terms = coeffx.shape[0]
    points = t.shape[0]
    x, y, z = np.zeros(points), np.zeros(points), np.zeros(points)
    
    for i in range(terms):
        if (i-o) >= 0:
            x +=  coeffx[i]*math.factorial(i)/math.factorial(i-o)*(1-t/Ts)**(i-o)*(-1/Ts)**o
            y +=  coeffy[i]*math.factorial(i)/math.factorial(i-o)*(1-t/Ts)**(i-o)*(-1/Ts)**o
            z +=  coeffz[i]*math.factorial(i)/math.factorial(i-o)*(1-t/Ts)**(i-o)*(-1/Ts)**o 
            
    return x, y, z

def fit_curve_3D(t, Ts, x, y, z, total_distance, order=None):
    """    
    Function to fit a 3D polynomial of the (N-1)-th order to a set of N datapoints given the time vector as for parameterization 
    [EXACT FIT using the Least Squares Method if order is not given]
    """ 
    t, x, y, z = np.array(t), np.array(x), np.array(y), np.array(z)

    if (x.shape != y.shape) + (x.shape != z.shape):
        raise Exception("Lists/arrays must have the same 1D shape: given x {}, y {}, z{}".format(x.shape, y.shape, z.shape))
    
    if order is None:
        order = x.shape[0]

    fit = cd.Opti()
    coeffx = fit.variable(order)
    coeffy = fit.variable(order)
    coeffz = fit.variable(order)

    X, Y, Z = polynomial3D(t, Ts, coeffx, coeffy, coeffz)

    tt = np.linspace(0, Ts, 250)
    XX, YY, ZZ = polynomial3D(tt, Ts, coeffx, coeffy, coeffz)
    XX_shift, YY_shift, ZZ_shift = cd.vertcat(cd.MX.zeros(1), XX[:-1]), cd.vertcat(cd.MX.zeros(1), YY[:-1]), cd.vertcat(cd.MX.zeros(1), ZZ[:-1])

    print('XX, XX[:-1], XX_shift : {}, {}, {}, {}'.format(XX.shape, XX[:-1].shape, cd.MX.zeros(1).shape, XX_shift.shape))

    distance_sq = cd.sum1((XX - XX_shift)**2 + (YY - YY_shift)**2 + (ZZ - ZZ_shift)**2)

    cost = cd.sum1((X-x)**2 + (Y-y)**2 + (Z-z)**2) + (distance_sq - total_distance*2)**2/total_distance**2

    fit.minimize(cost)
    p_opts, s_opts = {"print_time": True}, {"max_iter": 150}

    fit.solver("ipopt", p_opts, s_opts)
    sol = fit.solve()

    return sol.value(coeffx), sol.value(coeffy), sol.value(coeffz)

def polynomial_guess_Dijkstra_flat_output(spline: Bspline, Ts, init_state, goal_state, control_points, obstacles=None, discrWidth=1, flat_order=4, plot=False, offset=True, t_offsett=0.5, marge=1):
    """
    Initial guess flat output:
    Method 3 -> create flat output for shortest path via Dijkstra (preferable for Bsplines)
    """
    xrange = [int(init_state[0]), int(goal_state[0])]
    yrange = [int(init_state[1]), int(goal_state[1])]
    zrange = [int(init_state[2]), int(goal_state[2])]
    xrange.sort()
    yrange.sort()
    zrange.sort()

    init = [int(init_state[0]), int(init_state[1]), int(init_state[2])]
    goal = [int(goal_state[0]), int(goal_state[1]), int(goal_state[2])]

    if offset:
        startlocation   = [int(init_state[0]) + t_offsett*int(init_state[6]), int(init_state[1]) + t_offsett*int(init_state[7]), int(init_state[2]) + t_offsett*int(init_state[8])]
        endlocation     = [int(goal_state[0]) - t_offsett*int(goal_state[6]), int(goal_state[1]) - t_offsett*int(goal_state[7]), int(goal_state[2]) - t_offsett*int(goal_state[8])]
    else:
        startlocation   = init
        endlocation     = goal
    
    route, timefracroute,  minimumDistance, fixedVertices = Dijkstra_los_3D(xrange=xrange, yrange=yrange, zrange=zrange, startlocation=startlocation, endlocation=endlocation, obstacles=obstacles, discrWidth=discrWidth, marge=marge)
    
    #print('--------------------')
    #print(startlocation, endlocation)
    #print(route, timefracroute, minimumDistance, fixedVertices)
    #print('--------------------')

    route.reverse()

    print('---------------------------')
    print('Route via Dijkstra:  {}'.format(route))
    print('Minimum distance:    {}'.format(minimumDistance))
    
    
    xlist, ylist, zlist = [], [], []

    if offset:
        print('OFFSET')
        xlist.append(init[0])
        ylist.append(init[1])
        zlist.append(init[2])        

    for point in route:
        print('NO OFFSET')
        xlist.append(point[0])
        ylist.append(point[1])
        zlist.append(point[2])

    if offset:
        print('OFFSET')
        xlist.append(goal[0])
        ylist.append(goal[1])
        zlist.append(goal[2])

    if offset:
        print('OFFSET')
        timefracroute.reverse()
        time_offset = t_offsett*np.ones(len(timefracroute) + 2)
        time_offset[1:-1] += np.array(timefracroute)*(Ts - 2*t_offsett)
        time_offset[0], time_offset[-1] = 0, Ts

        timefracroute = list(time_offset/Ts)
    else:
        print('NO OFFSET')
        timefracroute.reverse()


    print('timefracroute:   {}'.format(timefracroute))
    print('xlist:   {}'.format(xlist))
    print('ylist:   {}'.format(ylist))
    print('zlist:   {}'.format(zlist))
    print('---------------------------')

    coeffx, coeffy, coeffz = fit_curve_3D(timefracroute, Ts, xlist, ylist, zlist, minimumDistance)
    
    print('coeffx : {}'.format(coeffx))
    print('coeffy : {}'.format(coeffy))
    print('coeffz : {}'.format(coeffz))

    tfracs = np.linspace(0, 1, control_points)

    print('tfracs:  {}'.format(tfracs))

    x, y, z = polynomial3D(tfracs, Ts, coeffx, coeffy, coeffz)
    dx, dy, dz = polynomial_derivative3D(tfracs, Ts, coeffx, coeffy, coeffz, 1)
    psi = np.arctan2(dy, dx)

    print('x    : {}'.format(x))
    print('y    : {}'.format(y))
    print('z    : {}'.format(z))


    #x, y, z = np.hstack(([init[0]], x, [goal[0]])), np.hstack(([init[1]], y, [goal[1]])), np.hstack(([init[2]], z, [goal[2]]))

    #print('x    : {}'.format(x))
    #print('y    : {}'.format(y))
    #print('z    : {}'.format(z))
    
    Y = np.array([x, y, z, psi]).T
    print('------------------')
    print('Y = ', Y)
    print('------------------')
    CPs = spline.compute_control_points(Y, flat_order)

    spline_eval = spline.evaluate_spline(CPs)
    x_eval = spline_eval[:,0]
    y_eval = spline_eval[:,1]
    z_eval = spline_eval[:,2]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_eval, y_eval, z_eval, c='g', marker='o')
        ax.plot(CPs[:,0], CPs[:,1], CPs[:,2], c='r', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_aspect('equal', 'box')
        ax.set_title('Initial guess flat output: Dijkstra')

        if obstacles is not None:
            N = 100
            if obstacles["spheres"] is not None:
                u = np.linspace(0, 2*np.pi, N)
                v = np.linspace(0, np.pi, N)
                for obs in obstacles["spheres"].values():
                    x = obs['xyz'][0]*np.ones(N) + obs['r']*np.outer(np.cos(u), np.sin(v))
                    y = obs['xyz'][1]*np.ones(N) + obs['r']*np.outer(np.sin(u), np.sin(v))
                    z = obs['xyz'][2]*np.ones(N) + obs['r']*np.outer(np.ones(N), np.cos(v))

                    ax.plot_surface(x, y, z)

            if obstacles["cylinders"] is not None:            
                u = np.linspace(0, 2*np.pi, N)
                for obs in obstacles["cylinders"].values():
                    z = np.linspace(obs["z"][0], obs["z"][1], N)
                    u_grid, z_grid =np.meshgrid(u, z)
                    x = obs['xy'][0] + obs['r']*np.cos(u_grid)
                    y = obs['xy'][1] + obs['r']*np.sin(u_grid)
                    
                    ax.plot_surface(x, y, z_grid)

        plt.show()

    return CPs

def polynomial_guess_flat_output(spline: Bspline, Ts, init_state, goal_state, control_points, flat_order=4, plot=False):
    """
    Initial guess flat output:
    Method 1 -> fit only the flat output (preferable for Bsplines)
    """
    x0, y0, z0, psi0, xdot0, ydot0, zdot0 = init_state[0], init_state[1], init_state[2], init_state[3], init_state[6], init_state[7], init_state[8]
    xN, yN, zN, psiN, xdotN, ydotN, zdotN = goal_state[0], goal_state[1], goal_state[2], goal_state[3], goal_state[6], goal_state[7], goal_state[8]

    xyz = lambda t, Ts, s0, sN, sdot0, sdotN: [-2*(s0-sN)-Ts*(sdot0+sdotN)]*(1-t/Ts)**3 + [3*(s0-sN)+Ts*(sdot0+2*sdotN)]*(1-t/Ts)**2 + [-Ts*sdotN]*(1-t/Ts) + sN
    psi = lambda t, Ts, angle0, angleN: [angle0-angleN]*(1-t/Ts) + angleN

    t = np.linspace(0, Ts, control_points)

    Y = np.array([xyz(t,Ts,x0,xN,xdot0,xdotN), xyz(t,Ts,y0,yN,ydot0,ydotN), xyz(t,Ts,z0,zN,zdot0,zdotN), psi(t,Ts,psi0,psiN)]).T
    CPs = spline.compute_control_points(Y, flat_order)

    spline_eval = spline.evaluate_spline(CPs)
    x_eval = spline_eval[:,0]
    y_eval = spline_eval[:,1]
    z_eval = spline_eval[:,2]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_eval, y_eval, z_eval, c='g', marker='o')
        ax.plot(CPs[:,0], CPs[:,1], CPs[:,2], c='r', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_aspect('equal', 'box')
        ax.set_title('Initial guess flat output: boundary conditions')
        plt.show()

    return CPs

def polynomial_guess_trajectory(Ts, N, init_state, goal_state, flat_order=4, plot=False):
    """
    Initial guess flat output:
    Method 2 -> fit all state variables (preferable for DMS)
    """
    x0, y0, z0, psi0, phi0, theta0, xdot0, ydot0, zdot0, psidot_b0, phidot_b0, thetadot_b0 = init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5], init_state[6], init_state[7], init_state[8], init_state[9], init_state[10], init_state[11]
    xN, yN, zN, psiN, phiN, thetaN, xdotN, ydotN, zdotN, psidot_bN, phidot_bN, thetadot_bN = goal_state[0], goal_state[1], goal_state[2], goal_state[3], goal_state[4], goal_state[5], goal_state[6], goal_state[7], goal_state[8], goal_state[9], goal_state[10], goal_state[11]

    pos = lambda t, Ts, s0, sN, sdot0, sdotN: (-2*(s0-sN)-Ts*(sdot0+sdotN))*(1-t/Ts)**3 + (3*(s0-sN)+Ts*(sdot0+2*sdotN))*(1-t/Ts)**2 + (-Ts*sdotN)*(1-t/Ts) + sN
    posdot = lambda t, Ts, s0, sN, sdot0, sdotN: 3*(-2*(s0-sN)-Ts*(sdot0+sdotN))*(1-t/Ts)**2*(-1/Ts) + 2*(3*(s0-sN)+Ts*(sdot0+2*sdotN))*(1-t/Ts)*(-1/Ts) + (-Ts*sdotN)*(-1/Ts)


    Epsi = Ts*(psidot_b0*np.sin(phi0)*np.sin(psi0) - np.sin(phi0)*phidot_b0*np.cos(psi0) + thetadot_b0*np.cos(phi0))*np.sin(theta0) + Ts*(psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.sin(thetaN) - Ts*(psidot_b0*np.cos(psi0) + np.sin(psi0)*phidot_b0)*np.cos(theta0) - Ts*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN)*np.cos(thetaN) - 2*psi0 + 2*psiN
    Ephi = -Ts*(np.cos(psi0)*phidot_b0 - np.sin(psi0)*psidot_b0)*np.cos(phi0) - Ts*(np.cos(psiN)*phidot_bN - np.sin(psiN)*psidot_bN)*np.cos(phiN) - thetadot_b0*Ts*np.sin(phi0) - thetadot_bN*Ts*np.sin(phiN) - 2*phi0 + 2*phiN
    Etheta = -Ts*(psidot_b0*np.sin(phi0)*np.sin(psi0) - np.sin(phi0)*phidot_b0*np.cos(psi0) + thetadot_b0*np.cos(phi0))*np.cos(theta0) - Ts*(psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.cos(thetaN) - Ts*(psidot_b0*np.cos(psi0) + np.sin(psi0)*phidot_b0)*np.sin(theta0) - Ts*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN)*np.sin(thetaN) - 2*theta0 + 2*thetaN

    Fpsi = -Ts*(psidot_b0*np.sin(phi0)*np.sin(psi0) - np.sin(phi0)*phidot_b0*np.cos(psi0) + thetadot_b0*np.cos(phi0))*np.sin(theta0) - 2*Ts*(psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.sin(thetaN) + Ts*(psidot_b0*np.cos(psi0) + np.sin(psi0)*phidot_b0)*np.cos(theta0) + 2*Ts*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN)*np.cos(thetaN) + 3*psi0 - 3*psiN
    Fphi = Ts*(np.cos(psi0)*phidot_b0 - np.sin(psi0)*psidot_b0)*np.cos(phi0) + 2*Ts*(np.cos(psiN)*phidot_bN - np.sin(psiN)*psidot_bN)*np.cos(phiN) + thetadot_b0*Ts*np.sin(phi0) + 2*thetadot_bN*Ts*np.sin(phiN) + 3*phi0 - 3*phiN
    Ftheta = Ts*(psidot_b0*np.sin(phi0)*np.sin(psi0) - np.sin(phi0)*phidot_b0*np.cos(psi0) + thetadot_b0*np.cos(phi0))*np.cos(theta0) + 2*Ts*(psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.cos(thetaN) + Ts*(psidot_b0*np.cos(psi0) + np.sin(psi0)*phidot_b0)*np.sin(theta0) + 2*Ts*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN)*np.sin(thetaN) + 3*theta0 - 3*thetaN

    Gpsi = ((psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.sin(thetaN) - np.cos(thetaN)*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN))*Ts
    Gphi = ((np.sin(psiN)*psidot_bN - np.cos(psiN)*phidot_bN)*np.cos(phiN) - np.sin(phiN)*thetadot_bN)*Ts
    Gtheta = -((psidot_bN*np.sin(phiN)*np.sin(psiN) - np.sin(phiN)*phidot_bN*np.cos(psiN) + thetadot_bN*np.cos(phiN))*np.cos(thetaN) + np.sin(thetaN)*(psidot_bN*np.cos(psiN) + np.sin(psiN)*phidot_bN))*Ts

    Hpsi = psiN
    Hphi = phiN
    Htheta = thetaN

    ang = lambda t, Ts, E, F, G, H: E*(1-t/Ts)**3 + F*(1-t/Ts)**2 + G*(1-t/Ts) + H
    psid_b = lambda t, Ts, psi, phi, theta: (np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-3*Epsi*(1 - t/Ts)**2/Ts - 2*Fpsi*(1 - t/Ts)/Ts - Gpsi/Ts) - np.sin(psi)*np.cos(phi)*(-3*Ephi*(1 - t/Ts)**2/Ts - 2*Fphi*(1 - t/Ts)/Ts - Gphi/Ts) + (np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta))*(-3*Etheta*(1 - t/Ts)**2/Ts - 2*Ftheta*(1 - t/Ts)/Ts - Gtheta/Ts)
    phid_b = lambda t, Ts, psi, phi, theta: (np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-3*Epsi*(1 - t/Ts)**2/Ts - 2*Fpsi*(1 - t/Ts)/Ts - Gpsi/Ts) + np.cos(psi)*np.cos(phi)*(-3*Ephi*(1 - t/Ts)**2/Ts - 2*Fphi*(1 - t/Ts)/Ts - Gphi/Ts) + (np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta))*(-3*Etheta*(1 - t/Ts)**2/Ts - 2*Ftheta*(1 - t/Ts)/Ts - Gtheta/Ts)
    thetad_b = lambda t, Ts, psi, phi, theta: -np.cos(phi)*np.sin(theta)*(-3*Epsi*(1 - t/Ts)**2/Ts - 2*Fpsi*(1 - t/Ts)/Ts - Gpsi/Ts) + np.sin(phi)*(-3*Ephi*(1 - t/Ts)**2/Ts - 2*Fphi*(1 - t/Ts)/Ts - Gphi/Ts) + np.cos(phi)*np.cos(theta)*(-3*Etheta*(1 - t/Ts)**2/Ts - 2*Ftheta*(1 - t/Ts)/Ts - Gtheta/Ts)

    t = np.linspace(0, Ts, control_points)

    x, y, z = pos(t,Ts,x0,xN,xdot0,xdotN), pos(t,Ts,y0,yN,ydot0,ydotN), pos(t,Ts,z0,zN,zdot0,zdotN)
    psi, phi, theta = ang(t,Ts,Epsi,Fpsi,Gpsi,Hpsi), ang(t,Ts,Ephi,Fphi,Gphi,Hphi), ang(t,Ts,Etheta,Ftheta,Gtheta,Htheta)
    xdot, ydot, zdot = posdot(t,Ts,x0,xN,xdot0,xdotN), posdot(t,Ts,y0,yN,ydot0,ydotN), posdot(t,Ts,z0,zN,zdot0,zdotN)
    psidot_b = psid_b(t,Ts,psi,phi,theta)
    phidot_b = phid_b(t,Ts,psi,phi,theta)
    thetadot_b = thetad_b(t,Ts,psi,phi,theta)

    X0_guess = np.array([x,y,z,psi,phi,theta,xdot,ydot,zdot,psidot_b,phidot_b,thetadot_b])

    if plot:
        plot_trajectory(X0_guess, "Initial guess states: boundary conditions", reduc=10, aspect='auto-adjusted')

    Y = np.array([x,y,z,psi]).T
    CPs = spline.compute_control_points(Y, flat_order)

    return CPs



def plot_positions(Ts, N, X, X_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Position [m]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), X[0,:], color='#0A75AD', label='$x(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[1,:], color='#CC0000', label='$y(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[2,:], color='#008000', label='$z(t)$')

    if X_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), X_guess[0,:], color='#0A75AD', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[1,:], color='#CC0000', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[2,:], color='#008000', linestyle='dashed', alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'positions'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_angles(Ts, N, X, X_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Orientation [°]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    
    ax.plot(np.linspace(0,Ts,N+1), X[4,:]*180/np.pi, color='#0A75AD', label=r'$\phi(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[5,:]*180/np.pi, color='#CC0000', label=r'$\theta(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[3,:]*180/np.pi, color='#008000', label=r'$\psi(t)$')

    if X_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), X_guess[4,:]*180/np.pi, color='#0A75AD', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[5,:]*180/np.pi, color='#CC0000', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[3,:]*180/np.pi, color='#008000', linestyle='dashed', alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'angles'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_linVelocities(Ts, N, X, X_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Linear velocity [m/s]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), X[6,:], color='#0A75AD', label=r"$\dot{x}(t)$")
    ax.plot(np.linspace(0,Ts,N+1), X[7,:], color='#CC0000', label=r"$\dot{y}(t)$")
    ax.plot(np.linspace(0,Ts,N+1), X[8,:], color='#008000', label=r"$\dot{z}(t)$")

    if X_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), X_guess[6,:], color='#0A75AD', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[7,:], color='#CC0000', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[8,:], color='#008000', linestyle='dashed', alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'linVelocities'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_angVelocities(Ts, N, X, X_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Angular velocity [rad/s]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    
    ax.plot(np.linspace(0,Ts,N+1), X[10,:], color='#0A75AD', label=r'$\dot{\phi}_{b}(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[11,:], color='#CC0000', label=r'$\dot{\theta}_{b}(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[9,:], color='#008000', label=r'$\dot{\psi}_{b}(t)$')

    if X_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), X_guess[10,:], color='#0A75AD', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[11,:], color='#CC0000', linestyle='dashed', alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), X_guess[9,:], color='#008000', linestyle='dashed', alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'angVelocities'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_wingAngles(Ts, N, U, U_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Elevon [°]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), U[2,:]*180/np.pi, color='#6A329F', marker='v', markersize=6, markevery=0.1, label=r'$\delta_{1}(t)$')
    ax.plot(np.linspace(0,Ts,N+1), U[3,:]*180/np.pi, color='#FF921F', marker='p', markersize=6, markevery=0.1, label=r'$\delta_{2}(t)$')

    if U_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), U_guess[2,:]*180/np.pi, color='#6A329F', marker='v', markersize=4, markevery=0.1, linestyle=(3, (5,5)), alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), U_guess[3,:]*180/np.pi, color='#FF921F', marker='p', markersize=4, markevery=0.1, linestyle=(0, (5,5)), alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'wingAngles'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_thrusts(Ts, N, U, U_guess=None, save=False, filename=None, directory=None, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Thrust [N]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), U[0,:], color='#6A329F', marker='v', markersize=6, markevery=0.1, label=r'$T_{1}(t)$')
    ax.plot(np.linspace(0,Ts,N+1), U[1,:], color='#FF921F', marker='p', markersize=6, markevery=0.1, label=r'$T_{2}(t)$')

    if U_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), U_guess[0,:], color='#6A329F', marker='v', markersize=4, markevery=0.1, linestyle=(3, (5,5)), alpha=0.8)
        ax.plot(np.linspace(0,Ts,N+1), U_guess[1,:], color='#FF921F', marker='p', markersize=4, markevery=0.1, linestyle=(0, (5,5)), alpha=0.8)

    ax.legend(loc=legendLoc, bbox_to_anchor=bbox, fontsize=fontsize)
    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'thrusts'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_sideslip(Ts, N, sideslip, sideslip_guess=None, save=False, filename=None, directory=None, fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'Sideslip $\beta$ [°]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), sideslip*180/np.pi, color='#008000', label=r'$\beta(t)$')

    if sideslip_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), sideslip_guess*180/np.pi, color='#008000', linestyle='dashed', alpha=0.8)

    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'sideslip'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600)
        print('Image saved: {}'.format(FN))

    plt.show()

def plot_AOA(Ts, N, AOA, AOA_guess=None, save=False, filename=None, directory=None, fontsize=16, FSaxis=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'Angle of attack $\alpha$ [°]', fontsize=FSaxis)
    ax.set_xlabel('Time [s]', fontsize=FSaxis)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_xlim((0,Ts))
    ax.grid(which='major', color='#CDCDCD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.plot(np.linspace(0,Ts,N+1), AOA*180/np.pi, color='#C80F0F', label=r'$\alpha(t)$')

    if AOA_guess is not None:
        ax.plot(np.linspace(0,Ts,N+1), AOA_guess*180/np.pi, color='#C80F0F', linestyle='dashed', alpha=0.8)

    plt.tight_layout()

    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'AOA'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600)
        print('Image saved: {}'.format(FN))

    plt.show()


#%%
""" Initializing all relevant flags and situational parameters """
# ----------    Problem   
Ts_guess = 12
N = 100        

m = 1
k = 6

MPC = True

warm_start          = True                 

flatOutput_guess    = True
fullTraject_guess   = False                 
Dijkstra_guess      = False
if Dijkstra_guess:
    discrWidth          = 5
    offset              = True    
    t_offset            = 0.5
    marge               = 2

obstacles       = False
spheresAdded    = False
marge_spheres   = 2
cylindersAdded  = False
marge_cylinders = 2

scaling     = False

slack_X0                = False
X0_fullyImposed         = True
slack_XN                = True
XN_fullyImposed         = True

regularization_weight = 5

# ----------    Savings

save_ref        = False
directory_ref   = r"_"
os.makedirs(directory_ref, exist_ok=True)

# ----------    Plots       
plot_initial_goal       = False
plot_y0_guess           = False

plot_ref_states         = False
plot_ref_inputs         = False
plot_ref_sideslip_AOA   = False

plot_ref                = True
plot_ref_convergence    = False

B, H    = 4, 9

# ----------    Trajectory  
straight_line   = False
straight_line2  = False
acrobatic_turn  = False
ascending_turn  = False
ascending_turn2 = False
ascending_turn3 = False
ClimbTurn       = False
ClimbTurn2      = True
GoUp            = False
Mountain        = False

if ClimbTurn or ClimbTurn2:
    x_ticks = np.array([0, 20, 40, 60, 80, 100, 120])
    y_ticks = np.array([-10, 0, 10, 20, 30, 40, 50, 60])
    z_ticks = np.array([-10, 0, 10, 20, 30])
    views3D = [(30,-60,0), (30,160,0)]
    views2D = [(90,-90,0), (0,90,0), (0,180,0)]

if ascending_turn or ascending_turn2 or ascending_turn3 or acrobatic_turn:
    x_ticks = np.array([0, 10, 20, 30, 40, 50, 60])
    y_ticks = np.array([0, 20, 40, 60, 80, 100])
    z_ticks = np.array([-10, 0, 10, 20, 30, 40])
    views3D = [(30,-60,0), (30,160,0)]
    views2D = [(90,180,0), (0,90,0), (0,180,0)]

# ----------    Usefull dictionaries  
scales = {"scaling_pos": 100*1e14,              
          "scaling_ang": 3*1e14,         
          "scaling_posdot": 20*1e14,       
          "scaling_angdot": 3*1e14,        
          "scaling_thr": 5*1e14,             
          "scaling_del": 0.0008*1e14,     
          }

if spheresAdded*(ClimbTurn + ClimbTurn2): 
    spheres = {'sphere_1': {'r': 3, 'xyz': [30,0,0]},
               'sphere_2': {'r': 6, 'xyz': [50,20,10]},
               'sphere_3': {'r': 5, 'xyz': [80,30,10]}} 
      
    spheres = {'sphere_1': {'r': 5, 'xyz': [30,-5,0]},   
               'sphere_2': {'r': 10, 'xyz': [70,0,0]},
               'sphere_3': {'r': 10, 'xyz': [90,30,20]}}
    
    spheres = {'sphere_1': {'r': 5, 'xyz': [30,-5,0]},   
               'sphere_2': {'r': 10, 'xyz': [70,0,0]},
               'sphere_3': {'r': 10, 'xyz': [90,30,20]}}
    
    spheres = {'sphere_1': {'r': 8, 'xyz': [30,-5,0]},   
               'sphere_2': {'r': 10, 'xyz': [70,5,0]},
               'sphere_3': {'r': 10, 'xyz': [90,15,20]},
               'sphere_4': {'r': 20, 'xyz': [50,25,10]}}
    
    spheres = {'sphere_1': {'r': 5, 'xyz': [30,-5,0]},
               'sphere_2': {'r': 10, 'xyz': [70,0,0]},
               'sphere_3': {'r': 10, 'xyz': [95,30,20]}}

elif spheresAdded*(ascending_turn + ascending_turn2 + ascending_turn3):
    spheres = {'sphere_1': {'r': 10, 'xyz': [40,60,15]},
               'sphere_2': {'r': 7, 'xyz': [20,10,5]},
               'sphere_3': {'r': 5, 'xyz': [50,20,10]}}
    
    spheres = {'sphere_1': {'r': 10, 'xyz': [40,60,15]},
               'sphere_2': {'r': 8, 'xyz': [20,10,5]},
               'sphere_3': {'r': 8, 'xyz': [50,20,5]}}

else:
    spheres = {}

if cylindersAdded*(ClimbTurn + ClimbTurn2):
    cylinders = {'cyl_1': {'r': 8, 'z': [0,30], 'xy': [80,20]}}

    cylinders = {'cyl_1': {'r': 20, 'z': [0,30], 'xy': [50,25]},
                 'cyl_2': {'r': 7, 'z': [0,30], 'xy': [0,10]},
                 'cyl_3': {'r': 7, 'z': [0,30], 'xy': [7,10]}}
    
    cylinders = {'cyl_1': {'r': 10, 'z': [-10,30], 'xy': [80,20]}}

elif cylindersAdded*(ascending_turn + ascending_turn2 + ascending_turn3):
    cylinders = {'cyl_1': {'r': 5, 'z': [-5,30], 'xy': [30,85]}}
else:
    cylinders = {}

variable_bounds = {"thrust":            [True,  30],
                   "delta":             [True, 1],
                   "linear_velocity":   [True,  30],
                   "psidot":            [False, 90*np.pi/180],
                   "phidot":            [False, 45*np.pi/180],
                   "thetadot":          [False, 45*np.pi/180],
                   "sideslip":          [True,  10*np.pi/180],
                   "angle_of_attack":   [True,  30*np.pi/180]
                   }

s_opts = { "max_iter": 150,                       # Maximum number of iterations (150 ideal)
          #"tol": 1e-5,                            # Default 1e-8      Overall optimization tolerance
          #"acceptable_tol": 5e-2,                 # Default 1e-6      Acceptable optimization tolerance
          #"constr_viol_tol": 5e-2,                # Default 1e-4      Constraint violation tolerance
          #"acceptable_constr_viol_tol": 0.1,      # Default 1e-2      Acceptable constraint violation tolerance
          #"dual_inf_tol": 1e-3,                   # Default 1e+0      Absolute tolerance on the dual infeasibility
          #"acceptable_dual_inf_tol": 1e+07,       # Default 1e+10     Acceptable tolerance for dual infeasibility
          "acceptable_iter": 1,                       # Default 15        Number of iterations to consider convergence under acceptable criteria
          "print_level": 5,                           # Level of detail in solver output, adjust as needed
          "sb": "yes",                                # Suppress IPOPT banner
          "print_user_options": "yes",                # Print s_opts
          "nlp_scaling_method": "gradient-based",     # Select the technique used for scaling the NLP
          "mu_strategy": "monotone"                   #  Update strategy for barrier parameter
        }


# %%
""" Define object """
tailsitter = TailsitterModel(N)
flat_order = 4

# %%
""" Define the spline and parameters, the control points are however still undetermined
    Ts              = time horizon
    N               = number of interval Ts is divided in 
    int_knots       = number of internal knots
    order           = highest degree of the B-splines (here: q + 2 with q = highest order derivative of flat output)
    control_points  = int_knots + order - 1
    kk              = internal knot vector
"""
int_knots = m + 1
order = k
control_points = int_knots + order - 1  
kk = np.linspace(0, 1, int_knots)  
spline = Bspline(kk, order, Ts=Ts_guess, N=N, flat_order=tailsitter.flat_order)


# %%
""" Define the optimal control problem """
ocp = Ocp_Bspline(nx=tailsitter.nx , nu=tailsitter.nu, generation=True, tracking=False, N=N, int_knots=int_knots, order=order, invdyn=tailsitter.InverseDynamicsArray, flat_order=tailsitter.flat_order, Ts=Ts_guess, store_intermediate=True, s_opts=s_opts)


# %%
""" Set initial state and goal state """
if straight_line:
    traject = 'straight_line'
    init_state = [1e-6, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6, 
                  17, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6]
    goal_state = [100, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6, 
                  17, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6]

elif straight_line2:
    traject = 'straight_line2'
    init_state = [1e-6, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6, 
                  10, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6]
    goal_state = [100, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6, 
                  15, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6]

elif ascending_turn:
    traject = 'ascending_turn'
    init_state = [1e-6, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6,  
                  17, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6]
    goal_state = [1e-6, 100, 30,    
                  180*np.pi/180, 1e-6, 1e-6, 
                  -20, 1e-6, 1e-6,    
                  1e-6, 1e-6, 1e-6]

elif ascending_turn2:
    traject = 'ascending_turn2'
    init_state = [1e-6, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6,  
                  17, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6]
    goal_state = [-10, 100, 30,    
                  180*np.pi/180, 1e-6, 1e-6, 
                  -20, 1e-6, 1e-6,    
                  1e-6, 1e-6, 1e-6]
    
elif ascending_turn3:
    traject = 'ascending_turn3'
    init_state = [1e-6, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6,  
                  17, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6]
    goal_state = [-50, 80, 30,    
                  225*np.pi/180, 1e-6, 1e-6, 
                  -10, -10, 1e-6,    
                  1e-6, 1e-6, 1e-6]
    
elif acrobatic_turn:
    traject = 'acrobatic_turn'
    init_state = [1e-6, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6,  
                  17, 1e-6, 1e-6,  
                  1e-6, 1e-6, 1e-6]
    goal_state = [1e-6, 100, 0,    
                  np.pi, 1e-6, 1e-6, 
                  -17, 1e-6, 1e-6,    
                  1e-6, 1e-6, 1e-6]

elif ClimbTurn:
    traject = 'ClimbTurn'
    init_state = [1e-6, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6,
                  17, 1e-6 ,1e-6,
                  1e-6, 1e-6, 1e-6]
    goal_state = [100, 50, 20, 
                  90*np.pi/180, 1e-6, 1e-6,
                  1e-6, 15, 4,
                  1e-6, 1e-6, 1e-6]  

elif ClimbTurn2:
    traject = 'ClimbTurn2'
    init_state = [1e-6, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6,
                  10, 1e-6 ,1e-6,
                  1e-6, 1e-6, 1e-6]
    goal_state = [100, 50, 20, 
                  90*np.pi/180, 1e-6, 1e-6,
                  1e-6, 15, 0,
                  1e-6, 1e-6, 1e-6]  
    
elif GoUp:
    traject = 'GoUp'
    init_state = [1e-6, 1e-6, 1e-6, 
                  1e-6, 1e-6, 1e-6,
                  17, 1e-6 ,1e-6,
                  1e-6, 1e-6, 1e-6]
    goal_state = [1e-6, 1e-6, 20, 
                  90*np.pi/180, 1e-6, 1e-6,
                  1e-6, 15, 4,
                  1e-6, 1e-6, 1e-6]
    
elif Mountain:
    traject = 'Mountain'
    init_state = [50, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,
                  17, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,]
    goal_state = [-50, 1e-6, 10,
                  225*np.pi/180, 1e-6, 1e-6,
                  1e-6, -12, 1e-6,
                  1e-6, 1e-6, 1e-6]

else:
    raise Exception("No initial state and goal state provided!")


situation = np.hstack([np.array(init_state).reshape(12,1), np.array(goal_state).reshape(12,1)])
if plot_initial_goal:
    plot_trajectory(situation, "Initial & Goal state", reduc=1, aspect='auto-adjusted')
  

# %%
""" Extend the ocp-problem with the necessary information """
set_X0_XN_constraints(ocp=ocp, slack_X0=slack_X0, X0_fullyImposed=X0_fullyImposed, slack_XN=slack_XN, XN_fullyImposed=XN_fullyImposed)

set_bound_variables(ocp=ocp, bounds_dict=variable_bounds, scales=scales)

set_cost_function(ocp=ocp, scaling=scaling, regularization_weight=regularization_weight, scales=scales)

if obstacles:
    objects = set_obstacles(ocp=ocp, spheres=spheres, cylinders=cylinders, marge_spheres=marge_spheres, marge_cylinders=marge_cylinders)
else:
    objects = None

# %%
""" Construct the initial guess if necessary """
if flatOutput_guess:
    guess = 'flatOutput'
    y0_guess = polynomial_guess_flat_output(spline=spline, Ts=Ts_guess, init_state=init_state, goal_state=goal_state, control_points=control_points, plot=plot_y0_guess)
elif Dijkstra_guess:
    guess = 'Dijkstra'
    y0_guess = polynomial_guess_Dijkstra_flat_output(spline=spline, Ts=Ts_guess, init_state=init_state, goal_state=goal_state, control_points=control_points, obstacles=objects, discrWidth=discrWidth, marge=marge, offset=offset, t_offsett=t_offset, plot=plot_y0_guess)
elif fullTraject_guess:
    guess = 'fullTraject'
    y0_guess = polynomial_guess_trajectory(Ts_guess, N, init_state, goal_state, plot=plot_y0_guess)
else:
    y0_guess = None


# %%
""" Solve the ocp via Bsplines and get the reference trajectory & input sequences """
if warm_start:
    if y0_guess is not None:
        Y_sol, Ts_sol, info = ocp.solve(init_state, goal_state, y0_guess=y0_guess)
    else:
        guess = 'LinInterpol'
        y0_guess = np.linspace(init_state[0 : flat_order], goal_state[0 : flat_order], control_points)
        Y_sol, Ts_sol, info = ocp.solve(init_state, goal_state, lin_interpol=True)
else:
    guess = 'Zero'
    Y_sol, Ts_sol, info = ocp.solve(init_state, goal_state)

if y0_guess is not None:
    X_guess, U_guess = ocp.get_state_action(y_coeff=y0_guess)
    sideslip_guess = ocp.get_sideslip(X_guess)
    angle_of_attack_guess = ocp.get_angle_of_attack(X_guess)

X_ref, U_ref = ocp.get_state_action(y_coeff=Y_sol)
sideslip_ref = ocp.get_sideslip(X_ref)
angle_of_attack_ref = ocp.get_angle_of_attack(X_ref)

# %%
""" Data """
print('------------------------------------------------')
print('B-splines:   m + 1   = {}    (internal knots)'.format(int_knots)) 
print('             n + 1   = {}    (control points)'.format(control_points))
print('             k       = {}    (degree)'.format(order))
print(' ')
print('Warmstart                = {}'.format(warm_start))
print('y0_guess                 = {}'.format(guess))
print('Success                  = {}'.format(info["success"]))
print('Intermediate solutions   = {}'.format(np.shape(ocp.Y_sol_intermediate)))
print('Iterations               = {}'.format(info['iter_count']))
print('Process Time             = {}'.format(info['time']))
print('Ts_sol                   = {}'.format(Ts_sol))
print('------------------------------------------------')

success_ref  = info["success"]
iters_ref    = info['iter_count']
time_ref     = info['time']


# %%
""" Saving situation in dictionary"""
filename_ref    = traject + '_' + guess + '__N_{}__Ts_{}_____(m,k)_({},{})'.format(N, int(Ts_guess), m, k)
dictname_ref    = traject + '_' + guess

DICT = {'traject': traject,
        'guess': guess,
        'y0_guess': y0_guess.full().tolist(),
        'X_guess': X_guess.full().tolist(),
        'U_guess': U_guess.full().tolist(),
        'sideslip_guess': sideslip_guess.full().tolist(),
        'angle_of_attack_guess': angle_of_attack_guess.full().tolist(),
        'Y_sol': Y_sol.tolist(),
        'X_ref': X_ref.full().tolist(),
        'U_ref': U_ref.full().tolist(),
        'sideslip_ref': sideslip_ref.full().tolist(),
        'angle_of_attack_ref': angle_of_attack_ref.full().tolist(),
        'int_knots': int_knots,
        'degree': order,
        'control_points': control_points,
        'info': info
        }

if save_ref:
    with open(os.path.join(directory_ref, dictname_ref), 'w') as file:
        json.dump(DICT, file, indent=4) 


# %%
""" Plotting reference quantities """
if y0_guess is not None:
    X_guess = np.array(X_guess)
    U_guess = np.array(U_guess)
    sideslip_guess = np.array(sideslip_guess).reshape(N+1,-1)
    angle_of_attack_guess = np.array(angle_of_attack_guess).reshape(N+1,-1)
else:
    X_guess = None
    U_guess = None
    sideslip_guess = None
    angle_of_attack_guess = None

X_ref = np.array(X_ref)
U_ref = np.array(U_ref)
sideslip_ref = np.array(sideslip_ref).reshape(N+1,-1)
angle_of_attack_ref = np.array(angle_of_attack_ref).reshape(N+1,-1)

# %%
if plot_ref:
    plot_final_vs_ref(X_ref, traject=traject, x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks, obstacles=objects, control_points=Y_sol, views3D=views3D, views2D=views2D, pos=True, orient=True, labels=True, reduc=10, aspect='auto-adjusted', B=B, H=H, save=save_ref, filename=filename_ref, directory=directory_ref)

# %%
if plot_ref_states:
    plot_positions(Ts_sol, N, X_ref, X_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_angles(Ts_sol, N, X_ref, X_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_linVelocities(Ts_sol, N, X_ref, X_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_angVelocities(Ts_sol, N, X_ref, X_guess, save=save_ref, filename=filename_ref, directory=directory_ref)

if plot_ref_inputs:
    plot_wingAngles(Ts_sol, N, U_ref, U_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_thrusts(Ts_sol, N, U_ref, U_guess, save=save_ref, filename=filename_ref, directory=directory_ref)

if plot_ref_sideslip_AOA:
    plot_sideslip(Ts_sol, N, sideslip_ref, sideslip_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_AOA(Ts_sol, N, angle_of_attack_ref, angle_of_attack_guess, save=save_ref, filename=filename_ref, directory=directory_ref)

# %%
if plot_ref_convergence:
    ocp.convergence_analyis()
    plt.tight_layout()

    if save_ref:
        if directory_ref.strip() == "":
            directory = os.getcwd()
        FN = filename_ref + '_' + 'convergence' + '_processTime_' + str(int(info['time'])) + '_iters_' + str(info['iter_count'])
        full_path = os.path.join(directory_ref, FN)
        plt.savefig(full_path, dpi=600)
        print('Image saved: {}'.format(FN))

    plt.show()


# %%
""" MPC     -   Settings"""
MPC_generation  = True
MPC_tracking    = False
if MPC_generation:
    kind = 'GEN'
elif MPC_tracking:
    kind = 'TRACK'
else:
    raise Exception('Please state generation or tracking for MPC.')

slack_X0                = False
X0_fullyImposed         = True
Full0                   = 12

slack_XN                = True
XN_fullyImposed         = True
FullN                   = 6

perturbation    = True
if perturbation:
    sigma_pos       = 2         #2             # 1
    sigma_ang       = 0.30      #0.30          # 0.15
else:
    sigma_pos       = 0             
    sigma_ang       = 0           

WH_ref = 10     # Waiting horizon 
PH_ref = 30     # Prediction horizon

warm_starts = True
fixed_PH    = False
fixed_CPs   = True

saveplots   = False

plot_sol_states         = True
plot_sol_inputs         = True
plot_sol_sideslip_AOA   = True
plot_sol                = True

if saveplots:
    directory_sol   = r"_"
    os.makedirs(directory_sol, exist_ok=True)

filename_sol    = traject + '_' + str(kind) + '__(m,k)_({},{})__(W,P)_({},{})'.format(m, k, WH_ref, PH_ref, sigma_pos, sigma_ang)
dictname_sol    = traject + '_' + str(kind)


# %% 
""" MPC     -   Defining reference as starting point"""
int_knots = m + 1
order = k
control_points = int_knots + order - 1  
kk = np.linspace(0, 1, int_knots)  

Ts_ref = Ts_guess
N_ref = N
dt_ref = Ts_ref/N_ref

int_knots_ref = int_knots
order_ref = order
CPs_ref = control_points

x0 = np.array(init_state)
xN = np.array(goal_state)

# %%
""" MPC     -   Controller parameters"""
memory = {"control_points": [],
          "state_trajectory": None,
          "input_trajectory": None,
          "sideslip": None,
          "angle_of_attack": None,
          "computation_time": [],
          "iter_count": [],
          "success": [],
          "jac_scales": [],
          "t_horizon": [],
          "t": [],
          "ocp.spline.K": [],
          "y0_guess": [],
          "y_horizon": [],
          "x_horizon": [],
          "x_WH": [],
          "CPs": [],
          "PH": [],
          "current_VS_horizon_OK?": [],
          }


s_opts_MPC = { "max_iter": 150,                        # Maximum number of iterations
                #"tol": 1e-5,                            # Default 1e-8      Overall optimization tolerance
                #"acceptable_tol": 5e-2,                 # Default 1e-6      Acceptable optimization tolerance
                #"constr_viol_tol": 5e-2,                # Default 1e-4      Constraint violation tolerance
                #"acceptable_constr_viol_tol": 0.1,      # Default 1e-2      Acceptable constraint violation tolerance
                #"dual_inf_tol": 1e-3,                   # Default 1e+0      Absolute tolerance on the dual infeasibility
                #"acceptable_dual_inf_tol": 1e+07,       # Default 1e+10     Acceptable tolerance for dual infeasibility
                "acceptable_iter": 1,                       # Default 15        Number of iterations to consider convergence under acceptable criteria
                "print_level": 5,                           # Level of detail in solver output, adjust as needed
                "sb": "yes",                                # Suppress IPOPT banner
                "print_user_options": "yes",                # Print s_opts
                "nlp_scaling_method": "gradient-based",     # Select the technique used for scaling the NLP
                "mu_strategy": "monotone"                   #  Update strategy for barrier parameter
                }


# %%
""" MPC     -   Functions the get Prediction Horizon (PH) and amount of Control Points to be used (CPs)"""
def get_amount_CPs(i, N_ref, CPs_ref, fixed_PH, fixed_CPs):
    """
    Note: 5 is the minimal amount to avoid getting error with m being 1
    """
    interval_length = math.floor(N_ref/(CPs_ref - 4))
    if fixed_CPs:
        return CPs_ref
    elif fixed_PH:
        return 5
    else:
        return max(CPs_ref - math.floor(i / interval_length), 5)

def get_PH(i, N_ref, fixed_PH=False):
    if fixed_PH and (PH_ref < N_ref - i):
        return PH_ref
    else:
        return (N_ref - i)
    

# %%
""" MPC     -   Control algorithm"""
if MPC:
    if MPC_generation:
        current_state = x0.reshape(12,)

        for i in range(N_ref):
            if i % WH_ref == 0 and i != 0:

                N_i = N_ref - i

                PH = get_PH(i, N_ref, fixed_PH)
                t_horizon = PH*dt_ref                     

                OK = np.sum(X_ref[:,i] - current_state)

                CPs = get_amount_CPs(i, N_ref, CPs_ref, fixed_PH, fixed_CPs)
                order = CPs - int_knots + 1 

                if i <= N_ref - PH:
                    horizon_state = X_ref[:,i + PH]
                else:
                    horizon_state = X_ref[:,-1]

                TS = TailsitterModel(N_i)
                ocp_i = Ocp_Bspline(nx=TS.nx , nu=TS.nu, generation=True, tracking=False, N=N_i, int_knots=int_knots, order=order, invdyn=TS.InverseDynamicsArray, flat_order=TS.flat_order, Ts=t_horizon, store_intermediate=True, s_opts=s_opts_MPC)
                
                set_X0_XN_constraints(ocp=ocp_i, slack_X0=slack_X0, X0_fullyImposed=X0_fullyImposed, slack_XN=slack_XN, XN_fullyImposed=XN_fullyImposed, Full0=Full0, FullN=FullN)
                set_bound_variables(ocp=ocp_i, bounds_dict=variable_bounds, scales=scales)
                if obstacles:
                    objects = set_obstacles(ocp=ocp_i, spheres=spheres, cylinders=cylinders)
                else:
                    objects = None
                set_cost_function(ocp=ocp_i, scaling=scaling, regularization_weight=regularization_weight, scales=scales)


                if warm_starts:
                    kk = np.linspace(0, 1, int_knots)
                    spline = Bspline(kk, order, Ts=t_horizon, N=PH, flat_order=TS.flat_order)
                    y0_guess = polynomial_guess_flat_output(spline=spline, Ts=t_horizon, init_state=current_state, goal_state=horizon_state, control_points=CPs)

                    y_horizon, t, info = ocp_i.solve(current_state, horizon_state, y0_guess=y0_guess)

                else:
                    y_horizon, t, info = ocp_i.solve(current_state, horizon_state, lin_interpol=True)

                x_horizon, u_horizon = ocp_i.get_state_action(y_coeff=y_horizon)
                x_horizon, u_horizon = np.array(x_horizon), np.array(u_horizon)

                if (PH > WH_ref):
                    LAST = False
                    x_WH = x_horizon[:,:WH_ref]       
                    u_WH = u_horizon[:,:WH_ref]    

                    if perturbation:
                        x_horizon[:6,WH_ref+1] += np.random.normal([0, 0, 0, 0, 0, 0], [i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_ang, i/N_ref*sigma_ang, i/N_ref*sigma_ang])
                    
                    current_state = np.array(x_horizon[:,WH_ref+1]).reshape(12,)
                else:
                    LAST = True
                    x_WH = x_horizon
                    u_WH = u_horizon

                sideslip_WH = ocp_i.get_sideslip(x_WH)
                angle_of_attack_WH = ocp_i.get_angle_of_attack(x_WH)

                memory["state_trajectory"] = np.concatenate((memory["state_trajectory"], x_WH), axis=1)
                memory["input_trajectory"] = np.concatenate((memory["input_trajectory"], u_WH), axis=1)
                memory["sideslip"] = np.hstack((memory["sideslip"], sideslip_WH.T)) 
                memory["angle_of_attack"] = np.hstack((memory["angle_of_attack"], angle_of_attack_WH.T))   

                memory["computation_time"].append(info["time"])
                memory["iter_count"].append(info["iter_count"])
                memory["success"].append(info["success"])
                memory["t_horizon"].append(t_horizon)
                memory["t"].append(t)
                memory["ocp.spline.K"].append(ocp_i.spline.K)
                memory["y0_guess"].append(y0_guess)
                memory["y_horizon"].append(y_horizon)
                memory["x_horizon"].append(x_horizon)
                memory["x_WH"].append(x_WH)
                memory["CPs"].append(CPs)
                memory["current_VS_horizon_OK?"].append(OK)

            if i == 0:
                x_WH = X_ref[:,:WH_ref]       
                u_WH = U_ref[:,:WH_ref] 
                sideslip_WH = sideslip_ref[:WH_ref] 
                angle_of_attack_WH = angle_of_attack_ref[:WH_ref] 

                if perturbation:
                    X_ref[:6,WH_ref+1] += np.random.normal([0, 0, 0, 0, 0, 0], [i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_ang, i/N_ref*sigma_ang, i/N_ref*sigma_ang])
                    
                current_state = np.array(X_ref[:,WH_ref+1]).reshape(12,)

                memory["control_points"].append(Y_sol)
                memory["state_trajectory"] = x_WH
                memory["input_trajectory"] = u_WH
                memory["sideslip"] = sideslip_WH.T
                memory["angle_of_attack"] = angle_of_attack_WH.T

                memory["computation_time"].append(info["time"])
                memory["iter_count"].append(info["iter_count"])
                memory["success"].append(True)
                memory["t_horizon"].append(Ts_sol)
                memory["t"].append(Ts_sol)
                memory["ocp.spline.K"].append(ocp.spline.K)
                memory["y0_guess"].append(y0_guess)
                memory["y_horizon"].append(Y_sol)
                memory["x_horizon"].append(X_ref)
                memory["x_WH"].append(x_WH)
                memory["CPs"].append(control_points)
                memory["current_VS_horizon_OK?"].append(True)


    if MPC_tracking:
        current_state = x0.reshape(12,)

        for i in range(N_ref):
            if i % WH_ref == 0 and i != 0:

                N_i = N_ref - i

                PH = get_PH(i, N_ref, fixed_PH)
                t_horizon = PH*dt_ref                   

                OK = np.sum(X_ref[:,i] - current_state)

                CPs = get_amount_CPs(i, N_ref, CPs_ref, fixed_PH, fixed_CPs)
                order = CPs - int_knots + 1 

                if i <= N_ref - PH:
                    horizon_state = X_ref[:,i + PH]
                else:
                    print('YES: final goal incorporated')
                    horizon_state = X_ref[:,-1]

                TS = TailsitterModel(N_i)
                ocp_i = Ocp_Bspline(nx=TS.nx , nu=TS.nu, generation=False, tracking=True, X_ref=X_ref[:, i:i+PH], N=PH-1, int_knots=int_knots, order=order, invdyn=TS.InverseDynamicsArray, flat_order=TS.flat_order, Ts=t_horizon, store_intermediate=True, s_opts=s_opts_MPC)
                
                set_X0_XN_constraints(ocp=ocp_i, slack_X0=slack_X0, X0_fullyImposed=X0_fullyImposed, slack_XN=slack_XN, XN_fullyImposed=XN_fullyImposed, Full0=Full0, FullN=FullN)
                set_bound_variables(ocp=ocp_i, bounds_dict=variable_bounds, scales=scales)
                if obstacles:
                    objects = set_obstacles(ocp=ocp_i, spheres=spheres, cylinders=cylinders)
                else:
                    objects = None
                set_cost_function(ocp=ocp_i, scaling=scaling, regularization_weight=regularization_weight, scales=scales, generation=False)


                if warm_starts:
                    kk = np.linspace(0, 1, int_knots)
                    spline = Bspline(kk, order, Ts=t_horizon, N=PH, flat_order=TS.flat_order)
                    y0_guess = polynomial_guess_flat_output(spline=spline, Ts=t_horizon, init_state=current_state, goal_state=horizon_state, control_points=CPs)

                    y_horizon, t, info = ocp_i.solve(current_state, horizon_state, y0_guess=y0_guess)

                else:
                    y_horizon, t, info = ocp_i.solve(current_state, horizon_state, lin_interpol=True)
                
                x_horizon, u_horizon = ocp_i.get_state_action(y_coeff=y_horizon)
                x_horizon, u_horizon = np.array(x_horizon), np.array(u_horizon)

                if (PH > WH_ref):
                    LAST = False
                    x_WH = x_horizon[:,:WH_ref]                
                    u_WH = u_horizon[:,:WH_ref]     
                    
                else:
                    LAST = True
                    x_WH = x_horizon
                    u_WH = u_horizon

                try:
                    if perturbation*(PH >= WH_ref):
                        x_horizon[:6,WH_ref+1] += np.random.normal([0, 0, 0, 0, 0, 0], [i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_ang, i/N_ref*sigma_ang, i/N_ref*sigma_ang])
                    current_state = np.array(x_horizon[:,WH_ref+1]).reshape(12,)
                except:
                    current_state = np.array(x_horizon[:,-1]).reshape(12,)

                sideslip_WH = ocp_i.get_sideslip(x_WH)
                angle_of_attack_WH = ocp_i.get_angle_of_attack(x_WH)

                memory["state_trajectory"] = np.concatenate((memory["state_trajectory"], x_WH), axis=1)
                memory["input_trajectory"] = np.concatenate((memory["input_trajectory"], u_WH), axis=1)
                memory["sideslip"] = np.hstack((memory["sideslip"], sideslip_WH.T)) 
                memory["angle_of_attack"] = np.hstack((memory["angle_of_attack"], angle_of_attack_WH.T))   

                memory["computation_time"].append(info["time"])
                memory["iter_count"].append(info["iter_count"])
                memory["success"].append(info["success"])
                memory["t_horizon"].append(t_horizon)
                memory["t"].append(t)
                memory["ocp.spline.K"].append(ocp_i.spline.K)
                memory["y0_guess"].append(y0_guess)
                memory["y_horizon"].append(y_horizon)
                memory["x_horizon"].append(x_horizon)
                memory["x_WH"].append(x_WH)
                memory["CPs"].append(CPs)
                memory["current_VS_horizon_OK?"].append(OK)
                memory["PH"].append(PH)

            if i == 0:
                x_WH = X_ref[:,:WH_ref+1]       
                u_WH = U_ref[:,:WH_ref+1] 
                sideslip_WH = sideslip_ref[:WH_ref+1] 
                angle_of_attack_WH = angle_of_attack_ref[:WH_ref+1] 

                if perturbation:
                    X_ref[:6,WH_ref+1] += np.random.normal([0, 0, 0, 0, 0, 0], [i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_pos, i/N_ref*sigma_ang, i/N_ref*sigma_ang, i/N_ref*sigma_ang])
                    
                current_state = np.array(X_ref[:,WH_ref+1]).reshape(12,)

                memory["control_points"].append(Y_sol)
                memory["state_trajectory"] = x_WH
                memory["input_trajectory"] = u_WH
                memory["sideslip"] = sideslip_WH.T
                memory["angle_of_attack"] = angle_of_attack_WH.T

                memory["computation_time"].append(info["time"])
                memory["iter_count"].append(info["iter_count"])
                memory["success"].append(True)
                memory["t_horizon"].append(Ts_sol)
                memory["t"].append(Ts_sol)
                memory["ocp.spline.K"].append(ocp.spline.K)
                memory["y0_guess"].append(y0_guess)
                memory["y_horizon"].append(Y_sol)
                memory["x_horizon"].append(X_ref)
                memory["x_WH"].append(x_WH)
                memory["CPs"].append(control_points)
                memory["current_VS_horizon_OK?"].append(True)
                memory["PH"].append(PH_ref)


    winsound.Beep(1000, 500)  


# %%
X_sol = np.array(memory["state_trajectory"])
U_sol = np.array(memory["input_trajectory"])

# %%
sideslip_sol = np.array(memory["sideslip"]).reshape(N+1,-1)
angle_of_attack_sol = np.array(memory["angle_of_attack"]).reshape(N+1,-1)

# %%
if plot_sol:
    plot_final_vs_ref(X=X_sol, X_ref=X_ref, traject=traject, x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks, obstacles=objects, control_points=Y_sol, views3D=views3D, views2D=views2D, pos=True, orient=True, pos_ref=True, orient_ref=False, labels=True, reduc=10, reduc_ref=10, aspect='auto-adjusted', B=B, H=H, save=saveplots, filename=filename_sol, directory=directory_sol)

# %%
if plot_sol_states:
    plot_positions(Ts_sol, N, X_sol, X_ref, save=saveplots, filename=filename_sol, directory=directory_sol)
    plot_angles(Ts_sol, N, X_sol, X_ref, save=saveplots, filename=filename_sol, directory=directory_sol)
    plot_linVelocities(Ts_sol, N, X_sol, X_ref, save=saveplots, filename=filename_sol, directory=directory_sol)
    plot_angVelocities(Ts_sol, N, X_sol, X_ref, save=saveplots, filename=filename_sol, directory=directory_sol)

if plot_sol_inputs:
    plot_wingAngles(Ts_sol, N, U_sol, U_ref, save=saveplots, filename=filename_sol, directory=directory_sol)
    plot_thrusts(Ts_sol, N, U_sol, U_ref, save=saveplots, filename=filename_sol, directory=directory_sol)

if plot_sol_sideslip_AOA:
    plot_sideslip(Ts_sol, N, sideslip_sol, sideslip_ref, save=saveplots, filename=filename_sol, directory=directory_sol)
    plot_AOA(Ts_sol, N, angle_of_attack_sol, angle_of_attack_ref, save=saveplots, filename=filename_sol, directory=directory_sol)


# %%
print('--------------------------------------------------------------------------------')
print('Successes:       {}'.format(memory["success"]))
print('Iters:           {}'.format(memory["iter_count"]))
print('CPs              {}'.format(memory['CPs']))
print('PH               {}'.format(memory["PH"]))
print('Process times:   {}'.format(memory['computation_time']))
print('     TOTAL       {}'.format(sum(memory['computation_time'])))
print('--------------------------------------------------------------------------------')

# %%
DICT_sol = {'Successes:': memory["success"],
            'Iters:': memory["iter_count"],
            'CPs:' : memory['CPs'],
            'sigma_pos: ': sigma_pos,
            'sigma_ang: ': sigma_ang,
            'Process times:': memory['computation_time'],
            'TOTAL:' : sum(memory['computation_time'])}


if saveplots:
    with open(os.path.join(directory_sol, dictname_sol), 'w') as file:
        json.dump(DICT_sol, file, indent=4) 

# %%
