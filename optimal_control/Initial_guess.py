# %%
import casadi as cd
import matplotlib.pyplot as plt
import numpy as np
import math 
import os

from model.Tailsitter import TailsitterModel, plot_trajectory, plot_final_vs_ref

from tools.Dijkstra_los_3D import Dijkstra_los_3D

from control_Bsplines import Ocp_Bspline

from tools.Bspline import Bspline

# %%
""" Functions """
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

    ax.plot(np.linspace(0,Ts,N+1), X[0,:].reshape((N+1,)), color='#0A75AD', label='$x(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[1,:].reshape((N+1,)), color='#CC0000', label='$y(t)$')
    ax.plot(np.linspace(0,Ts,N+1), X[2,:].reshape((N+1,)), color='#008000', label='$z(t)$')

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
Ts_guess = 10
N = 100        

m = 1
k = 6

flatOutput_guess    = True
fullTraject_guess   = False                 
Dijkstra_guess      = False
if Dijkstra_guess:
    discrWidth          = 5
    offset              = False    
    t_offset            = 0
    marge               = 2

obstacles       = True
spheresAdded    = True
marge_spheres   = 2
cylindersAdded  = True
marge_cylinders = 2

obs     = 'S1S2S3C1'

# ----------    Savings


save_guess          = False
directory_guess     = r"_"
os.makedirs(directory_guess, exist_ok=True)

# ----------    Plots       
plot_guess              = True
plot_guess_states       = False
plot_guess_inputs       = False
plot_guess_sideslip_AOA = False

B, H    = 4, 9

# ----------    Trajectory  
straight_line   = False
straight_line2  = False
acrobatic_turn  = False
ascending_turn  = True
ascending_turn2 = False
ascending_turn3 = False
ClimbTurn       = False
ClimbTurn2      = False
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

    #cylinders = {'cyl_1': {'r': 10, 'z': [-10,30], 'xy': [50,25]}}

elif cylindersAdded*(ascending_turn + ascending_turn2 + ascending_turn3):
    cylinders = {'cyl_1': {'r': 5, 'z': [-5,30], 'xy': [30,85]}}
else:
    cylinders = {}


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
ocp = Ocp_Bspline(nx=tailsitter.nx , nu=tailsitter.nu, generation=True, tracking=False, N=N, int_knots=int_knots, order=order, invdyn=tailsitter.InverseDynamicsArray, flat_order=tailsitter.flat_order, Ts=Ts_guess, store_intermediate=True)


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

# %%
if obstacles:
    objects = set_obstacles(ocp=ocp, spheres=spheres, cylinders=cylinders, marge_spheres=marge_spheres, marge_cylinders=marge_cylinders)
else:
    objects = None

# %%
""" Construct the initial guess if necessary """
if flatOutput_guess:
    guess = 'flatOutput'
    y0_guess = polynomial_guess_flat_output(spline=spline, Ts=Ts_guess, init_state=init_state, goal_state=goal_state, control_points=control_points, plot=False)
elif Dijkstra_guess:
    guess = 'Dijkstra'
    y0_guess = polynomial_guess_Dijkstra_flat_output(spline=spline, Ts=Ts_guess, init_state=init_state, goal_state=goal_state, control_points=control_points, obstacles=objects, discrWidth=discrWidth, marge=marge, offset=offset, t_offsett=t_offset, plot=True)
elif fullTraject_guess:
    guess = 'fullTraject'
    y0_guess = polynomial_guess_trajectory(Ts_guess, N, init_state, goal_state, plot=False)
else:
    guess = 'LinInterpol'
    y0_guess = np.linspace(init_state[0 : flat_order], goal_state[0 : flat_order], control_points)


# %%
X_guess, U_guess = ocp.get_state_action(y_coeff=y0_guess)
X_guess, U_guess = X_guess.full(), U_guess.full()
try:
    y0_guess = y0_guess.full()
except:
    pass
sideslip_guess = ocp.get_sideslip(X_guess)
angle_of_attack_guess = ocp.get_angle_of_attack(X_guess)

filename_guess      = traject + '_' + guess + '_' + obs + '__N_{}__Ts_{}_____(m,k)_({},{})'.format(N, int(Ts_guess), m, k)

# %%
if plot_guess:
    plot_final_vs_ref(X_guess, traject=traject, x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks, obstacles=objects, control_points=y0_guess, views3D=views3D, views2D=views2D, pos=True, orient=True, labels=True, reduc=10, aspect='auto-adjusted', B=B, H=H, save=save_guess, filename=filename_guess, directory=directory_guess)

# %%
if plot_guess_states:
    plot_positions(Ts_guess, N, X_guess, save=save_guess, filename=filename_guess, directory=directory_guess)
    plot_angles(Ts_guess, N, X_guess, save=save_guess, filename=filename_guess, directory=directory_guess)
    plot_linVelocities(Ts_guess, N, X_guess, save=save_guess, filename=filename_guess, directory=directory_guess)
    plot_angVelocities(Ts_guess, N, X_guess, save=save_guess, filename=filename_guess, directory=directory_guess)

if plot_guess_inputs:
    plot_wingAngles(Ts_guess, N, U_guess, save=save_guess, filename=filename_guess, directory=directory_guess)
    plot_thrusts(Ts_guess, N, U_guess, save=save_guess, filename=filename_guess, directory=directory_guess)

if plot_guess_sideslip_AOA:
    plot_sideslip(Ts_guess, N, sideslip_guess, save=save_guess, filename=filename_guess, directory=directory_guess)
    plot_AOA(Ts_guess, N, angle_of_attack_guess, save=save_guess, filename=filename_guess, directory=directory_guess)


# %%