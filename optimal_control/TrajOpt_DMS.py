# %%
import casadi as cd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from model.Tailsitter import TailsitterModel, plot_trajectory, plot_final_vs_ref

from control_DMS_Ts import Ocp_DMS_Ts

# %%
""" Functions """
def set_X0_XN_constraints(ocp: Ocp_DMS_Ts, slack_X0, X0_fullyImposed, slack_XN, XN_fullyImposed, Full=6, Partly=4, Ts_fixed=False):
    X = ocp.X

    print('Full = {}, Partly = {}'.format(Full, Partly))

    if Ts_fixed == False:
       print('Ts_bound                 OK')
       ocp.opti.subject_to(1 < ocp.Ts)

    # Initial and goal state
    if slack_X0:
        if X0_fullyImposed:
            print('X0 Fully + Slack')
            slack = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
            ocp.opti.subject_to(ocp.bounded((ocp.X0[0:Full] - slack), X[0:Full, 0], (ocp.X0[0:Full] + slack)))
        else:
            print('X0 Partly + Slack')
            slack = np.array([1e-1, 1e-1, 1e-1, 1e-1])
            ocp.opti.subject_to(ocp.bounded((ocp.X0[0:Partly] - slack), X[0:Partly, 0], (ocp.X0[0:Partly] + slack)))    

    else:
        if X0_fullyImposed:
            print('X0 Fully')
            ocp.opti.subject_to(X[0:Full, 0] == ocp.X0[0:Full])
        else:
            print('X0 Partly')
            ocp.opti.subject_to(X[0:Partly, 0] == ocp.X0[0:Partly])

    if slack_XN:
        if XN_fullyImposed:
            print('XN Fully + Slack')
            slack = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
            ocp.opti.subject_to(ocp.bounded((ocp.XN[0:Full] - slack), X[0:Full, -1], (ocp.XN[0:Full] + slack)))
        else:
            print('XN Partly + Slack')
            slack = np.array([1e-1, 1e-1, 1e-1, 1e-1])
            ocp.opti.subject_to(ocp.bounded((ocp.XN[0:Partly] - slack), X[0:Partly, -1], (ocp.XN[0:Partly] + slack)))    
    else:
        if XN_fullyImposed:
            print('XN Fully')
            ocp.opti.subject_to(X[0:Full, -1] == ocp.XN[0:Full])
        else:
            print('XN Partly')
            ocp.opti.subject_to(X[0:Partly, -1] == ocp.XN[0:Partly])
       
def set_bound_variables(ocp: Ocp_DMS_Ts, bounds_dict={}, scales={}): 
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
        ocp.opti.subject_to(ocp.opti.bounded(0, U[0, :]  / scales["scaling_thr"], bounds_dict["thrust"][1]  / scales["scaling_thr"]))
        ocp.opti.subject_to(ocp.opti.bounded(0, U[1, :]  / scales["scaling_thr"], bounds_dict["thrust"][1]  / scales["scaling_thr"]))

        #ocp.opti.subject_to(ocp.opti.bounded(0, U[0, :], bounds_dict["thrust"][1]))
        #ocp.opti.subject_to(ocp.opti.bounded(0, U[1, :], bounds_dict["thrust"][1]))

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

def set_obstacles(ocp: Ocp_DMS_Ts, marge_spheres=2, marge_cylinders=2, spheres={}, cylinders={}):
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

def set_cost_function(ocp: Ocp_DMS_Ts, scaling, Q=None, R=None, P=None, scales={}, terminal_cost=False, generation=True):
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
    if P is None:
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
        print('SCALED cost function')
        Q[0,:] = Q[0,:] / scales["scaling_pos"]*2
        Q[1,:] = Q[1,:] / scales["scaling_ang"]*2
        Q[2,:] = Q[2,:] / scales["scaling_posdot"]*2
        Q[3,:] = Q[3,:] / scales["scaling_angdot"]*2

        R[0,:] = R[0,:] / scales["scaling_thr"]*2
        R[1,:] = R[1,:] / scales["scaling_del"]*2

    else:
        print('UNSCALED cost function')

    ocp.running_cost = ((ocp.x.T - ocp.xN.T)@Q@(ocp.x-ocp.xN) + ocp.u.T@R@ocp.u)

    if terminal_cost:
        print('Terminal cost added')
        if scaling and scales != {}:
            P[0,:] = P[0,:] / scales["scaling_pos"]*2
            P[1,:] = P[1,:] / scales["scaling_ang"]*2
            P[2,:] = P[2,:] / scales["scaling_posdot"]*2
            P[3,:] = P[3,:] / scales["scaling_angdot"]*2

        ocp.terminal_cost = (ocp.x.T - ocp.xN.T)@P@(ocp.x-ocp.xN)



def polynomial_guess_flat_output(Ts, N, init_state, goal_state, plot=False):
    x0, y0, z0, psi0, phi0, theta0, xdot0, ydot0, zdot0, psidot_b0, phidot_b0, thetadot_b0 = init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5], init_state[6], init_state[7], init_state[8], init_state[9], init_state[10], init_state[11]
    xN, yN, zN, psiN, phiN, thetaN, xdotN, ydotN, zdotN, psidot_bN, phidot_bN, thetadot_bN = goal_state[0], goal_state[1], goal_state[2], goal_state[3], goal_state[4], goal_state[5], goal_state[6], goal_state[7], goal_state[8], goal_state[9], goal_state[10], goal_state[11]

    qua = lambda t, Ts, s0, sN, sdot0, sdotN: [-2*(s0-sN)-Ts*(sdot0+sdotN)]*(1-t/Ts)**3 + [3*(s0-sN)+Ts*(sdot0+2*sdotN)]*(1-t/Ts)**2 + [-Ts*sdotN]*(1-t/Ts) + sN
    lin = lambda t, Ts, angle0, angleN: [angle0-angleN]*(1-t/Ts) + angleN

    t = np.linspace(0, Ts, N+1)

    X0_guess = np.array([qua(t,Ts,x0,xN,xdot0,xdotN), 
                        qua(t,Ts,y0,yN,ydot0,ydotN), 
                        qua(t,Ts,z0,zN,zdot0,zdotN), 
                        lin(t,Ts,psi0,psiN),
                        lin(t,Ts,phi0,phiN),
                        lin(t,Ts,theta0,thetaN),
                        lin(t,Ts,xdot0,xdotN),
                        lin(t,Ts,ydot0,ydotN),
                        lin(t,Ts,zdot0,zdotN),
                        lin(t,Ts,psidot_b0,psidot_bN),
                        lin(t,Ts,phidot_b0,phidot_bN),
                        lin(t,Ts,thetadot_b0,thetadot_bN)]).T

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X0_guess[:,0], X0_guess[:,1], X0_guess[:,2], c='g', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_aspect('equal', 'box')
        ax.set_title('Polynomial guess path')
        plt.show()

    plot_final_vs_ref(X0_guess.T, "Tailsitter guess", reduc=10, aspect='auto-adjusted')
    
    return X0_guess.T

def polynomial_guess_trajectory(Ts, N, init_state, goal_state, plot=False):
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

    t = np.linspace(0, Ts, N+1)

    x, y, z = pos(t,Ts,x0,xN,xdot0,xdotN), pos(t,Ts,y0,yN,ydot0,ydotN), pos(t,Ts,z0,zN,zdot0,zdotN)
    psi, phi, theta = ang(t,Ts,Epsi,Fpsi,Gpsi,Hpsi), ang(t,Ts,Ephi,Fphi,Gphi,Hphi), ang(t,Ts,Etheta,Ftheta,Gtheta,Htheta)
    xdot, ydot, zdot = posdot(t,Ts,x0,xN,xdot0,xdotN), posdot(t,Ts,y0,yN,ydot0,ydotN), posdot(t,Ts,z0,zN,zdot0,zdotN)
    psidot_b = psid_b(t,Ts,psi,phi,theta)
    phidot_b = phid_b(t,Ts,psi,phi,theta)
    thetadot_b = thetad_b(t,Ts,psi,phi,theta)

    X0_guess = np.array([x,y,z,psi,phi,theta,xdot,ydot,zdot,psidot_b,phidot_b,thetadot_b])

    if plot:
        plot_trajectory(X0_guess, "Initial guess states: boundary conditions", reduc=10, aspect='auto-adjusted')

    return X0_guess



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

    ax.plot(np.linspace(0,Ts,N), U[2,:]*180/np.pi, color='#6A329F', marker='v', markersize=6, markevery=0.1, label=r'$\delta_{1}(t)$')
    ax.plot(np.linspace(0,Ts,N), U[3,:]*180/np.pi, color='#FF921F', marker='p', markersize=6, markevery=0.1, label=r'$\delta_{2}(t)$')

    if U_guess is not None:
        ax.plot(np.linspace(0,Ts,N), U_guess[2,:]*180/np.pi, color='#6A329F', marker='v', markersize=4, markevery=0.1, linestyle=(3, (5,5)), alpha=0.8)
        ax.plot(np.linspace(0,Ts,N), U_guess[3,:]*180/np.pi, color='#FF921F', marker='p', markersize=4, markevery=0.1, linestyle=(0, (5,5)), alpha=0.8)

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

    ax.plot(np.linspace(0,Ts,N), U[0,:], color='#6A329F', marker='v', markersize=6, markevery=0.1, label=r'$T_{1}(t)$')
    ax.plot(np.linspace(0,Ts,N), U[1,:], color='#FF921F', marker='p', markersize=6, markevery=0.1, label=r'$T_{2}(t)$')

    if U_guess is not None:
        ax.plot(np.linspace(0,Ts,N), U_guess[0,:], color='#6A329F', marker='v', markersize=4, markevery=0.1, linestyle=(3, (5,5)), alpha=0.8)
        ax.plot(np.linspace(0,Ts,N), U_guess[1,:], color='#FF921F', marker='p', markersize=4, markevery=0.1, linestyle=(0, (5,5)), alpha=0.8)

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
N = 500

reduc = 50

warm_start          = True
Ts_fixed            = False                 

flatOutput_guess    = False
fullTraject_guess   = True                 

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

# ----------    Savings

save_ref        = False
directory_ref   = r"_"
os.makedirs(directory_ref, exist_ok=True)

# ----------    Plots       
plot_initial_goal       = True
plot_y0_guess           = True

plot_ref_states         = True
plot_ref_inputs         = True
plot_ref_sideslip_AOA   = True

plot_ref                = True
plot_ref_convergence    = True

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

Straight        = False
Turn90          = False

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

if Straight:
    x_ticks = np.array([0, 20, 40, 60, 80])
    y_ticks = np.array([-10, 0, 10])
    z_ticks = np.array([0, 10])
    views3D = [(30,-60,0), (30,160,0)]
    views2D = [(90,180,0), (0,90,0), (0,180,0)]

if Turn90:
    x_ticks = np.array([0, 20, 40, 60])
    y_ticks = np.array([0, 20, 40, 60])
    z_ticks = np.array([0, 10])
    views3D = [(30,-60,0), (30,160,0)]
    views2D = [(90,180,0), (0,90,0), (0,180,0)]

# ----------    Usefull dictionaries  
scales = {"scaling_pos": 100,          #*1e14
          "scaling_ang": 3,            #*1e14
          "scaling_posdot": 20,        #*1e14
          "scaling_angdot": 3,         #*1e14
          "scaling_thr": 5,            #*1e14
          "scaling_del": 0.0008,       # = 0.0008*1e14
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

variable_bounds = {"thrust":            [True, 30],
                   "delta":             [True, 1],
                   "linear_velocity":   [True, 30],
                   "psidot":            [False, 90*np.pi/180],
                   "phidot":            [False, 45*np.pi/180],
                   "thetadot":          [False, 45*np.pi/180],
                   "sideslip":          [True,  10*np.pi/180],
                   "angle_of_attack":   [True,  30*np.pi/180]
                   }

s_opts = { "max_iter": 500,                        # Maximum number of iterations (150 ideal)
          "tol": 1e-5,                             # Default 1e-8      Overall optimization tolerance
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

# %%
""" Define the optimal control problem """
if Ts_fixed:
    ocp = Ocp_DMS_Ts(nx=tailsitter.nx, nu=tailsitter.nu, Ts=Ts_guess, N=N, f=tailsitter.ForwardDynamics1D, store_intermediate=True, integrator='rk4', s_opts=s_opts)
else:
    ocp = Ocp_DMS_Ts(nx=tailsitter.nx, nu=tailsitter.nu, N=N, f=tailsitter.ForwardDynamics1D, store_intermediate=True, integrator='rk4', s_opts=s_opts)

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
    
elif Straight:
    traject = 'Straight'
    init_state = [1e-6, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,
                  10, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,]
    goal_state = [80, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,
                  15, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6]

elif Turn90:
    traject = 'Turn90'
    init_state = [1e-6, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,
                  5, 1e-6, 1e-6,
                  1e-6, 1e-6, 1e-6,]
    goal_state = [50, 50, 1e-6,
                  np.pi/2, 1e-6, 1e-6,
                  1e-6, 5, 1e-6,
                  1e-6, 1e-6, 1e-6]

else:
    raise Exception("No initial state and goal state provided!")


situation = np.hstack([np.array(init_state).reshape(12,1), np.array(goal_state).reshape(12,1)])
if plot_initial_goal:
    plot_trajectory(situation, "Initial & Goal state", reduc=1, aspect='auto-adjusted')
  

# %%
""" Extend the ocp-problem with the necessary information """
set_X0_XN_constraints(ocp=ocp, slack_X0=slack_X0, X0_fullyImposed=X0_fullyImposed, slack_XN=slack_XN, XN_fullyImposed=XN_fullyImposed, Ts_fixed=Ts_fixed)

set_bound_variables(ocp=ocp, bounds_dict=variable_bounds, scales=scales)

set_cost_function(ocp=ocp, scaling=scaling, scales=scales)

if obstacles:
    objects = set_obstacles(ocp=ocp, spheres=spheres, cylinders=cylinders, marge_spheres=marge_spheres, marge_cylinders=marge_cylinders)
else:
    objects = None

# %%
""" Construct the initial guess if necessary """
if flatOutput_guess:
    guess = 'flatOutput'
    X0_guess = polynomial_guess_flat_output(Ts=Ts_guess, N=N, init_state=init_state, goal_state=goal_state, plot=plot_y0_guess)
elif fullTraject_guess:
    guess = 'fullTraject'
    X0_guess = polynomial_guess_trajectory(Ts_guess, N, init_state, goal_state, plot=plot_y0_guess)
else:
    X0_guess = None


# %%
""" Solve the ocp via DMS and get the reference trajectory & input sequences """
if warm_start:
    if X0_guess is not None:
        Usol, Xsol, Ts_sol, info = ocp.solve(init_state, goal_state, X0_guess=X0_guess)
    else:
        guess = 'LinInterpol'
        Usol, Xsol, Ts_sol, info = ocp.solve(init_state, goal_state, lin_interpol=True)
else:
    guess = 'Zero'
    Usol, Xsol, Ts_sol, info = ocp.solve(init_state, goal_state)

if X0_guess is not None:
    sideslip_guess = ocp.get_sideslip(X0_guess)
    angle_of_attack_guess = ocp.get_angle_of_attack(X0_guess)

X_ref, U_ref = Xsol, Usol
sideslip_ref = ocp.get_sideslip(Xsol)
angle_of_attack_ref = ocp.get_angle_of_attack(Xsol)

# %%
""" Data """
print('------------------------------------------------')
print('Warmstart                = {}'.format(warm_start))
print('Success                  = {}'.format(info["success"]))
print('Intermediate solutions   = {}'.format(np.shape(ocp.X_sol_intermediate)))
print('Iterations               = {}'.format(info['iter_count']))
print('Process Time             = {}'.format(info['time']))
print('Ts_sol                   = {}'.format(Ts_sol))
print('------------------------------------------------')


# %%
""" Saving situation in dictionary"""
filename_ref    = traject + '_' + guess + '__N_{}__Ts_{}'.format(N, int(Ts_guess))
dictname_ref    =  traject + '_' + guess

DICT = {'traject': traject,
        'guess': guess,
        'X_guess': X0_guess.tolist(),
        'sideslip_guess': sideslip_guess.full().tolist(),
        'angle_of_attack_guess': angle_of_attack_guess.full().tolist(),
        'X_ref': X_ref.tolist(),
        'U_ref': U_ref.tolist(),
        'sideslip_ref': sideslip_ref.full().tolist(),
        'angle_of_attack_ref': angle_of_attack_ref.full().tolist(),
        'info': info
        }

if save_ref:
    with open(os.path.join(directory_ref, dictname_ref), 'w') as file:
        json.dump(DICT, file, indent=4) 


# %%
""" Plotting reference quantities """
if X0_guess is not None:
    X0_guess = np.array(X0_guess)
    U0_guess = None
    sideslip_guess = np.array(sideslip_guess).reshape(N+1,-1)
    angle_of_attack_guess = np.array(angle_of_attack_guess).reshape(N+1,-1)
else:
    X0_guess = None
    U0_guess = None
    sideslip_guess = None
    angle_of_attack_guess = None

X_ref = np.array(X_ref)
U_ref = np.array(U_ref)
sideslip_ref = np.array(sideslip_ref).reshape(N+1,-1)
angle_of_attack_ref = np.array(angle_of_attack_ref).reshape(N+1,-1)

# %%
if plot_ref:
    plot_final_vs_ref(X_ref, traject=traject, x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks, obstacles=objects, views3D=views3D, views2D=views2D, pos=True, orient=True, labels=True, reduc=reduc, aspect='auto-adjusted', B=B, H=H, save=save_ref, filename=filename_ref, directory=directory_ref)

# %%
if plot_ref_states:
    plot_positions(Ts_sol, N, X_ref, X0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_angles(Ts_sol, N, X_ref, X0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_linVelocities(Ts_sol, N, X_ref, X0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_angVelocities(Ts_sol, N, X_ref, X0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)

if plot_ref_inputs:
    plot_wingAngles(Ts_sol, N, U_ref, U0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)
    plot_thrusts(Ts_sol, N, U_ref, U0_guess, save=save_ref, filename=filename_ref, directory=directory_ref)

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