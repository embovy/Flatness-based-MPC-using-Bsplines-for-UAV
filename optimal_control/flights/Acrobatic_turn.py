import numpy as np
from tool_functions import Rotx, Roty, Rotz

def AcroTurn_flat_output_with_derivates(R, Ts, N):
    """
    Args:
        R (float): radius of the turn
        Ts (float): end time
        N (int): number of time intervals

    Outputs:
        s (array_like: matrix of column vectors):       flat output   s = [x, y, z, psi].T                      
        ds (array_like: matrix of column vectors):      flat output   ds = [dx, dy, dz, dpsi].T                
        dds (array_like: matrix of column vectors):     flat output   dds = [ddx, ddy, ddz, ddpsi].T            
        ddds (array_like: matrix of column vectors):    flat output   ddds = [dddx, dddy, dddz, dddpsi].T       
        dddds (array_like: matrix of column vectors):   flat output   dddds = [ddddx, ddddy, ddddz, ddddpsi].T 
    """
    t = np.linspace(0,Ts,N)

    s = np.zeros((4,N))
    s[0,:] = R*np.cos(np.pi/Ts*t-np.pi/2)
    s[1,:] = R*np.sin(np.pi/Ts*t-np.pi/2) + R
    s[2,:] = R/4 - R/4*np.cos(2*np.pi/Ts*t)
    s[3,:] = np.pi/Ts*t

    ds = np.zeros((4,N))
    ds[0,:] = -R*np.sin(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    ds[1,:] = R*np.cos(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    ds[2,:] = R/4*np.sin(2*np.pi/Ts*t)*2*np.pi/Ts
    ds[3,:] = np.pi/Ts

    dds = np.zeros((4,N))
    dds[0,:] = -R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**2
    dds[1,:] = -R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**2
    dds[2,:] = R/4*np.cos(2*np.pi/Ts*t)*(2*np.pi/Ts)**2

    ddds = np.zeros((4,N))
    ddds[0,:] = R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**3
    ddds[1,:] = -R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**3
    ddds[2,:] = -R/4*np.sin(2*np.pi/Ts*t)*(2*np.pi/Ts)**3

    dddds = np.zeros((4,N))
    dddds[0,:] = R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**4
    dddds[1,:] = R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**4
    dddds[2,:] = -R/4*np.cos(2*np.pi/Ts*t)*(2*np.pi/Ts)**4 

    vx = ds[0,:]
    vy = ds[1,:]
    vz = ds[2,:]
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    return s, ds, dds, ddds, dddds, v

def AcroTurn_states(R, Ts, N):
    """
    Args:
        R (float): radius of the turn
        Ts (float): end time
        N (int): number of time intervals

    Outputs:
        p (array_like: matrix of column vectors):       flat output   p = [x, y, z].T                      
        q (array_like: matrix of column vectors):       flat output   q = [psi, phi, theta].T                
        v (array_like: matrix of column vectors):       flat output   v = [xdot, ydot, zdot].T            
        ω_b (array_like: matrix of column vectors):     flat output   ω_b = [psidot_b, phidot_b, thetadot_b].T       
    """
    t = np.linspace(0,Ts,N)

    p = np.zeros((3,N))
    p[0,:] = R*np.cos(np.pi/Ts*t-np.pi/2)
    p[1,:] = R*np.sin(np.pi/Ts*t-np.pi/2) + R
    p[2,:] = R/4 - R/4*np.cos(2*np.pi/Ts*t)

    q = np.zeros((3,N))
    q[0,:] = np.pi/Ts*t                         # psi = arctan2(ydot, xdot)
    q[1,:] = np.arctan2(np.cos(np.pi/Ts*t), 1)  # phi = arctan2(zdot, ydot)
    q[2,:] = np.arctan2(1, np.sin(np.pi/Ts*t))  # theta = arctan2(xdot, zdot)

    v = np.zeros((3,N))
    v[0,:] = -R*np.sin(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    v[1,:] = R*np.cos(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    v[2,:] = R/4*np.sin(2*np.pi/Ts*t)*2*np.pi/Ts

    ω = np.zeros((3,N))
    ω[0,:] = np.pi/Ts                                                   # psidot
    ω[1,:] = -1/(1 + np.cos(np.pi/Ts*t)**2)*np.sin(np.pi/Ts*t)*np.pi/Ts # phidot
    ω[2,:] = -1/(1 + np.sin(np.pi/Ts*t)**2)*np.cos(np.pi/Ts*t)*np.pi/Ts # thetadot

    ω_b = np.zeros((3,N))
    for i in range(N):
        psi, phi, theta = q[0,i], q[1,i],q[2,i],
        ω_b[:,i] = np.dot(Roty(theta).T, np.dot(Rotx(phi).T, np.dot(Rotz(psi).T, ω[:,i])))
    
    return p, q, v, ω_b  