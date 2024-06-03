import numpy as np

def AscenTurn_flat_output_with_derivates(R, Ts, N):
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
    s[2,:] = R/4 - R/4*np.cos(np.pi/Ts*t)
    s[3,:] = np.pi/Ts*t

    ds = np.zeros((4,N))
    ds[0,:] = -R*np.sin(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    ds[1,:] = R*np.cos(np.pi/Ts*t-np.pi/2)*np.pi/Ts
    ds[2,:] = R/4*np.sin(np.pi/Ts*t)*np.pi/Ts
    ds[3,:] = np.pi/Ts

    dds = np.zeros((4,N))
    dds[0,:] = -R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**2
    dds[1,:] = -R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**2
    dds[2,:] = R/4*np.cos(np.pi/Ts*t)*(np.pi/Ts)**2

    ddds = np.zeros((4,N))
    ddds[0,:] = R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**3
    ddds[1,:] = -R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**3
    ddds[2,:] = -R/4*np.sin(np.pi/Ts*t)*(np.pi/Ts)**3

    dddds = np.zeros((4,N))
    dddds[0,:] = R*np.cos(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**4
    dddds[1,:] = R*np.sin(np.pi/Ts*t-np.pi/2)*(np.pi/Ts)**4
    dddds[2,:] = -R/4*np.cos(np.pi/Ts*t)*(np.pi/Ts)**4

    vx = ds[0,:]
    vy = ds[1,:]
    vz = ds[2,:]
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    return s, ds, dds, ddds, dddds, v