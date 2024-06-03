import numpy as np

def Landing_flat_output_with_derivates(R, Ts, N):
    """
    Args:
        R (float): radius of the take off
        Ts (float): end time
        N (int): number of time intervals

    Outputs:
        s (array_like: matrix of column vectors):       flat output   s = [x, y, z, psi].T                      
        ds (array_like: matrix of column vectors):      flat output   ds = [dx, dy, dz, dpsi].T                
        dds (array_like: matrix of column vectors):     flat output   dds = [ddx, ddy, ddz, ddpsi].T            
        ddds (array_like: matrix of column vectors):    flat output   ddds = [dddx, dddy, dddz, dddpsi].T       
        dddds (array_like: matrix of column vectors):   flat output   dddds = [ddddx, ddddy, ddddz, ddddpsi].T 
        v (array_like: 1D): speed at each time instant
    """
    t = np.linspace(0,Ts,N)

    Vx_start = 20
    Vx_end = 2
    A = -(-Ts*Vx_start-Ts*Vx_end+2*R)/Ts**3
    B = (-2*Ts*Vx_start-Ts*Vx_end+3*R)/Ts**2
    C = Vx_start
    x = A*t**3+B*t**2+C*t
    dx = 3*A*t**2 + 2*B*t + C
    ddx = 6*A*t**1 + 2*B
    dddx = 6*A*t**0

    Vz_start = 1
    Vz_end = 5
    A = -(-Ts*Vz_start-Ts*Vz_end+2*R)/Ts**3
    B = (-2*Ts*Vz_start-Ts*Vz_end+3*R)/Ts**2
    C = Vz_start
    z = A*t**3+B*t**2+C*t
    dz = 3*A*t**2 + 2*B*t + C
    ddz = 6*A*t**1 + 2*B
    dddz = 6*A*t**0

    s = np.zeros((4,N))
    s[0,:] = x*np.sqrt(2)/2
    s[1,:] = x*np.sqrt(2)/2
    s[2,:] = z
    s[3,:] = np.pi/4

    ds = np.zeros((4,N))
    ds[0,:] = dx*np.sqrt(2)/2
    ds[1,:] = dx*np.sqrt(2)/2
    ds[2,:] = dz

    dds = np.zeros((4,N))
    dds[0,:] = ddx*np.sqrt(2)/2
    dds[1,:] = ddx*np.sqrt(2)/2
    dds[2,:] = ddz

    ddds = np.zeros((4,N))
    ddds[0,:] = dddx*np.sqrt(2)/2
    ddds[1,:] = dddx*np.sqrt(2)/2
    ddds[2,:] = dddz

    dddds = np.zeros((4,N))

    vx = ds[0,:]
    vy = ds[1,:]
    vz = ds[2,:]
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    return s, ds, dds, ddds, dddds, v