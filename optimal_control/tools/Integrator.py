def rk4(f, x0, u, DT, M=1):
    """The runge-kutta method integration with the derivative f, initial state x0, control u,
    time interval DT and number of integration intervals M

    Args:
        f (callable): dynamics, derivative of state
        x0 (array): initial state
        u (array): control
        DT (float): total integration time
        M (int): number of integration steps

    Returns:
        x: state at the end of the integration, at time T
    """
    dt = DT/M
    x = x0
    
    for _ in range(M):
        k1 = f(x, u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3, u)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
    return x

def Euler(f, x0, u, dt, M=1):
    """
    Euler integration method over 1 time step dt, perhaps divided into M smaller steps

    Args:
        f (callable): dynamics, derivative of state
        x0 (array): initial state
        u (array): control
        DT (float): total integration time
        M (int): number of integration steps

    Returns:
        x: state at the end of the integration, at time T
    """
    dT = dt/M
    x = x0
    for _ in range(M):
        x = x + dT*f(x, u)
    
    return x