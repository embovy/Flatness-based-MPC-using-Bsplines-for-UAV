import casadi
import numpy as np

from optimal_control.integrator import rk4, euler


class Env:
    def __init__(self) -> None:
        pass

    def discretize(self, f, DT, M=1, integrator="rk4"):

        nx = self.state_space.shape[0]
        nu = self.action_space.shape[0]

        x = casadi.SX.sym("x", nx)
        u = casadi.SX.sym("u", nu)

        if integrator == "rk4":
            x_new = rk4(f, x, u, DT, M=M)
        elif integrator == "euler":
            x_new = euler(f, x, u, DT, M=M)
        else:
            raise Exception("integrator option not recognized")

        if isinstance(x_new, list):
            x_new = casadi.vertcat(*x_new)
        F = casadi.Function("F", [x, u], [x_new], ["x", "u"], ["x_new"])

        return F

    def step(self, u):

        x = self.state
        x_new = self.F(x, u)

        self.state = np.array(x_new).squeeze()

        return self.state

    @property
    def nx(self):
        return self.state_space.shape[0]

    @property
    def nu(self):
        return self.action_space.shape[0]
