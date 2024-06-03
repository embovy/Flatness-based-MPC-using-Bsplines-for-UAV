import casadi
import numpy as np

from tools.Integrator import Euler


class Env:
    def __init__(self) -> None:
        pass

    def discretize(self, f, dt, M=1, integrator="euler"):
        nx = self.state_space_shape[0]
        nu = self.action_space_shape[0]

        x = casadi.SX.sym("x", nx)
        u = casadi.SX.sym("u", nu)

        if integrator == "euler":
            x_new = Euler(f, x, u, dt, M=M)
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
        return self.state_space_shape[0]

    @property
    def nu(self):
        return self.action_space_shape[0]
