import numpy as np
import casadi as cd
import tools.spline as spl
from matplotlib import pyplot as plt

class Bspline:
    def __init__(self, kk, order, Ts, N, flat_order, pureBase=False):
        """
        kk (array_like):    internal knot vector [t0,...,tm]    -> shape(1; n+1 - k + 1)
        order (int):        order k of the spline   
        Ts (float):         time horizon on which the spline is evaluated (DIFFERENT FROM INTERNAL KNOT VECTOR!)
        N (int):            number of intervals this time horizon is divided in
        """
        # Extended knot vector to ensure the spline is clamped at start- and endpoint
        if pureBase == True:
            knots = kk
        else:
            knots = np.concatenate([np.ones(order) * kk[0], kk, np.ones(order) * kk[-1]])

        # Define B-spline basis
        self.base = spl.BSplineBasis(knots, order)

        # Define time horzion (actually already given by internal knot vector kk, but here with other stepnumber N)
        self.Ts = Ts
        self.N = N                        
        self.tau = np.linspace(0, Ts, N)  

        self.flat_order = flat_order  

        # Number of internal knots (m+1); number of control points (n+1)
        nn = len(kk)  
        self.K = nn + order - 1 

        self.degree = order 

        #print('----------------')
        #print('m + 1 = {}'.format(nn))
        #print('n + 1 = {}'.format(self.K))
        #print('k = {}'.format(order))
        #print('----------------')

    def plot_basis(self):
        """
        Plot the B-splines of order k on the unit interval
        """
        plt.plot(cd.densify(self.base(np.linspace(0, 1))))
        plt.grid()
        plt.show()

    def gen_curve(self, coeff):
        """
        Given the control points (== coeff), we create the curve using the basis we previously constructed
        """
        curve = spl.BSpline(self.base, coeff)

        return curve

    def evaluate_spline(self, coeff, tau=None):
        """
        Evaluate over time horizon tau with given control points (== coeff)
        """
        curve = self.gen_curve(coeff)
        if tau is None:
            tau = self.tau

        return curve(tau)

    def get_flat_output(self, coeff, order, N=None, tau=None):
        """
        Gives the flat output and its derivatives up till the given order, based on the given control points (== coeff)
        """
        if (tau is None) and (N is not None):
            tau = np.linspace(0, self.Ts, int(N))
        elif (tau is None) and (N is None):
            tau = self.tau

        curve = self.gen_curve(coeff)
        s = curve(tau)

        try:
            dcurve = [curve.derivative(o=o) for o in range(1, order + 1)]
            ds = [dcurve_k(tau) for dcurve_k in dcurve]
            #print('Type ds[0], shape [TRY] : {}, {}'.format(type(ds[0]), ds[0].shape))
        except:
            dcurve = []
            
            if type(coeff) == np.ndarray:
                null = 1e-8*np.ones((N, self.flat_order))
            else:
                null = 1e-8*cd.MX.ones(N, self.flat_order)

            for _ in range(1, order + 1):
                dcurve.append(null)
            ds = dcurve
            #print('Type ds[0], shape [EXCEPT]: {}, {}'.format(type(ds[0]), ds[0].shape))

        flat_output = cd.horzcat(s, *ds)

        #print('----------------------------------------------')
        #print('Bspline.py: flat_output : {}'.format(flat_output.shape))
        #print(type(flat_output))
        #print('----------------------------------------------')

        return flat_output

    def compute_control_points(self, Y, ny):
        """
        Compute the control points of flat trajectory Y with shape(nr control points, flat output) = shape(n+1, ny)
        """
        #print(Y.shape, self.K)
        assert Y.shape == (self.K, ny)

        A = self.evaluate_base(np.linspace(0, 1, num=self.K))  # shape=(K,K)
        return cd.inv(A) @ Y

    def evaluate_base(self, tau):
        return cd.densify(self.base(tau))
