# %%
import numpy as np
from matplotlib import pyplot as plt
from tools.Bspline import Bspline

import os

#----------
basis   = True
spline  = True
#----------

save = False

directory = r'_'

Ts = 4
N = 500

N_spline = 1000

order = 3
num_int_knots = 5

# %%
if basis:
    for k in range(order + 1):
        degree = k
        num_control_points = num_int_knots + degree - 1

        kk = np.linspace(0, Ts, num_int_knots)
        knots = np.concatenate([np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]])

        Spline = Bspline(kk, degree, Ts, N, flat_order=0, pureBase=False)
        tau = np.linspace(0,Ts,N)
        base = Spline.evaluate_base(tau)


        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(base.shape[1]):
            if knots[i] - int(knots[i]) == 0:
                t_i = int(knots[i])
            else:
                t_i = knots[i]
            ax.plot(tau, base[:,i], label=f'$B_{{{t_i}, {degree}}}$')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

        xint = range(0, Ts+1)
        yint = np.linspace(0, 1, 5)
        plt.xticks(xint)
        plt.yticks(yint)
        plt.tight_layout()

        if save:
            filename = 'Bsplines_(k,m)_({},{})'.format(degree, num_int_knots-1)
            full_path = os.path.join(directory, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.show()

        print('-------------------------------------------------------')
        print('B-spline:        k = {}'.format(degree))
        print('             m + 1 = {}'.format(num_int_knots))
        print('             n + 1 = {}'.format(num_control_points))
        print('Extended knot vector: {}'.format(knots))
        print('-------------------------------------------------------')

# %%
if spline:
    kk = np.linspace(0, Ts, num_int_knots)
    knots = np.concatenate([np.ones(order) * kk[0], kk, np.ones(order) * kk[-1]])

    Spline = Bspline(kk, order, Ts, N_spline, flat_order=0)

    #control_points = np.array([[0,0,0],
     #                       [50,11,5],
      #                      [40,15,10],
       #                     [-10,35,2],
        #                    [0,27,14],
         #                   [10,45,8],
          #                  [30,50,12]])
    
    control_points = np.array([[0,0,0],
                            [50,0,5],
                            [40,35,-5],
                            [0,10,25],
                            [0,27,40],
                            [30,45,8],
                            [50,50,50]])

    spline_eval = Spline.evaluate_spline(control_points)
    x_eval = spline_eval[:,0]
    y_eval = spline_eval[:,1]
    z_eval = spline_eval[:,2]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_eval, y_eval, z_eval, c='b', marker='o', s=1)
    ax.plot(control_points[:,0], control_points[:,1], control_points[:,2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1, 1, 1, 1))  # RGBA where 1, 1, 1 is white, and 1 is full opacity
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))

    if save:
        filename = 'splineCurve3D'.format(order, num_int_knots-1)
        full_path = os.path.join(directory, filename)
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# %%
