# %%
import numpy as np
from matplotlib import pyplot as plt
import os

from tools.Bspline import Bspline
from model.Tailsitter import TailsitterModel, plot_final_vs_ref
from tools.Integrator import Euler, rk4

from flights.Ascending_turn import AscenTurn_flat_output_with_derivates

# %%
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



def get_sideslip(X):
    psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]
    
    if wind is None:
        #va_xbyb_xb = -(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot + np.sin(psi)*np.sin(phi)*np.cos(theta)
        #va_xbyb_yb = -(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot + np.cos(psi)*np.sin(phi)*np.cos(theta)
        sin_beta = ((np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta))*((np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot) + np.cos(phi)*np.sin(theta)*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot)) + (np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta))*(-(np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot) - np.cos(phi)*np.sin(theta)*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot)) + np.cos(phi)*np.cos(theta)*((np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot) - (np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot)))/np.sqrt((-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot)**2 + (-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot)**2 + (-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot)**2 + 1e-6)
        
        sideslip = np.arcsin(sin_beta) #+ np.pi*((va_xbyb_yb > 0)/2 - (va_xbyb_yb < 0)/2 + (va_xbyb_yb == 0))*(va_xbyb_xb < 0)
        return sideslip
    
    wx, wy, wz = wind[0], wind[1], wind[2]
    va_xbyb_xb = -(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot + (wx*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + wy*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + wz*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) - wx
    va_xbyb_yb = -(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot + (wx*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + wy*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + wz*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) - wy
    
    sin_beta_wind = ((np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta))*((np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot) + np.cos(phi)*np.sin(theta)*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot)) + (np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta))*(-(np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot) - np.cos(phi)*np.sin(theta)*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot)) + np.cos(phi)*np.cos(theta)*((np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot) - (np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot)))/np.sqrt((-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + xdot + (wx*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + wy*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + wz*np.cos(phi)*np.cos(theta))*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) - wx)**2 + (-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + ydot + (wx*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + wy*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + wz*np.cos(phi)*np.cos(theta))*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) - wy)**2 + (-(xdot*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + ydot*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + zdot*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) + zdot + (wx*(np.cos(psi)*np.sin(theta) + np.sin(psi)*np.sin(phi)*np.cos(theta)) + wy*(np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)*np.cos(theta)) + wz*np.cos(phi)*np.cos(theta))*np.cos(phi)*np.cos(theta) - wz)**2 + 1e-6)
    
    sideslip = np.arcsin(sin_beta_wind) + np.pi*((va_xbyb_yb > 0)/2 - (va_xbyb_yb < 0)/2 + (va_xbyb_yb == 0))*(va_xbyb_xb < 0)
    return sideslip
    
def get_angle_of_attack(X):
    psi, phi, theta, xdot, ydot, zdot = X[3,:], X[4,:], X[5,:], X[6,:], X[7,:], X[8,:]

    if wind is None:
        #va_xbzb_xb = (-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot
        #va_xbzb_zb = -(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot

        sin_alpha = (-np.sin(psi)*np.cos(phi)*((np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot) + np.cos(phi)*np.sin(theta)*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot)) + np.cos(psi)*np.cos(phi)*(-(np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot) - np.cos(phi)*np.sin(theta)*((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot)) + np.sin(phi)*((np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot) - (np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot)))/np.sqrt(((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot)**2 + (-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot)**2 + (-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot)**2 + 1e-6)
        
        angle_of_attack = np.arcsin(-sin_alpha) #+ np.pi*((va_xbzb_zb > 0)/2 - (va_xbzb_zb < 0)/2 + (va_xbzb_zb == 0))*(va_xbzb_xb < 0)
        return angle_of_attack
    
    wx, wy, wz = wind[0], wind[1], wind[2]
    va_xbzb_xb = (-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot - (-wx*np.sin(psi)*np.cos(phi) + wy*np.cos(psi)*np.cos(phi) + wz*np.sin(phi))*np.sin(psi)*np.cos(phi) - wx
    va_xbzb_zb = -(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot + (-wx*np.sin(psi)*np.cos(phi) + wy*np.cos(psi)*np.cos(phi) + wz*np.sin(phi))*np.sin(phi) - wz
    
    sin_alpha_wind = (-np.sin(psi)*np.cos(phi)*((np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot) + np.cos(phi)*np.sin(theta)*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot)) + np.cos(psi)*np.cos(phi)*(-(np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot) - np.cos(phi)*np.sin(theta)*((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot)) + np.sin(phi)*((np.cos(psi)*np.cos(theta) - np.sin(psi)*np.sin(phi)*np.sin(theta))*(-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot) - (np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta))*((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot)))/np.sqrt(((-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(psi)*np.cos(phi) + xdot - (-wx*np.sin(psi)*np.cos(phi) + wy*np.cos(psi)*np.cos(phi) + wz*np.sin(phi))*np.sin(psi)*np.cos(phi) - wx)**2 + (-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.cos(psi)*np.cos(phi) + ydot + (-wx*np.sin(psi)*np.cos(phi) + wy*np.cos(psi)*np.cos(phi) + wz*np.sin(phi))*np.cos(psi)*np.cos(phi) - wy)**2 + (-(-xdot*np.sin(psi)*np.cos(phi) + ydot*np.cos(psi)*np.cos(phi) + zdot*np.sin(phi))*np.sin(phi) + zdot + (-wx*np.sin(psi)*np.cos(phi) + wy*np.cos(psi)*np.cos(phi) + wz*np.sin(phi))*np.sin(phi) - wz)**2 + 1e-6)
    
    angle_of_attack = np.arcsin(-sin_alpha_wind) + np.pi*((va_xbzb_zb > 0)/2 - (va_xbzb_zb < 0)/2 + (va_xbzb_zb == 0))*(va_xbzb_xb < 0)
    return angle_of_attack



def get_CPs(int_knots, order, Ts, N, flat_order, save, plot, x_ticks, y_ticks, z_ticks):
    kk = np.linspace(0, 1, int_knots)
    spline = Bspline(kk, order, Ts, N, flat_order=flat_order)

    CPs = spline.compute_control_points(Y.T, flat_order)

    spline_eval = spline.evaluate_spline(CPs)
    x_eval = spline_eval[:,0]
    y_eval = spline_eval[:,1]
    z_eval = spline_eval[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1, 1, 1, 1))  # RGBA where 1, 1, 1 is white, and 1 is full opacity
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.xaxis._axinfo['grid'].update(color='#DCDCDC', alpha=0.8, linewidth=0.8)
    ax.yaxis._axinfo["grid"].update(color='#DCDCDC', alpha=0.8, linewidth=0.8)
    ax.zaxis._axinfo["grid"].update(color='#DCDCDC', alpha=0.8, linewidth=0.8)
    ax.plot(x_eval[:int(N/Ts)], y_eval[:int(N/Ts)], z_eval[:int(N/Ts)], c='r', linewidth=4.0)
    ax.plot(CPs[:,0], CPs[:,1], CPs[:,2], c='b', marker='o', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4
    ax.zaxis.labelpad = 4
    ax.set_aspect('equal', 'box')

    if save:
        FN = filename + '_' + 'flatTrajectory'
        full_path = os.path.join(directory, FN)
        plt.tight_layout()
        plt.savefig(full_path, dpi=1200)
        print('Image saved')

    if plot:
        plt.tight_layout()
        plt.show()

    return CPs

# %%
Ts  = 10
N   = 350
flat_order = 4
m, k = 1, 6    

ascending_turn = True
R, Z = 50, 30

wind = None

plot_states         = True
plot_inputs         = True
plot_sideslip_AOA   = True

plot                = True
plot_OL             = True
B, H = 4, 9

save = True
filename = 'AT_N_{}__Ts_____(m,k)_({},{})'.format(N, Ts, m, k)
directory = r"_"

# %%
# Define set of B-splines
int_knots = m + 1
order = k
control_points = int_knots + order - 1

knot_vector = np.linspace(0, Ts, int_knots)
Spline = Bspline(knot_vector, order, Ts, N, flat_order=flat_order)

# %%
# Define an arbitray 3D path
# Ascending turn
if ascending_turn == True:
    Y, dY, ddY, dddY, ddddY, V = AscenTurn_flat_output_with_derivates(R, Ts, control_points)
    x_ticks = np.array([0, 10, 20, 30, 40, 50, 60])
    y_ticks = np.array([0, 20, 40, 60, 80, 100])
    z_ticks = np.array([-10, 0, 10, 20, 30, 40])


# %%
# Get the necessary derivatives from the spline which are used later on in the InverseDynamics
CPs = get_CPs(int_knots, order, Ts, N, flat_order, save, plot, x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks)
flat_output_with_derivatives = Spline.get_flat_output(CPs, 4).full() #cas.DM to np.array

y = flat_output_with_derivatives[:,:4].T
dy = flat_output_with_derivatives[:,4:8].T
ddy = flat_output_with_derivatives[:,8:12].T
dddy = flat_output_with_derivatives[:,12:16].T
ddddy = flat_output_with_derivatives[:,16:20].T

# %%
# Define the tailsitter object
TS = TailsitterModel(N)

x, u, v = TS.InverseDynamics(y, dy, ddy, dddy, ddddy)
sideslip = get_sideslip(x)
angle_of_attack = get_angle_of_attack(x)

# %%
FN = filename + '_InvDyn_'
if plot_states:
    plot_positions(Ts, N-1, x, save=save, filename=FN, directory=directory)
    plot_angles(Ts, N-1, x, save=save, filename=FN, directory=directory)
    plot_linVelocities(Ts, N-1, x, save=save, filename=FN, directory=directory)
    plot_angVelocities(Ts, N-1, x, save=save, filename=FN, directory=directory)

if plot_inputs:
    plot_wingAngles(Ts, N-1, u, save=save, filename=filename, directory=directory)
    plot_thrusts(Ts, N-1, u, save=save, filename=filename, directory=directory)

if plot_sideslip_AOA:
    plot_sideslip(Ts, N-1, sideslip, save=save, filename=FN, directory=directory)
    plot_AOA(Ts, N-1, angle_of_attack, save=save, filename=FN, directory=directory)

# %%
# OPEN-LOOP 

def open_loop(x0, u, Ts, N, integrator='Euler'):
    X_OL = np.zeros((12, N))
    X_OL[:,0] = x0

    dX_OL = np.zeros((12, N))

    for i in range(N-1):
        """Numpy implementation"""
        #dX_OL[:,i] = TS.ForwardDynamics(X_OL[:,i][:,np.newaxis], u[:,i][:,np.newaxis]).reshape(12)
        """Casadi to numpy implementation"""
        dX_OL[:,i] = TS.ForwardDynamics1D(X_OL[:,i], u[:,i]).full().reshape(12)
        
        X_OL[:,i+1]= X_OL[:,i] + Ts/N*dX_OL[:,i]

        X_OL[:,i+1]= Euler(TS.ForwardDynamics1D, X_OL[:,i], u[:,i], Ts/N).full().reshape(12,)
        X_OL[:,i+1]= rk4(TS.ForwardDynamics1D, X_OL[:,i], u[:,i], Ts/N).full().reshape(12,)

    return X_OL

x0 = x[:,0]
x_OL = open_loop(x0, u, Ts, N)
sideslip_OL = get_sideslip(x_OL)
angle_of_attack_OL = get_angle_of_attack(x_OL)


if plot_OL:
    FN = filename + '_OL_'

    plot_final_vs_ref(x_OL, control_points=CPs, views3D=[(30,-60,0)], x_ticks=x_ticks, y_ticks=y_ticks, z_ticks=z_ticks, pos=True, orient=True, labels=True, reduc=35, aspect='auto-adjusted', B=B, H=H, save=save, filename=FN, directory=directory)

    if plot_states:
        plot_positions(Ts, N-1, x_OL, save=save, filename=FN, directory=directory)
        plot_angles(Ts, N-1, x_OL, save=save, filename=FN, directory=directory)
        plot_linVelocities(Ts, N-1, x_OL, save=save, filename=FN, directory=directory)
        plot_angVelocities(Ts, N-1, x_OL, save=save, filename=FN, directory=directory)

    if plot_sideslip_AOA:
        plot_sideslip(Ts, N-1, sideslip_OL, save=save, filename=FN, directory=directory)
        plot_AOA(Ts, N-1, angle_of_attack_OL, save=save, filename=FN, directory=directory)


# %%