# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
def barplot_processTimes(N, W, process_times, convergence_status, save=False, show=True, directory="", filename="", barwidth=0.8, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16, offset=5):
    time_steps = np.arange(0, N-W+1, W) # Time steps at which process time is available

    # Ensure that the lengths match
    assert len(process_times) == len(time_steps)
    assert len(convergence_status) == len(time_steps)

    # Define bar colors based on convergence status
    bar_colors = ['green' if status else 'red' for status in convergence_status]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))  # You can manually change the size here
    #fig, ax = plt.subplots(figsize=(15, 4))

    # Plot the bars
    ax.bar(time_steps, process_times, color=bar_colors, width=barwidth)

    # Set the x-axis ticks and labels
    ax.set_xticks(np.arange(0, N+1, W))
    ax.set_xticklabels(np.arange(0, N+1, W))
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)

    # Set the x and y labels
    ax.set_xlabel("Time instant", fontsize=FSaxis)
    ax.set_ylabel("Process time [s]", fontsize=FSaxis)

    # Set the x-axis limits
    ax.set_xlim(-offset, N + 1)

    # Save the plot
    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'processTimes'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    # Display the plot
    if show:
        plt.tight_layout()
        plt.show()

# %%
save        = True
directory   = r"c:\Users\emile\Documents\2023 - 2024\THESIS\images\Bsplines\MPC\ClimbTurn2\Generation\Reference_NoPerturbation\W&P"
os.makedirs(directory, exist_ok=True)

ClimbTurn2      = True
N = 100

W = 20

CPs = 'fixed'

extra = 'ExtAbs'

offset = 5

if ClimbTurn2 and W == 5 and CPs == 'fixed':
    process_times       = [26.48, 25.21, 22.18, 20.04, 20.97, 19.22, 18.84, 19.56, 23.33, 21.33, 24.54, 32.84, 123.44, 125.01, 116.24, 96.53, 129.98, 128.60, 95.54, 109.82]
    convergence_status  = 14*[True] + 6*[False]

    filename = "ClimbTurn2" + '_W' + str(W) + extra
    barwidth = 2

if ClimbTurn2 and W == 10 and CPs == 'fixed':
    process_times       = [26.48, 220.82, 17.58, 17.67, 18.54, 17.94, 82.79, 156.58, 136.27, 71.40]
    convergence_status  = 6*[True] + 4*[False]

    filename = "ClimbTurn2" + '_W' + str(W) + extra
    barwidth = 4

if ClimbTurn2 and W == 20 and CPs == 'fixed':
    process_times       = [26.48, 20.56, 16.07, 81.84, 88.45]
    convergence_status  = 3*[True] + 2*[False]

    filename = "ClimbTurn2" + '_W' + str(W) + extra
    barwidth = 8



if ClimbTurn2 and W == 10 and CPs == 'change':
    process_times       = [26.48, 24.08, 22.43, 21.87, 164.60, 131.22, 119.46, 105.96, 58.40, 61.44]
    convergence_status  = 4*[True] + 6*[False]

    filename = "ClimbTurn2" + '_W' + str(W) + '_' + 'CPs_change'
    barwidth = 4

# %%
barplot_processTimes(N=N, W=W, process_times=process_times, convergence_status=convergence_status, save=save, directory=directory, filename=filename, barwidth=barwidth, offset=offset)

#%%