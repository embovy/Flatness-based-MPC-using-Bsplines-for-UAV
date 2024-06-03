# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# %%
def barplot_processTimes(N, W, process_times_subcase1, convergence_status_subcase1, process_times_subcase2, convergence_status_subcase2, save=False, show=True, directory="", filename="", barwidth=0.8, group_separation=0.1, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16, offset=5, Subcase1='S1', Subcase2='S2'):
    time_steps = np.arange(0, N-W+1, W) # Time steps at which process time is available

    # Ensure that the lengths match
    assert len(process_times_subcase1) == len(time_steps)
    assert len(convergence_status_subcase1) == len(time_steps)
    assert len(process_times_subcase2) == len(time_steps)
    assert len(convergence_status_subcase2) == len(time_steps)

    # Define bar colors based on convergence status
    bar_colors_subcase1 = ['green' if status else 'red' for status in convergence_status_subcase1]
    bar_colors_subcase2 = ['lightgreen' if status else 'tomato' for status in convergence_status_subcase2]

    # Calculate x-positions for each bar within each time step

    x_positions_subcase1 = time_steps - barwidth/2 - group_separation/2
    x_positions_subcase2 = time_steps + barwidth/2 + group_separation/2

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))  # You can manually change the size here

    # Plot the bars
    ax.bar(x_positions_subcase1, process_times_subcase1, color=bar_colors_subcase1, width=barwidth, label=Subcase1)
    ax.bar(x_positions_subcase2, process_times_subcase2, color=bar_colors_subcase2, width=barwidth, label=Subcase2, hatch='////')


    # Set the x-axis ticks and labels
    ax.set_xticks(np.arange(0, N+1, W))
    ax.set_xticklabels(np.arange(0, N+1, W))
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)

    # Set the x and y labels
    ax.set_xlabel("Time instants", fontsize=FSaxis)
    ax.set_ylabel("Process time [s]", fontsize=FSaxis)

    # Set the x-axis limits
    ax.set_xlim(-offset, N + 1)

    # Custom legend handles
    subcase1_patch = [mpatches.Patch(color='green', label='Converged'), 
                      mpatches.Patch(color='red', label='Not converged')]
    subcase2_patch = [mpatches.Patch(facecolor='lightgreen', label='Converged', hatch='/////'),
                      mpatches.Patch(facecolor='tomato', label='Not converged', hatch='/////')]

    # Create a legend with the custom handles
    legend1 = ax.legend(handles=subcase1_patch, title="Small disturbance", bbox_to_anchor=(1, 1), loc='upper left', title_fontsize=FSaxis, fontsize=FSaxis)
    legend2 = ax.legend(handles=subcase2_patch, title="Large disturbance", bbox_to_anchor=(1, 0.5), loc='upper left', title_fontsize=FSaxis, fontsize=FSaxis)

    # Add the custom legends to the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Save the plot
    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'processTimes'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight', bbox_extra_artists=(legend1,legend2))
        print('Image saved: {}'.format(FN))

    # Display the plot
    if show:
        plt.tight_layout()
        plt.show()

# %%
save        = True
directory   = r"c:\Users\emile\Documents\2023 - 2024\THESIS\images\Bsplines\MPC\ClimbTurn2\Generation\Reference_PERTURBATION\W&P"
os.makedirs(directory, exist_ok=True)

ClimbTurn2      = True

N = 100
W = 10

offset = 5

if ClimbTurn2 and W == 10:
    process_times_smallD       = [26.48, 21.75, 20.35, 16.98, 15.88, 21.87, 14.07, 55.37, 135.27, 84.64]
    convergence_status_smallD  = 7*[True] + 3*[False]
    case1   = 'Small disturbance'

    process_times_largeD       = [26.48, 21.61, 29.90, 48.48, 29.17, 48.09, 45.25, 93.08, 154.06, 159.10]
    convergence_status_largeD  = 7*[True] + 3*[False]
    case2   = 'Large disturbance'

    filename = "ClimbTurn2" + '_W' + str(W) + '_Disturbances'
    barwidth = 2

# %%
barplot_processTimes(N=N, W=W, process_times_subcase1=process_times_smallD, convergence_status_subcase1=convergence_status_smallD, process_times_subcase2=process_times_largeD, convergence_status_subcase2=convergence_status_largeD, save=save, directory=directory, filename=filename, barwidth=barwidth, offset=offset, Subcase1=case1, Subcase2=case2)

#%%