# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# %%
def barplot_processTimes(N, W, process_times_subcase1, convergence_status_subcase1, process_times_subcase2, convergence_status_subcase2, process_times_subcase3, convergence_status_subcase3, save=False, show=True, directory="", filename="", barwidth=0.8, group_separation=0.3, legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16, offset=5, Subcase1='S1', Subcase2='S2', Subcase3='S3'):
    time_steps = np.arange(0, N-W+1, W) # Time steps at which process time is available

    # Ensure that the lengths match
    assert len(process_times_subcase1) == len(time_steps)
    assert len(convergence_status_subcase1) == len(time_steps)
    assert len(process_times_subcase2) == len(time_steps)
    assert len(convergence_status_subcase2) == len(time_steps)
    assert len(process_times_subcase3) == len(time_steps)
    assert len(convergence_status_subcase3) == len(time_steps)

    # Define bar colors based on convergence status
    bar_colors_subcase1 = ['green' if status else 'red' for status in convergence_status_subcase1]
    bar_colors_subcase2 = ['lightgreen' if status else 'tomato' for status in convergence_status_subcase2]
    bar_colors_subcase3 = ['lawngreen' if status else 'darkorange' for status in convergence_status_subcase3]

    # Calculate x-positions for each bar within each time step

    x_positions_subcase1 = time_steps - barwidth - group_separation/3
    x_positions_subcase2 = time_steps 
    x_positions_subcase3 = time_steps + barwidth + group_separation/3

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))  # You can manually change the size here

    # Plot the bars
    ax.bar(x_positions_subcase1, process_times_subcase1, color=bar_colors_subcase1, width=barwidth, label=Subcase1)
    ax.bar(x_positions_subcase2, process_times_subcase2, color=bar_colors_subcase2, width=barwidth, label=Subcase2, hatch='//')
    ax.bar(x_positions_subcase3, process_times_subcase3, color=bar_colors_subcase3, width=barwidth, label=Subcase3, hatch='.')

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
    subcase2_patch = [mpatches.Patch(facecolor='lightgreen', label='Converged', hatch='//'),
                      mpatches.Patch(facecolor='tomato', label='Not converged', hatch='//')]
    subcase3_patch = [mpatches.Patch(facecolor='lawngreen', label='Converged', hatch='.'),
                      mpatches.Patch(facecolor='darkorange', label='Not converged', hatch='.')]

    # Create a legend with the custom handles
    legend1 = ax.legend(handles=subcase1_patch, title="P = 10", bbox_to_anchor=(1.05, 1.00), loc='upper left', title_fontsize=FSaxis, fontsize=FSaxis)
    legend2 = ax.legend(handles=subcase2_patch, title="P = 20", bbox_to_anchor=(1.05, 0.65), loc='upper left', title_fontsize=FSaxis, fontsize=FSaxis)
    legend3 = ax.legend(handles=subcase3_patch, title="P = 30", bbox_to_anchor=(1.05, 0.30), loc='upper left', title_fontsize=FSaxis, fontsize=FSaxis)

    # Add the custom legends to the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)

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
directory   = r"c:\Users\emile\Documents\2023 - 2024\THESIS\images\Bsplines\MPC\ClimbTurn2\Tracking\Reference_PERTURBATION\W&P\(SP,SA)_(2,0.30)"
os.makedirs(directory, exist_ok=True)

ClimbTurn2      = True
N = 100
W = 10

CPs = 'fixed'

offset = 5

if ClimbTurn2:
    process_times1       = [26.48, 29.23, 10.73, 12.07, 10.16, 11.97, 20.08, 65.70, 103.33, 67.93]
    convergence_status1  = 7*[True] + 3*[False]

    process_times2       = [26.48, 11.98, 17.56, 12.91, 9.41, 16.97, 16.07, 41.27, 64.68, 41.14]
    convergence_status2  = 7*[True] + 3*[False]

    process_times3       = [26.48, 13.20, 13.28, 13.80, 14.94, 13.85, 71.63, 145.76, 140.35, 125.30]
    convergence_status3  = 6*[True] + 4*[False]

    filename = "ClimbTurn2" + '_P' + '_10_20_30_'
    barwidth = 2

# %%
barplot_processTimes(N=N, W=W, process_times_subcase1=process_times1, convergence_status_subcase1=convergence_status1, process_times_subcase2=process_times2, convergence_status_subcase2=convergence_status2, process_times_subcase3=process_times3, convergence_status_subcase3=convergence_status3, save=save, directory=directory, filename=filename, barwidth=barwidth, offset=offset)

#%%