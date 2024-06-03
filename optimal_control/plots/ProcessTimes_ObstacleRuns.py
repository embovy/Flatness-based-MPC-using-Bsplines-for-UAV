# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
def barplot_processTimes(situations, conditions, processTimes, save=False, show=True, directory="", filename="", legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    data = {
        "Case": situations,
        "Subcase": conditions,
        "Process_Time": processTimes  
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Define cases and subcases
    cases = df["Case"].unique()
    subcases = df["Subcase"].unique()

    # Define colors for subcases
    colors = {"Full": "red", "Full + slack": "blue", "Partly": "green", "Partly + slack": "orange"}

    # Define bar width and separations
    bar_width = 0.5  
    subcase_separation = 0.1  
    case_separation = 1.5  #

    # Calculate x-positions for each subcase within each case
    x_positions = []
    case_position = 0  

    for i, case in enumerate(cases):
        if i > 0:
            case_position += case_separation  # Additional space between groups
        
        # Generate positions for each subcase within the current case
        subcase_positions = [case_position + j * (bar_width + subcase_separation) for j in range(len(subcases))]
        
        x_positions.extend(subcase_positions)  
        case_position = subcase_positions[-1] 

    # Plot bars for each subcase with proper positions
    fig, ax = plt.subplots()
    ax.grid(axis='y')
    fig.set_size_inches(10, 6)
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_ylabel("Average process time [s]", fontsize=FSaxis)

    for i, subcase in enumerate(subcases):
        subcase_data = df[df["Subcase"] == subcase]
        bar_positions = [x_positions[i + j * len(subcases)] for j in range(len(cases))]

        ax.bar(bar_positions, subcase_data["Process_Time"], color=colors[subcase], width=bar_width, label=subcase)

    # Calculate the midpoint for each case for label positioning
    midpoints = [(x_positions[i * len(subcases) + 1] + x_positions[i * len(subcases) + 2]) / 2 for i in range(len(cases))]

    # Set the x-axis ticks and labels for the main cases
    ax.set_xticks(midpoints)
    ax.set_xticklabels(cases)

    # Add legend
    ax.legend(title="Final boundary condition", fontsize=fontsize, title_fontsize=fontsize)

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

def barplot_iterations(situations, conditions, iterations, save=False, show=True, directory="", filename="", legendLoc='center left', bbox=(1, 0.5), fontsize=16, FSaxis=16):
    data = {
        "Case": situations,
        "Subcase": conditions,
        "Iteration": iterations  
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Define cases and subcases
    cases = df["Case"].unique()
    subcases = df["Subcase"].unique()

    # Define colors for subcases
    colors = {"Full": "red", "Full + slack": "blue", "Partly": "green", "Partly + slack": "orange"}

    # Define bar width and separations
    bar_width = 0.5  
    subcase_separation = 0.1  
    case_separation = 1.5  #

    # Calculate x-positions for each subcase within each case
    x_positions = []
    case_position = 0  

    for i, case in enumerate(cases):
        if i > 0:
            case_position += case_separation  # Additional space between groups
        
        # Generate positions for each subcase within the current case
        subcase_positions = [case_position + j * (bar_width + subcase_separation) for j in range(len(subcases))]
        
        x_positions.extend(subcase_positions)  
        case_position = subcase_positions[-1] 

    # Plot bars for each subcase with proper positions
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=FSaxis)
    ax.set_ylabel("Iterations", fontsize=FSaxis)

    for i, subcase in enumerate(subcases):
        subcase_data = df[df["Subcase"] == subcase]
        bar_positions = [x_positions[i + j * len(subcases)] for j in range(len(cases))]

        ax.bar(bar_positions, subcase_data["Iteration"], color=colors[subcase], width=bar_width, label=subcase)

    # Calculate the midpoint for each case for label positioning
    midpoints = [(x_positions[i * len(subcases) + 1] + x_positions[i * len(subcases) + 2]) / 2 for i in range(len(cases))]

    # Set the x-axis ticks and labels for the main cases
    ax.set_xticks(midpoints)
    ax.set_xticklabels(cases)

    # Add legend
    ax.legend(title="Final boundary condition", fontsize=fontsize, title_fontsize=fontsize)
   
    # Save the plot
    if save:
        if directory.strip() == "":
            directory = os.getcwd()
        FN = filename + '_' + 'iterations'
        full_path = os.path.join(directory, FN)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print('Image saved: {}'.format(FN))

    # Display the plot
    if show:
        plt.tight_layout()
        plt.show()

# %%
save        = True
directory   = r"c:\Users\emile\Documents\2023 - 2024\THESIS\images\Bsplines\ClimbTurn2"
os.makedirs(directory, exist_ok=True)

AscendingTurn   = False
ClimbTurn2      = True

# Sample data with process times for each case and subcase
situations = 4*["Reference"] + 4*["Case 1"] + 4*["Case 2"] + 4*["Case 3"] + 4*["Case 4"]
conditions = 5*["Full", "Full + slack", "Partly", "Partly + slack"] 

if AscendingTurn:
    ref = [19, 19, 17, 19]
    S1 = [23, 28, 26, 27]
    S1S2 = [104, 57, 42, 104]
    S1S2S3 = [166, 155, 92, 146]
    S1S2S3_C1 = [90, 104, 277, 273]
    processTimes = ref + S1 + S1S2 + S1S2S3 + S1S2S3_C1

    ref = [14, 14, 13, 14]
    S1 = [17, 21, 19, 19]
    S1S2 = [78, 39, 30, 67]
    S1S2S3 = [120, 108, 68, 103]
    S1S2S3_C1 = [65, 78, 204, 206]
    iterations = ref + S1 + S1S2 + S1S2S3 + S1S2S3_C1

    filename = "AscendingTurn"

if ClimbTurn2:
    ref = [25, 29, 32, 30]
    S1 = [50, 54, 39, 46]
    S1S2 = [50, 53, 38, 44]
    S1S2S3 = [184, 84, 48, 56]
    S1S2S3_C1 = [495, 98, 65, 68]
    processTimes = ref + S1 + S1S2 + S1S2S3 + S1S2S3_C1

    ref = [19, 19, 19, 19]
    S1 = [37, 40, 29, 30]
    S1S2 = [38, 37, 28, 29]
    S1S2S3 = [133, 56, 33, 42]
    S1S2S3_C1 = [323, 70, 49, 49]
    iterations = ref + S1 + S1S2 + S1S2S3 + S1S2S3_C1

    filename = "ClimbTurn2"


# %%
barplot_processTimes(situations, conditions, processTimes, save=save, directory=directory, filename=filename)
barplot_iterations(situations, conditions, iterations, save=save, directory=directory, filename=filename)

#%%