import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import copy

#=========================================================#
#                  SETUP VISUALIZATION                    #
#=========================================================#
def build_fig(x_label='number of function evaluations',
    y_label=r'average objective function value $\bar{f_\Theta}$',
    xLim=None,yLim=None):
    
    # Comment to disable latex like labels
    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(7,5))
    fig.subplots_adjust()

    ax.set_xlabel(x_label,fontsize=18)
    ax.set_ylabel(y_label,fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    if xLim is not None:
        ax.set_xlim(xLim)
    
    if yLim is not None:
        ax.set_ylim(yLim)

    return fig,ax

#=========================================================#
#                    CONSTRUCT LEGEND                     #
#=========================================================#
def build_legend(fig,ax,labels,palette,styles=None,location='upper right'):

    # Legend
    import matplotlib.lines as mlines

    handles = []; i = 0
    for label in labels:
        if styles is not None:
            style = styles[i]
        else:
            style ="-"

        handle = mlines.Line2D([], [], color=palette[i], marker='', markersize=5, linestyle=style)
        handles += [handle]
        i += 1

    lx = ax.legend(handles, labels, loc=location, fontsize = 11, )

    # # Get the bounding box of the original legend
    # bb = lx.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    # # Change to location of the legend. 
    # xOffset = -0.14; yOffset = -0.15
    # bb.x0 += xOffset; bb.x1 += xOffset # Move anchor point x0 and x1 to gether
    # bb.y0 += yOffset; bb.y1 += yOffset # Move anchor point y0 and y1 to gether
    # lx.set_bbox_to_anchor(bb, transform = ax.transAxes)

#=========================================================#
#          PROCESS RESULTS OF SINGLE ALGORITHM            #
#=========================================================#
def process_group(axs,group_indices,algo,algo_dict,palette,labels,i,data_dict,best_x,style='-',limit=6000,plot=True):

    # Group by "group" index
    for group in group_indices:
        group_keys = [key for key,value in algo_dict["stats"].items() if value['group'] == group]
        group_values = [value for key,value in algo_dict["stats"].items() if value['group'] == group]

        # algorithmic parameters labels
        algo_params = [group_values[0][p] for p in algo["algo_params"]]

        # Average results over a group
        columns = ['mean_f','min_f','max_f','mean_c','min_c','max_c','mean_p','min_p','max_p']
        df_group = pd.DataFrame(index=range(11000), columns=columns) # create an empty data frame

        df_true_f = pd.Series(dtype=np.float64); df_true_c = pd.Series(dtype=np.float64); df_true_p = pd.Series(dtype=np.float64); 
        for key,value in zip(group_keys,group_values):
            index = value["index"]
            n_k = value["n_k"]

            run_folder = r'opt_data/%s/Run_%i/' %(algo["folder"],index)\
            
            f_filename = run_folder + "f_hist_%s.txt" %(algo["file_suffix"])
            f_true = run_folder + "f_progress_%s.txt" %(algo["file_suffix"])

            data = pd.read_csv(f_filename, sep=",") 
            data_true = pd.read_csv(f_true, sep=",") 

            map = np.searchsorted(data_true['bbe'], data['bbe'])
            data['true_f'] = [data_true['f'][x-1] for x in map]
            data['true_cstr'] = [data_true['cstr'][x-1] for x in map]
            data['true_p_value'] = [data_true['p_value'][x-1] for x in map]
            
            # Compute norm with respect to best known solution
            x_keys = ['x1','x2','x3']
            x_diffs = data[x_keys].sub(best_x)
            data['distance'] = np.sqrt(np.square(x_diffs).sum(axis=1))

            data_dict[key] = data # store dataframe in data dictionary

            df_true_f = pd.concat([df_true_f,data['true_f']], ignore_index=True, axis=1)
            df_true_c = pd.concat([df_true_c,data['true_cstr']], ignore_index=True, axis=1)
            df_true_p = pd.concat([df_true_p,data['true_p_value']], ignore_index=True, axis=1)

        df_group['mean_f'] = df_true_f.mean(axis=1)
        df_group['min_f'] = df_true_f.min(axis=1)
        df_group['max_f'] = df_true_f.max(axis=1)

        df_group['mean_c'] = df_true_c.mean(axis=1)
        df_group['min_c'] = df_true_c.min(axis=1)
        df_group['max_c'] = df_true_c.max(axis=1)

        df_group['mean_p'] = df_true_p.mean(axis=1)
        df_group['min_p'] = df_true_p.min(axis=1)
        df_group['max_p'] = df_true_p.max(axis=1)

        if plot:
            axs[0].plot(df_group.index[:limit],df_group['mean_f'][:limit],color=palette[i],linestyle=style)
            # axs[0].fill_between(df_group.index,df_group['min_f'], df_group['max_f'], color=palette[i], alpha=0.2)
            axs[1].plot(df_group.index[:limit],df_group['mean_c'][:limit],color=palette[i],linestyle=style)
            # axs[1].fill_between(df_group.index,df_group['min_p'], df_group['max_p'], color=palette[i], alpha=0.2)

        legend_string = algo["legend_title"]
        for param_label,param_value in zip(algo["legend_labels"],algo_params):
            legend_string += param_label %(param_value)

        labels += [legend_string]
        i += 1

    return i

#=========================================================#
#                          MAIN                           #
#=========================================================#
if __name__ == "__main__":

    #==========================================================
    # Experiment stats

    StoMADS_config = {
        "default" : {
            "Delta" : 1,
            "gamma" : 5,
            "tau" : 0.5,
            "OPPORTUNISTIC_EVAL" : True,
        },
    }

    NOMAD_config = {
        "default" : {
            "ANISOTROPY_FACTOR" : 0.1,
            "ANISOTROPIC_MESH" : True,
            "DIRECTION_TYPE" : "ORTHO_NP1_NEG",
            "MODEL_SEARCH" : True,
            "NM_SEARCH" : True,
            "OPPORTUNISTIC_EVAL" : True,
        },
        "basic" : {
            "ANISOTROPY_FACTOR" : 0.1,
            "ANISOTROPIC_MESH" : False,
            "DIRECTION_TYPE" : "ORTHO_2N",
            "MODEL_SEARCH" : False,
            "NM_SEARCH" : False,
            "OPPORTUNISTIC_EVAL" : True,
        },
    }

    GA_config = {
        "default" : {
            "FunctionTolerance" : 1e-6,
            "ConstraintTolerance" : 1e-3,
            "NonlinearConstraintAlgorithm" : "auglag",
            "CrossoverFraction" : 0.8,
            "MutationFcn" : {"mutationadaptfeasible" : {"scale" : 1.0, "shrink": 1.0}},
        },
        "custom" : {
            "FunctionTolerance" : 1e-6,
            "ConstraintTolerance" : 0,
            "NonlinearConstraintAlgorithm" : "auglag",
            "CrossoverFraction" : 0.1,
            "MutationFcn" : {"mutationadaptfeasible" : {"scale" : 1.0, "shrink": 0.5}},
        },
    }

    StoMADS_dict = {
        "folder" : "StoMADS",
        "stats" : {
            0   : { "index" : 4,    "n_k" : 1,    "epsilon_f" : 0.09, "config" : "default", "n_cores" : 8, "group" : 0}, # best ignore
            1   : { "index" : 3,    "n_k" : 4,    "epsilon_f" : 0.09, "config" : "default", "n_cores" : 4, "group" : 0}, # best ignore
            2   : { "index" : 1,    "n_k" : 20,   "epsilon_f" : 0.09, "config" : "default", "n_cores" : 8, "group" : 0}, # best ignore
            3   : { "index" : 2,    "n_k" : 20,   "epsilon_f" : 0.09, "config" : "default", "n_cores" : 8, "group" : 0}, # best ignore
            4   : { "index" : 6,    "n_k" : 4,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 4, "group" : 0}, # best ignore
            5   : { "index" : 9,    "n_k" : 1,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 4, "group" : 1},
            6   : { "index" : 27,   "n_k" : 1,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 1},
            7   : { "index" : 28,   "n_k" : 1,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 1},
            8   : { "index" : 29,   "n_k" : 1,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 1},
            9   : { "index" : 5,    "n_k" : 4,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 2},
            10  : { "index" : 30,   "n_k" : 4,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 2},
            11  : { "index" : 31,   "n_k" : 4,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 2},
            12  : { "index" : 32,   "n_k" : 4,    "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 2},
            13  : { "index" : 7,    "n_k" : 20,   "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 3},
            14  : { "index" : 33,   "n_k" : 20,   "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 3},
            15  : { "index" : 34,   "n_k" : 20,   "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 3},
            16  : { "index" : 35,   "n_k" : 20,   "epsilon_f" : 0.01, "config" : "default", "n_cores" : 8, "group" : 3},
            17  : { "index" : 14,   "n_k" : 1,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 4},
            18  : { "index" : 21,   "n_k" : 1,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 4},
            19  : { "index" : 22,   "n_k" : 1,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 4},
            20  : { "index" : 23,   "n_k" : 1,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 4},
            21  : { "index" : 12,   "n_k" : 4,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 5},
            22  : { "index" : 15,   "n_k" : 4,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 5},
            23  : { "index" : 16,   "n_k" : 4,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 5},
            24  : { "index" : 17,   "n_k" : 4,    "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 5},
            25  : { "index" : 13,   "n_k" : 20,   "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 6},
            26  : { "index" : 18,   "n_k" : 20,   "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 6},
            27  : { "index" : 19,   "n_k" : 20,   "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 6},
            28  : { "index" : 20,   "n_k" : 20,   "epsilon_f" : 0.1,  "config" : "default", "n_cores" : 8, "group" : 6},
            29  : { "index" : 10,   "n_k" : 1,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 7},
            30  : { "index" : 24,   "n_k" : 1,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 7},
            31  : { "index" : 25,   "n_k" : 1,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 7},
            32  : { "index" : 26,   "n_k" : 1,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 7},
            33  : { "index" : 8,    "n_k" : 4,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 8},
            34  : { "index" : 36,   "n_k" : 4,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 8},
            35  : { "index" : 37,   "n_k" : 4,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 8},
            36  : { "index" : 38,   "n_k" : 4,    "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 8},
            37  : { "index" : 11,   "n_k" : 20,   "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 9},
            38  : { "index" : 39,   "n_k" : 20,   "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 9},
            39  : { "index" : 40,   "n_k" : 20,   "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 9},
            40  : { "index" : 41,   "n_k" : 20,   "epsilon_f" : 0.2,  "config" : "default", "n_cores" : 8, "group" : 9},
        },
    }

    NOMAD_dict = {
        "folder" : "NOMAD",
        "stats" : {
            0   : { "index" : 1,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            1   : { "index" : 2,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            2   : { "index" : 3,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            3   : { "index" : 4,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            4   : { "index" : 5,    "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-6,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            5   : { "index" : 6,    "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-6,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0},
            6   : { "index" : 7,    "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 8, "group" : 1},
            7   : { "index" : 16,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 8, "group" : 1},
            8   : { "index" : 17,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 8, "group" : 1},
            9   : { "index" : 18,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 8, "group" : 1},
            10  : { "index" : 8,    "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 4, "group" : 2},
            11  : { "index" : 13,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 4, "group" : 2},
            12  : { "index" : 14,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 4, "group" : 2},
            13  : { "index" : 15,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "default",   "n_cores" : 4, "group" : 2},
            14  : { "index" : 9,    "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 3},
            15  : { "index" : 10,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 3},
            16  : { "index" : 11,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 3},
            17  : { "index" : 12,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 3},
            18  : { "index" : 19,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4},
            19  : { "index" : 20,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4},
            20  : { "index" : 21,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4},
            21  : { "index" : 22,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4},
            22  : { "index" : 27,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5},
            23  : { "index" : 28,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5},
            24  : { "index" : 29,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5},
            25  : { "index" : 30,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5},
            26  : { "index" : 23,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6},
            27  : { "index" : 24,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6},
            28  : { "index" : 25,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6},
            29  : { "index" : 26,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6},
        },
    }

    GA_dict = {
        "folder" : "GA",
        "stats" : {
            0   : { "index" : 1,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 1}, # running
            1   : { "index" : 2,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 1}, # running
            2   : { "index" : 3,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 1}, # running
            3   : { "index" : 4,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 1}, # running
            4   : { "index" : 5,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            5   : { "index" : 6,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            6   : { "index" : 7,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            7   : { "index" : 8,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            8   : { "index" : 9,    "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            9   : { "index" : 10,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            10  : { "index" : 11,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            11  : { "index" : 12,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            12  : { "index" : 13,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4}, # running
            13  : { "index" : 14,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4}, # running
            14  : { "index" : 15,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4}, # running
            15  : { "index" : 16,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4}, # running
            16  : { "index" : 17,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5}, # running
            17  : { "index" : 18,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5}, # running
            18  : { "index" : 19,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5}, # running
            19  : { "index" : 20,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5}, # running
            20  : { "index" : 21,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            21  : { "index" : 22,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            22  : { "index" : 23,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            23  : { "index" : 24,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            24  : { "index" : 25,   "n_k" : 4,  "n_population" : 50,    "R_initial" : 5.0,      "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 0},
            25  : { "index" : 26,   "n_k" : 4,  "n_population" : 50,    "R_initial" : 2.0,      "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 0},
        },
    }

    # Initialize data dictionaries (dictionaries are mutable)
    StoMADS_data_dict = copy.deepcopy(StoMADS_dict["stats"])
    for key in StoMADS_dict["stats"].keys():
        StoMADS_data_dict[key] = ''

    # Initialize data dictionaries (dictionaries are mutable)
    NOMAD_data_dict = copy.deepcopy(NOMAD_dict["stats"])
    for key in NOMAD_dict["stats"].keys():
        NOMAD_data_dict[key] = ''

    #==========================================================
    # Setup color palette (fixed random seed)

    r = lambda: random.randint(0,255)
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(20):
        # random.seed(i)
        palette += ['#%02X%02X%02X' % (r(),r(),r())]

    styles = ['-','-','-','--','--',]
    # xlims = [0, 3250]
    # ylims1 = [-6.5, -0.6]
    # ylims2 = [-60, 800]
    xlims = None
    ylims1 = None
    ylims2 = None
    plot = False
    load_results = True

    # best known result
    best_x = np.array([[1, 0.31, 0.94],])
    # best_x = np.array([[0.75674921, 0.417047454, 0.971785993],])

    if not load_results:
        #=========================================================#
        #                        n_k = 1                          #
        #=========================================================#

        labels = [] # mutable
        i = 0 # immutable

        # Setup figures
        x_label = 'number of function evaluations'
        y_label1 = r'average objective function value $\bar{f_\Theta}$'
        y_label2 = r'average constraint function value $\bar{g_\Theta}$'
        # y_label2 = r'Probability of satisfying constraint $\bar{P}$'

        if plot:
            fig1,ax1 = build_fig(x_label,y_label1,xLim=xlims,yLim=ylims1)
            fig2,ax2 = build_fig(x_label,y_label2,xLim=xlims,yLim=ylims2)

            figs = [fig1,fig2]; axs = [ax1,ax2]
        else:
            figs = None; axs = None

        #-------------------- STOMADS RESULTS --------------------#

        group_indices = [1,4,7] # groups for n_k = 1

        algo = {
            "folder" : StoMADS_dict["folder"],
            "file_suffix" : "STOMADS34",
            "legend_title" : r'StoMADS-PB ',
            "legend_labels" : [r'$\epsilon_f=%.2f$',],
            "algo_params" : ["epsilon_f",]
        }

        i = process_group(axs,group_indices,algo,StoMADS_dict,palette,labels,i,StoMADS_data_dict,best_x,style='-',plot=plot)

        #-------------------- NOMAD RESULTS --------------------#

        group_indices = [1,4] # groups for n_k = 1

        algo = {
            "folder" : NOMAD_dict["folder"],
            "file_suffix" : "NOMAD",
            "legend_title" : r'NOMAD-',
            "legend_labels" : [r'%s',],
            "algo_params" : ["config",]
        }

        i = process_group(axs,group_indices,algo,NOMAD_dict,palette,labels,i,NOMAD_data_dict,best_x,style='--',plot=plot)

        if plot:
            build_legend(figs[0],axs[0],labels,palette,styles=styles,location='upper right')
            build_legend(figs[1],axs[1],labels,palette,styles=styles,location='lower right')

            figs[0].savefig('opt_data/f_nk=1.pdf', format='pdf', dpi=200)
            figs[1].savefig('opt_data/g_nk=1.pdf', format='pdf', dpi=200)

            plt.show()

        #=========================================================#
        #                        n_k = 4                          #
        #=========================================================#

        labels = [] # mutable
        i = 0 # immutable

        # Setup figures
        if plot:
            fig1,ax1 = build_fig(x_label,y_label1,xLim=xlims,yLim=ylims1)
            fig2,ax2 = build_fig(x_label,y_label2,xLim=xlims,yLim=ylims2)

            figs = [fig1,fig2]; axs = [ax1,ax2]
        else:
            figs = None; axs = None

        #-------------------- STOMADS RESULTS --------------------#

        group_indices = [2,5,8] # groups for n_k = 4

        algo = {
            "folder" : StoMADS_dict["folder"],
            "file_suffix" : "STOMADS34",
            "legend_title" : r'StoMADS-PB ',
            "legend_labels" : [r'$\epsilon_f=%.2f$',],
            "algo_params" : ["epsilon_f",]
        }

        i = process_group(axs,group_indices,algo,StoMADS_dict,palette,labels,i,StoMADS_data_dict,best_x,style='-',plot=plot)

        #-------------------- NOMAD RESULTS --------------------#

        group_indices = [2,5] # groups for n_k = 4

        algo = {
            "folder" : NOMAD_dict["folder"],
            "file_suffix" : "NOMAD",
            "legend_title" : r'NOMAD-',
            "legend_labels" : [r'%s',],
            "algo_params" : ["config",]
        }

        i = process_group(axs,group_indices,algo,NOMAD_dict,palette,labels,i,NOMAD_data_dict,best_x,style='--',plot=plot)

        if plot:
            build_legend(figs[0],axs[0],labels,palette,styles=styles,location='upper right')
            build_legend(figs[1],axs[1],labels,palette,styles=styles,location='lower right')

            figs[0].savefig('opt_data/f_nk=4.pdf', format='pdf', dpi=200)
            figs[1].savefig('opt_data/g_nk=4.pdf', format='pdf', dpi=200)

            plt.show()

        #=========================================================#
        #                       n_k = 20                          #
        #=========================================================#

        labels = [] # mutable
        i = 0 # immutable

        # Setup figures
        if plot:
            fig1,ax1 = build_fig(x_label,y_label1,xLim=xlims,yLim=ylims1)
            fig2,ax2 = build_fig(x_label,y_label2,xLim=xlims,yLim=ylims2)

            figs = [fig1,fig2]; axs = [ax1,ax2]
        else:
            figs = None; axs = None

        #-------------------- STOMADS RESULTS --------------------#

        group_indices = [3,6,9] # groups for n_k = 20

        algo = {
            "folder" : StoMADS_dict["folder"],
            "file_suffix" : "STOMADS34",
            "legend_title" : r'StoMADS-PB ',
            "legend_labels" : [r'$\epsilon_f=%.2f$',],
            "algo_params" : ["epsilon_f",]
        }

        i = process_group(axs,group_indices,algo,StoMADS_dict,palette,labels,i,StoMADS_data_dict,best_x,style='-',plot=plot)

        #-------------------- NOMAD RESULTS --------------------#

        group_indices = [3,6] # groups for n_k = 20

        algo = {
            "folder" : NOMAD_dict["folder"],
            "file_suffix" : "NOMAD",
            "legend_title" : r'NOMAD-',
            "legend_labels" : [r'%s',],
            "algo_params" : ["config",]
        }

        i = process_group(axs,group_indices,algo,NOMAD_dict,palette,labels,i,NOMAD_data_dict,best_x,style='--',plot=plot)

        if plot:
            build_legend(figs[0],axs[0],labels,palette,styles=styles,location='upper right')
            build_legend(figs[1],axs[1],labels,palette,styles=styles,location='lower right')

            figs[0].savefig('opt_data/f_nk=20.pdf', format='pdf', dpi=200)
            figs[1].savefig('opt_data/g_nk=20.pdf', format='pdf', dpi=200)

            plt.show()

        with open('opt_data/data_dicts.pkl', 'wb') as file:
            pickle.dump(StoMADS_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(NOMAD_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open('opt_data/data_dicts.pkl', 'rb') as file:
            StoMADS_data_dict = pickle.load(file)
            NOMAD_data_dict = pickle.load(file)

    #==========================================================
    # find best known solution 
    x_keys = ['x1','x2','x3']
    d_threshold = 0.08
    d_threshold = 0.3

    x_opt = np.empty((0,3))
    f_opt = np.empty((0,1))
    c_opt = np.empty((0,1))
    p_opt = np.empty((0,1))
    n_k = np.empty((0,1))
    d_best = np.empty((0,1))
    converged = np.empty((0,1))
    algos = []; n = []; bbe = []

    # Append StoMADS final results
    for key,value in StoMADS_data_dict.items():
        if isinstance(value, pd.DataFrame):
            n += [key]
            bbe += [value['bbe'].iloc[-1]]
            x_opt = np.vstack((x_opt,value[x_keys].iloc[-1].to_numpy()))
            f_opt = np.vstack((f_opt,value['true_f'].iloc[-1]))
            c_opt = np.vstack((c_opt,value['true_cstr'].iloc[-1]))
            p_opt = np.vstack((p_opt,value['true_p_value'].iloc[-1]))
            n_k = np.vstack((n_k,StoMADS_dict['stats'][key]['n_k']))
            d_best = np.vstack((d_best,value['distance'].iloc[-1]))

            # find iteration threshold
            close_iterations = value[value['distance'] < d_threshold]['bbe']
            if len(close_iterations.index) > 0:
                converged = np.vstack((converged,close_iterations.iloc[0]))
            else:
                converged = np.vstack((converged,np.inf))

            algos += [StoMADS_dict['folder']]

    # Append NOMAD final results
    for key,value in NOMAD_data_dict.items():
        if isinstance(value, pd.DataFrame):
            n += [key]
            bbe += [value['bbe'].iloc[-1]]
            x_opt = np.vstack((x_opt,value[x_keys].iloc[-1].to_numpy()))
            f_opt = np.vstack((f_opt,value['true_f'].iloc[-1]))
            c_opt = np.vstack((c_opt,value['true_cstr'].iloc[-1]))
            p_opt = np.vstack((p_opt,value['true_p_value'].iloc[-1]))
            n_k = np.vstack((n_k,NOMAD_dict['stats'][key]['n_k']))
            d_best = np.vstack((d_best,value['distance'].iloc[-1]))

            # find iteration threshold
            close_iterations = value[value['distance'] < d_threshold]['bbe']
            if len(close_iterations.index) > 0:
                converged = np.vstack((converged,close_iterations.iloc[0]))
            else:
                converged = np.vstack((converged,np.inf))

            algos += [NOMAD_dict['folder']]

    columns = ['n','bbe','x1*','x2*','x3*','f','c','p','d_best','converged','algo','n_k']
    data = [n,bbe,x_opt[:,0],x_opt[:,1],x_opt[:,2],f_opt,c_opt,p_opt,d_best,converged,algos,n_k]
    
    df_final_results = pd.DataFrame(index=range(len(n)), columns=columns) # create an empty data frame
    for column,data_column in zip(columns,data):
        df_final_results[column] = data_column

    df_final_results.to_excel('opt_data/opt_results.xlsx') # save optimal results
    
    df_final_results = df_final_results.sort_values(['p', 'converged'], ascending=[False, True])
    print(df_final_results)