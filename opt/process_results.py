import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

#=========================================================#
#                  SETUP VISUALIZATION                    #
#=========================================================#
def build_fig():
    
    # mpl.rc('text', usetex = True)
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    # mpl.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(9,6))
    fig.subplots_adjust()

    ax.set_xlabel('number of model evaluations',fontsize=18)
    ax.set_ylabel(r'average objective function value $\bar{f_\Theta}$',fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    return fig,ax

#=========================================================#
#                    CONSTRUCT LEGEND                     #
#=========================================================#
def build_legend(fig,ax,labels,palette,style='-'):

    # Legend
    import matplotlib.lines as mlines

    handles = []; i = 0
    for label in labels:
        handle = mlines.Line2D([], [], color=palette[i], marker='', markersize=5, linestyle=style)
        handles += [handle]
        i += 1

    lx = fig.legend(handles, labels, loc='upper right', fontsize = 12, )

    # Get the bounding box of the original legend
    bb = lx.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    # Change to location of the legend. 
    xOffset = -0.14; yOffset = -0.15
    bb.x0 += xOffset; bb.x1 += xOffset # Move anchor point x0 and x1 to gether
    bb.y0 += yOffset; bb.y1 += yOffset # Move anchor point y0 and y1 to gether
    lx.set_bbox_to_anchor(bb, transform = ax.transAxes)

#=========================================================#
#                          MAIN                           #
#=========================================================#
if __name__ == "__main__":

    #==========================================================
    # Experiment stats

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

    StoMADS_config = {
        "default" : {
            "Delta" : 1,
            "gamma" : 5,
            "tau" : 0.5,
            "OPPORTUNISTIC_EVAL" : True,
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
            18  : { "index" : 19,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4}, # running
            19  : { "index" : 20,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4}, # running
            20  : { "index" : 21,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4}, # running
            21  : { "index" : 22,   "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 4}, # running
            22  : { "index" : 27,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5}, # left to go
            23  : { "index" : 28,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5}, # left to go
            24  : { "index" : 29,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5}, # left to go
            25  : { "index" : 30,   "n_k" : 4,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 5}, # left to go
            26  : { "index" : 23,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6}, # running
            27  : { "index" : 24,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6}, # running
            28  : { "index" : 25,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6}, # running
            29  : { "index" : 26,   "n_k" : 20, "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-31,    "epsilon" : 1e-31, "config" : "basic",     "n_cores" : 8, "group" : 6}, # running
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

    #==========================================================
    # Setup figures
    fig1,ax1 = build_fig()
    fig2,ax2 = build_fig()

    # Setup color palette (fixed random seed)

    r = lambda: random.randint(0,255)
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(20):
        # random.seed(i)
        palette += ['#%02X%02X%02X' % (r(),r(),r())]


    #==========================================================
    # Load raw data from text

    labels = []; i = 0
    #--------------------- NOMAD RESULTS ---------------------#

    # pick_indices = [2,3,7]
    pick_indices = []
    stats = { x: NOMAD_dict["stats"][x] for x in pick_indices }
    folder = NOMAD_dict["folder"]

    for key in stats:
        index = stats[key]["index"]
        n_k = stats[key]["n_k"]

        run_folder = r'opt_data/%s/Run_%i/' %(folder,index)\
        
        f_filename = run_folder + "f_hist_NOMAD.txt"
        f_true = run_folder + "f_progress_NOMAD.txt"

        data = pd.read_csv(f_filename, sep=",") 
        if os.path.isfile(f_true):
            data_true = pd.read_csv(f_true, sep=",") 
            map = np.searchsorted(data_true['bbe'], data['bbe'])
            data['true_f'] = [data_true['f'][x-1] for x in map]
            data['true_cstr'] = [data_true['cstr'][x-1] for x in map]
            data['true_p_value'] = [data_true['p_value'][x-1] for x in map]
            ax1.plot(data['bbe'],data['true_f'],color=palette[i])
            ax2.plot(data['bbe'],data['true_p_value'],color=palette[i])
        else:
            ax1.plot(data['bbe'],data['f'],color=palette[i])
            ax2.plot(data['bbe'],data['p_value'],color=palette[i])

        label = r'NOMAD-default $n^k=%i$' %(n_k)
        labels += [label]
        i += 1

    #-------------------- STOMADS RESULTS --------------------#
    pick_indices = [6,2,1,0,5,9,7]
    # pick_indices = [1,5,7]
    stats = { x: StoMADS_dict["stats"][x] for x in pick_indices }
    folder = StoMADS_dict["folder"]

    for key in stats:
        index = stats[key]["index"]
        n_k = stats[key]["n_k"]
        epsilon_f = stats[key]["epsilon_f"]

        run_folder = r'opt_data/%s/Run_%i/' %(folder,index)\
        
        f_filename = run_folder + "f_hist_STOMADS34.txt"
        f_true = run_folder + "f_progress_STOMADS34.txt"

        data = pd.read_csv(f_filename, sep=",") 
        if os.path.isfile(f_true):
            data_true = pd.read_csv(f_true, sep=",") 
            map = np.searchsorted(data_true['bbe'], data['bbe'])
            data['true_f'] = [data_true['f'][x-1] for x in map]
            data['true_cstr'] = [data_true['cstr'][x-1] for x in map]
            data['true_p_value'] = [data_true['p_value'][x-1] for x in map]
            ax1.plot(data['bbe'],data['true_f'],color=palette[i])
            ax2.plot(data['bbe'],data['true_p_value'],color=palette[i])
        else:
            ax1.plot(data['bbe'],data['f'],color=palette[i])
            ax2.plot(data['bbe'],data['p_value'],color=palette[i])

        label = r'StoMADS-PB $n^k=%i$, $\epsilon_f=%.2f$' %(n_k,epsilon_f)
        labels += [label]
        i += 1

    build_legend(fig1,ax1,labels,palette,style='-')
    build_legend(fig2,ax2,labels,palette,style='-')
    plt.show()