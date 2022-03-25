import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import copy

from process_results import process_group,build_legend_horizontal,build_legend,build_fig_blank,build_fig

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
            0   : { "index" : 1,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
            1   : { "index" : 2,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
            2   : { "index" : 3,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
            3   : { "index" : 4,    "n_k" : 20, "min_mesh_size_enabled" : True,   "min_mesh_size" : None,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
            4   : { "index" : 5,    "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-6,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
            5   : { "index" : 6,    "n_k" : 1,  "min_mesh_size_enabled" : False,  "min_mesh_size" : 1e-6,     "epsilon" : 1e-13, "config" : "default",   "n_cores" : 8, "group" : 0}, # best ignore
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
            0   : { "index" : 1,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 1},
            1   : { "index" : 2,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 1},
            2   : { "index" : 3,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 1},
            3   : { "index" : 4,    "n_k" : 1,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 1},
            4   : { "index" : 5,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            5   : { "index" : 6,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            6   : { "index" : 7,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            7   : { "index" : 8,    "n_k" : 4,  "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 2},
            8   : { "index" : 9,    "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            9   : { "index" : 10,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            10  : { "index" : 11,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            11  : { "index" : 12,   "n_k" : 20, "n_population" : 50,    "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 3},
            12  : { "index" : 13,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4},
            13  : { "index" : 14,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4},
            14  : { "index" : 15,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4},
            15  : { "index" : 16,   "n_k" : 1,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 4},
            16  : { "index" : 17,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5},
            17  : { "index" : 18,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5},
            18  : { "index" : 19,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5},
            19  : { "index" : 20,   "n_k" : 4,  "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 5},
            20  : { "index" : 21,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            21  : { "index" : 22,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            22  : { "index" : 23,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            23  : { "index" : 24,   "n_k" : 20, "n_population" : 100,   "R_initial" : "auto",   "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 6},
            24  : { "index" : 25,   "n_k" : 4,  "n_population" : 50,    "R_initial" : 5.0,      "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 4,  "group" : 0}, # best ignore
            25  : { "index" : 26,   "n_k" : 4,  "n_population" : 50,    "R_initial" : 2.0,      "R_factor" : 100,   "max_stall_G" : 50, "config" : "default",   "n_cores" : 8,  "group" : 0}, # best ignore
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

    # Initialize data dictionaries (dictionaries are mutable)
    GA_data_dict = copy.deepcopy(GA_dict["stats"])
    for key in GA_dict["stats"].keys():
        GA_data_dict[key] = ''

    #==========================================================
    # Setup color palette (fixed random seed)

    styles = ['-','--','-.',]
    xlims = [-10, 6250]
    ylims1 = [-2.75, -0.5]
    ylims2 = [-100, 250]
    # xlims = None
    # ylims1 = None
    # ylims2 = None
    plot = True

    # best known result
    # best_x = np.array([[1, 0.31, 0.94],])
    # best_x = np.array([[0.75674921, 0.417047454, 0.971785993],])
    best_x = np.array([[0.804693604,0.310076383,0.940647972]])


    # Setup figures
    x_label = 'number of function evaluations'
    y_label1 = r'Objective function estimate $\bar{f_{\Theta}}$'
    y_label2 = r'Constraint function estimate $\bar{c_{\Theta}}$'
    # y_label2 = r'Probability of satisfying constraint $\bar{P}$'
    
    for n_plot in range(3):
            
        # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
        palette = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        for i in range(20):
            palette += ['#FF0000']

        styles_leg = []
        palette_leg = []

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

        group_indices = [3,] # groups for n_k = 20

        algo = {
            "folder" : StoMADS_dict["folder"],
            "file_suffix" : "STOMADS34",
            "legend_title" : r'StoMADS-PB ',
            "legend_labels" : [r'$\epsilon_f=%.2f$',],
            "algo_params" : []
        }

        if n_plot >= 0 :
            styles_leg += [styles[i]]
            palette_leg += [palette[i]]
            i = process_group(axs,group_indices,algo,StoMADS_dict,palette,labels,i,StoMADS_data_dict,best_x,style='-',plot=plot)

        #-------------------- NOMAD RESULTS --------------------#

        group_indices = [3,] # groups for n_k = 20

        algo = {
            "folder" : NOMAD_dict["folder"],
            "file_suffix" : "NOMAD",
            "legend_title" : r'NOMAD',
            "legend_labels" : [r'%s',],
            "algo_params" : []
        }

        if n_plot >= 1:
            styles_leg += [styles[i]]
            palette_leg += [palette[i]]
            i = process_group(axs,group_indices,algo,NOMAD_dict,palette,labels,i,NOMAD_data_dict,best_x,style='--',plot=plot)

        #---------------------- GA RESULTS ---------------------#

        group_indices = [3,] # groups for n_k = 1

        algo = {
            "folder" : GA_dict["folder"],
            "file_suffix" : "GA",
            "legend_title" : r'GA ',
            "legend_labels" : [r'population size $\bar{p}=%i$',],
            "algo_params" : []
        }

        if n_plot >= 2:
            styles_leg += [styles[i]]
            palette_leg += [palette[i]]
            i = process_group(axs,group_indices,algo,GA_dict,palette,labels,i,GA_data_dict,best_x,style='-.',plot=plot)

        if plot:
            # build_legend(figs[0],axs[0],labels,palette,styles=styles,location='upper right')
            # build_legend(figs[1],axs[1],labels,palette,styles=styles,location='lower right')

            figs[0].savefig('opt_data/f_nk=20_%i.pdf' %n_plot, format='pdf', dpi=200, bbox_inches = None, pad_inches = 0.0)
            figs[1].savefig('opt_data/g_nk=20_%i.pdf' %n_plot, format='pdf', dpi=200, bbox_inches = None, pad_inches = 0.0)



        if plot:
            labels += [r'$\bar{c}_{\Theta}=0$']; palette_leg += 'k'; styles_leg += ['--'] # c= 0 label, line color, and style
            fig_leg = build_fig_blank()
            build_legend_horizontal(fig_leg,labels,palette_leg,styles=styles_leg)
            fig_leg.savefig('opt_data/legend_%i.pdf' %n_plot, format='pdf', dpi=200, bbox_inches = 'tight', pad_inches = 0.0)

    plt.show()