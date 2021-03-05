import os
import numpy as np
import scipy.stats as st
import statsmodels as sm
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import pickle
import matplotlib.patches as patches
import subprocess
from subprocess import PIPE,STDOUT
from pyDOE import lhs
from utils import check_folder
from MCS_model_LHS_post import scaling, best_fit_distribution, plot_distribution

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #===================================================================#
    fit_cond = False # Do not fit data
    color_mode = 'color' # Choose color mode (black_White)
    run_index = 0 # starting point

    # load optimization points in LHS format file
    with open('data/opts/points_opts.pkl','rb') as fid:
            lob_var = pickle.load(fid)
            upb_var = pickle.load(fid)
            opts = pickle.load(fid)
            opts_unscaled = pickle.load(fid)

    #===================================================================#
    # n_samples = 1000
    # n_bins = 30 # for continuous distributions
    # min_bin_width_i = 15 # for discrete distributions
    # min_bin_width_f = 5 # for discrete distributions

    n_samples = 500 # <-------------------------- adjust the number of observations according to MCS_model_points
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = False

    same_axis = True
    if same_axis:
        fig_infections = plt.figure(figsize=(6,5))
        fig_fatalities = plt.figure(figsize=(6,5))
        fig_dist = plt.figure(figsize=(6,5))
        fig_GC = plt.figure(figsize=(6,5))
    else:
        fig_infections = fig_fatalities = fig_dist = fig_GC = None

    auto_limits = "determine" # also can be "fixed", "load"
    if auto_limits == "determine":
        dataXLim_i = dataYLim_i = None
        dataXLim_f = dataYLim_f = None
        dataXLim_d = dataYLim_d = None
        dataXLim_GC = dataYLim_GC = None
    elif auto_limits == "load":
        with open('data/MCS_data_limits.pkl','rb') as fid:
            dataXLim_i = pickle.load(fid)
            dataYLim_i = pickle.load(fid)
            dataXLim_f = pickle.load(fid)
            dataYLim_f = pickle.load(fid)
            dataXLim_d = pickle.load(fid)
            dataYLim_d = pickle.load(fid)
            dataXLim_GC = pickle.load(fid)
            dataYLim_GC = pickle.load(fid)
    elif auto_limits == "fixed":
        dataXLim_i  = ( 0 , 1000 )
        dataYLim_i  = ( 0 , 0.05 )
        dataXLim_f  = ( 0 , 250  )
        dataYLim_f  = ( 0 , 0.2  )
        dataXLim_d  = ( 75, 175  )
        dataYLim_d  = ( 0 , 0.4  )
        dataXLim_GC = ( 0.5 , 5  )
        dataYLim_GC = ( 0 , 10   )

    mean_i_runs = []; std_i_runs = []; rel_i_runs = []; mean_f_runs = []; std_f_runs = []
    mean_d_runs = []; std_d_runs = []; mean_gc_runs = []; std_gc_runs = []

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                           r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    #===================================================================#
    # Initialize
    handles_lgd = []; labels_lgd = [] # initialize legend

    if color_mode == 'color':
        # hatches = ['/'] * 10
        hatches = [''] * 10
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
        colors = colors * 30 # repeat colors 30 times
    elif color_mode == 'black_white':
        hatches = ['/','//','x','o','|||']
        colors = ['#FFFFFF'] * 300

    # terminate
    select_indices = [0,1,2,3]
    # select_indices = [0,1,2,3,4,5,6]
    # select_indices = [0,5,6]
    opts_unscaled = [opts_unscaled[index] for index in select_indices]
    runs = select_indices

    for point,run in zip(opts_unscaled,runs):

        # Legend labels
        print('point #%i: E = %f, S_D = %f, T = %f' %(run+1,point[0],point[1],point[2]))
        # legend_label = "Run %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))
        legend_label = "Solution %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))

        # Model variables
        n_violators = int(point[0])
        SD = point[1]
        test_capacity = int(point[2])

        # Model parameters
        healthcare_capacity = 90

        with open('data/opts/MCS_data_r%i.pkl' %run,'rb') as fid:
            infected_i = pickle.load(fid)
            fatalities_i = pickle.load(fid)
            GC_i = pickle.load(fid)
            distance_i = pickle.load(fid)

        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='k', alpha=0.5, facecolor=colors[run], fill=True ,hatch=hatches[run])

        handles_lgd += [a]
        labels_lgd += [legend_label]

        # Infected plot
        label_name = u'Maximum number of infections $\max(n_I^k)$'
        fun_name = 'infections'
        data = infected_i

        # pass lists by value instead of by reference as python does by default
        handles_lgd_i = handles_lgd[:]
        labels_lgd_i = labels_lgd[:]

        [dataXLim_i_out, dataYLim_i_out, mean_i, std_i, rel_i] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_i, fig_swept = fig_infections, 
            run_label = legend_label, color = colors[run],  hatch_pattern = hatches[run], 
            dataXLim = dataXLim_i, dataYLim = dataYLim_i, constraint = healthcare_capacity,
            fit_distribution = fit_cond, handles = handles_lgd_i, labels = labels_lgd_i)

        mean_i_runs += [mean_i]
        std_i_runs += [std_i]
        rel_i_runs += [rel_i]

        # Fatalities plot
        label_name = u'Number of fatalities $n_I^t$'
        fun_name = 'fatalities'
        data = fatalities_i

        [dataXLim_f_out, dataYLim_f_out, mean_f, std_f, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_f, fig_swept = fig_fatalities, 
            run_label = legend_label, color = colors[run], hatch_pattern = hatches[run], 
            dataXLim = dataXLim_f, dataYLim = dataYLim_f, constraint = None,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_f_runs += [mean_f]
        std_f_runs += [std_f]

        # Distance plot
        label_name = u'Average cumulative distance travelled $D(\mathbf{x})$'
        fun_name = 'distance'
        data = distance_i

        [dataXLim_d_out, dataYLim_d_out, mean_d, std_d, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_dist, run_label = legend_label, color = colors[run], constraint = None,
            hatch_pattern = hatches[run], dataXLim = dataXLim_d, dataYLim = dataYLim_d,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_d_runs += [mean_d]
        std_d_runs += [std_d]

        label_name = u'Mobility $M^t$'
        fun_name = 'ground covered'
        data = GC_i

        # Ground covered plot
        [dataXLim_GC_out, dataYLim_GC_out, mean_gc, std_gc, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_GC, run_label = legend_label, color = colors[run], constraint = None, 
            hatch_pattern = hatches[run], dataXLim = dataXLim_GC, dataYLim = dataYLim_GC,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_gc_runs += [mean_gc]
        std_gc_runs += [std_gc]

        if auto_limits != "determine":
            fig_infections.savefig('data/%i_PDF_%s.png' %(run , 'infections'), 
                                    format='png', dpi=100,bbox_inches='tight')
            fig_fatalities.savefig('data/%i_PDF_%s.png' %(run , 'fatalities'), 
                                format='png', dpi=100,bbox_inches='tight')
            fig_dist.savefig('data/%i_PDF_%s.png' %(run , 'distance'), 
                            format='png', dpi=100,bbox_inches='tight')
            fig_GC.savefig('data/%i_PDF_%s.png' %(run , 'ground_covered'), 
                            format='png', dpi=100,bbox_inches='tight')

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean infections: %f; std infections: %f; rel infections: %f' %(mean_i,std_i,rel_i))
        print('mean fatalities: %f; std fatalities: %f' %(mean_f,std_f))
        print('mean distance: %f; std distance: %f' %(mean_d,std_d))
        print('mean ground covered: %f; std ground covered: %f' %(mean_gc,std_gc))
        print('==============================================')
        run_index += 1

    with open('data/MCS_data_limits.pkl','wb') as fid:
        pickle.dump(dataXLim_i_out,fid)
        pickle.dump(dataYLim_i_out,fid)
        pickle.dump(dataXLim_f_out,fid)
        pickle.dump(dataYLim_f_out,fid)
        pickle.dump(dataXLim_d_out,fid)
        pickle.dump(dataYLim_d_out,fid)
        pickle.dump(dataXLim_GC_out,fid)
        pickle.dump(dataYLim_GC_out,fid)

    with open('data/opts/MCS_data_stats_opts.pkl','wb') as fid:
        pickle.dump(mean_i_runs,fid)
        pickle.dump(std_i_runs,fid)
        pickle.dump(rel_i_runs,fid)
        pickle.dump(mean_f_runs,fid)
        pickle.dump(std_f_runs,fid)
        pickle.dump(mean_d_runs,fid)
        pickle.dump(std_d_runs,fid)
        pickle.dump(mean_gc_runs,fid)
        pickle.dump(std_gc_runs,fid)

    if same_axis:
        fig_infections.savefig('data/PDF_%s.png' %('infections'), 
                                format='png', dpi=100,bbox_inches='tight')
        fig_fatalities.savefig('data/PDF_%s.png' %('fatalities'), 
                            format='png', dpi=100,bbox_inches='tight')
        fig_dist.savefig('data/PDF_%s.png' %('distance'), 
                        format='png', dpi=100,bbox_inches='tight')
        fig_GC.savefig('data/PDF_%s.png' %('ground_covered'), 
                       format='png', dpi=100,bbox_inches='tight')
        plt.show()