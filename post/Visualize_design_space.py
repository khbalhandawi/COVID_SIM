# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:24 2017

@author: Khalil
"""
import pickle, time
import numpy as np
import matplotlib.pyplot as plt

from functionsDesignSpace.stats_functions import LHS_sampling
from functionsDesignSpace.visualization import train_server,visualize_surrogate

#------------------------------------------------------------------------------#
# MAIN FILE
def main():

    ####################### SETUP #######################
    '''
        1) Run Statistics_summary_LHS_points.py first
        2) Run Visualize_model_points.py second
    '''
    #####################################################

    #============================== TRAINING DATA =================================#
    # one-liner to read a single variable
    with open('data_vis/LHS/MCS_data_stats.pkl','rb') as fid:
        mean_i_runs = pickle.load(fid)
        std_i_runs = pickle.load(fid)
        rel_i_runs = pickle.load(fid)
        mean_f_runs = pickle.load(fid)
        std_f_runs = pickle.load(fid)
        mean_d_runs = pickle.load(fid)
        std_d_runs = pickle.load(fid)
        mean_gc_runs = pickle.load(fid)
        std_gc_runs = pickle.load(fid)

    with open('data_vis/opts/MCS_data_stats_opts.pkl','rb') as fid:
        mean_i_opts = pickle.load(fid)
        std_i_opts = pickle.load(fid)
        rel_i_opts = pickle.load(fid)
        mean_f_opts = pickle.load(fid)
        std_f_opts = pickle.load(fid)
        mean_d_opts = pickle.load(fid)
        std_d_opts = pickle.load(fid)
        mean_gc_opts = pickle.load(fid)
        std_gc_opts = pickle.load(fid)

    # Variables
    n_samples_LH = 299 + 1 # <-------------------------- adjust the number of observations according to MCS_model_LHS
    [lob_var, upb_var, points, points_us] = LHS_sampling(n_samples_LH,new_LHS=False,folder='data_vis/LHS/')

    # Opts
    n_samples_opts = 7
    [_, _, opts, opts_us] = LHS_sampling(n_samples_opts,base_name='opts/points_opts',new_LHS=False,folder='data_vis/')

    print(opts_us[6:7,:])

    # Design space
    training_X = points_us

    # #============================= MAIN EXECUTION =================================#
    start_time = time.time()

    # Define design space bounds
    bounds = np.column_stack((lob_var, upb_var))

    #===========================================================================
    # Visualize surrogate model of expectation 
    # Outputs
    obj = np.array(mean_gc_runs[:n_samples_LH])
    cstr = np.array(rel_i_runs[:n_samples_LH])
    
    training_Y = np.column_stack((obj, cstr))

    variable_lbls = ['Essential workers $n_E$','Social distancing $S_D$','Tests/frame $n_T$']
    output_lbls = [r'$-\bar{{f}_{\Theta}}(\mathbf{x})$', '$\mathbb{P}({g}_{\Theta}(\mathbf{x}) - H_{\mathrm{max}} \le 0) \le 0.9$' ]

    # Train surrogate model for use in subsequent plotting
    server_mean = train_server(training_X,training_Y,bounds)
    
    # Visualize surrogate model
    resolution = 20
    vis_output = 0

    # # For overlaying optimization results on design space
    # visualize_surrogate(bounds,variable_lbls,server_mean,training_X,
    #                     training_Y,plt,threshold=0.9,
    #                     resolution=resolution,output_lbls=output_lbls,
    #                     opts=opts_us[6:7,:],base_name='opt_results')

    # For design space projections only
    visualize_surrogate(bounds,variable_lbls,server_mean,training_X,
                        training_Y,plt,threshold=0.9,
                        resolution=resolution,output_lbls=output_lbls,
                        opts=None,cmax=4,cmin=1,base_name='sensitivity_E')

    plt.show()

    #===========================================================================
    # Visualize surrogate model of variance
    # Outputs
    obj = np.array(std_gc_runs[:n_samples_LH])
    cstr = np.array(std_i_runs[:n_samples_LH])
    
    print(max(std_i_runs))

    training_Y = np.column_stack((std_i_runs,))

    variable_lbls = ['Essential workers $n_E$','Social distancing $S_D$','Tests/frame $n_T$',]
    output_lbls = ['Variance $\mathbb{E}_{\mathbf{\Theta}}[\left({f}_{\Theta}(\mathbf{x}) - \mu \right)^2]$',]
    output_lbls = ['Standard deviation $s$',]

    # Train surrogate model for use in subsequent plotting
    server_var = train_server(training_X,training_Y,bounds)
    
    # Visualize surrogate model
    resolution = 20
    vis_output = 0

    visualize_surrogate(bounds,variable_lbls,server_var,training_X,
                        training_Y,plt,threshold=None,
                        resolution=resolution,output_lbls=output_lbls,
                        opts=None,cmax=None, cmin=None,base_name='sensitivity_V')
    
    plt.show()
        
if __name__ == '__main__':
    main()
