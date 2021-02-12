# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:24 2017

@author: Khalil
"""
import sys, os, pickle, time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from visualization import gridsamp, hyperplane_SGTE_vis_norm
from MCS_model_LHS_post import LHS_sampling

#==============================================================================#
# SCALING BY A RANGE
def scaling(x,l,u,operation):
    ''' 
    scaling() scales or unscales the vector x according to the bounds
    specified by u and l. The flag type indicates whether to scale (1) or
    unscale (2) x. Vectors must all have the same dimension.
    '''
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

#==============================================================================#
# POSTPROCESS DOE DATA
def train_server(training_X, training_Y, bounds):
    
    from visualization import define_SGTE_model
    from SGTE_library import SGTE_server
   
    #======================== SURROGATE META MODEL ============================#
    # %% SURROGATE modeling
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    Y = training_Y; S_n = scaling(training_X, lob, upb, 1)
    # fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE']
    # run_types = ['optimize hyperparameters','load hyperparameters'] (1 or 2)
    fit_type = 0; run_type = 1 # optimize all hyperparameters
    model,sgt_file = define_SGTE_model(fit_type,run_type)
    server = SGTE_server(model)
    server.sgtelib_server_start()
    server.sgtelib_server_ping()
    server.sgtelib_server_newdata(S_n,Y)    
    #===========================================================================
    # M = server.sgtelib_server_metric('RMSECV')
    # print('RMSECV Metric: %f' %(M[0]))
    #===========================================================================    

    return server

def visualize_surrogate(bounds,variable_lbls,server,training_X,training_Y,plt,current_path=os.getcwd(),
                        vis_output=0,threshold=None,resolution=20,
                        output_lbls=None,base_name="surrogate_model",
                        opts=None,cmax=None,cmin=None):

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'

    #===========================================================================
    # Plot 2D projectionsoutput_label="$\hat{f}(\mathbf{x})$"
    if opts is not None:
        nominal = scaling(opts[0,:], bounds[:,0], bounds[:,1], 1)
        nn = resolution
    else:
        nominal = [0.5]*len(bounds[:,0]); nn = resolution
    fig = plt.figure()  # create a figure object

    if output_lbls:
        if threshold is not None:
            hyperplane_SGTE_vis_norm(server,training_X,bounds,variable_lbls,nominal,training_Y,nn,fig,plt,
                output_label=output_lbls[0],constraint_label=output_lbls[1],opts=opts,threshold=threshold,
                cmax = cmax, cmin = cmin)
        else:
            hyperplane_SGTE_vis_norm(server,training_X,bounds,variable_lbls,nominal,training_Y,nn,fig,plt,
                output_label=output_lbls[0],constraint_label=None,opts=opts,threshold=threshold,
                cmax = cmax, cmin = cmin)
    else:
        hyperplane_SGTE_vis_norm(server,training_X,bounds,variable_lbls,nominal,training_Y,nn,fig,plt,opts=opts,threshold=threshold, cmax=cmax, cmin=cmin)

    fig_name = '%s.pdf' %(base_name)
    fig_file_name = os.path.join(current_path,fig_name)
    fig.savefig(fig_file_name, bbox_inches='tight')

#------------------------------------------------------------------------------#
# MAIN FILE
def main():

    #============================== TRAINING DATA =================================#
    # one-liner to read a single variable
    with open('data/LHS/MCS_data_stats.pkl','rb') as fid:
        mean_i_runs = pickle.load(fid)
        std_i_runs = pickle.load(fid)
        rel_i_runs = pickle.load(fid)
        mean_f_runs = pickle.load(fid)
        std_f_runs = pickle.load(fid)
        mean_d_runs = pickle.load(fid)
        std_d_runs = pickle.load(fid)
        mean_gc_runs = pickle.load(fid)
        std_gc_runs = pickle.load(fid)

    with open('data/opts/MCS_data_stats_opts.pkl','rb') as fid:
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
    n_samples_LH = 60 + 1
    [lob_var, upb_var, points, points_us] = LHS_sampling(n_samples_LH,new_LHS=False)

    # Opts
    n_samples_opts = 7
    [_, _, opts, opts_us] = LHS_sampling(n_samples_opts,base_name='opts/points_opts',new_LHS=False)

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

    # visualize_surrogate(bounds,variable_lbls,server_mean,training_X,
    #                     training_Y,plt,threshold=0.9,
    #                     resolution=resolution,output_lbls=output_lbls,
    #                     opts=opts_us[6:7,:],base_name='opt_results')

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
