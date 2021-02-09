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
from MCS_model_points import scaling, system_command, cpp_application, processInput, parallel_sampling, serial_sampling

#==============================================================================
# Create or retrieve LHS data
def LHS_sampling(n_samples, lob_var=None, upb_var=None, folder='data/',base_name='LHS_points', new_LHS=False,):
    # LHS distribution
    if new_LHS and lob_var is not None and upb_var is not None :
        points = lhs(len(lob_var), samples=n_samples, criterion='maximin') # generate LHS grid (maximin criterion)
        points_us = scaling(points,lob_var,upb_var,2) # unscale latin hypercube points
        
        check_folder('data/')

        DOE_full_name = base_name +'.npy'
        DOE_filepath = folder + DOE_full_name
        np.save(DOE_filepath, points_us) # save DOE array
        
            
        DOE_full_name = base_name +'.pkl'
        DOE_filepath = folder + DOE_full_name
        
        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(lob_var, resultsfile)
        pickle.dump(upb_var, resultsfile)
        pickle.dump(points, resultsfile)
        pickle.dump(points_us, resultsfile)
    
        resultsfile.close()

    else:
        DOE_full_name = base_name +'.pkl'
        DOE_filepath = folder + DOE_full_name
        resultsfile=open(DOE_filepath,'rb')
        
        lob_var = pickle.load(resultsfile)
        upb_var = pickle.load(resultsfile)
        # read up to n_samples
        points = pickle.load(resultsfile)[:n_samples,:]
        points_us = pickle.load(resultsfile)[:n_samples,:]
    
        resultsfile.close()

    return lob_var, upb_var, points, points_us

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #===================================================================#
    # LHS search

    # Model variables
    # bounds = np.array([[   16    , 101   ], # number of essential workers
    #                    [   0.0001, 0.15  ], # Social distancing factor
    #                    [   10    , 81    ]]) # Testing capacity

    # bounds = np.array([[   16    , 101   ], # number of essential workers
    #                    [   0.0001, 0.15  ], # Social distancing factor
    #                    [   10    , 51    ]]) # Testing capacity

    bounds = np.array([[   10    , 51   ], # number of essential workers
                       [   0.0001, 0.15  ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    fit_cond = False # Do not fit data
    color_mode = 'color' # Choose color mode (black_White)
    run = 0 # starting point

    # Points to plot
    lob_var = bounds[:,0] # lower bounds
    upb_var = bounds[:,1] # upper bounds
    
    new_LHS = False
    n_samples_LH = 300

    # LHS distribution
    [_,_,_,points] = LHS_sampling(n_samples_LH,lob_var,upb_var,new_LHS=False)

    labels = [None] * len(points)
    run = 0 # starting point

    #===================================================================#
    # n_samples = 1000
    # n_bins = 30 # for continuous distributions
    # min_bin_width_i = 15 # for discrete distributions
    # min_bin_width_f = 5 # for discrete distributions

    n_samples = 500
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = True

    #===================================================================#
    # Initialize
    handles_lgd = []; labels_lgd = [] # initialize legend

    if new_run:
        # New MCS
        run = 0
        
        #============== INITIALIZE WORKING DIRECTORY ===================#
        current_path = os.getcwd()
        job_dir = os.path.join(current_path,'data')
        for f in os.listdir(job_dir):
            dirname = os.path.join(job_dir, f)
            if dirname.endswith(".log"):
                os.remove(dirname)

    # # Resume MCS
    # run = 46
    # points = points[run:]
    # labels = labels[run:]

    # # terminate MCS
    # run = 3
    # run_end = 3 + 1
    # points = points[run:run_end]
    # labels = labels[run:run_end]

    for point,legend_label in zip(points,labels):

        # Model variables
        n_violators = int(point[0])
        SD = point[1]
        test_capacity = int(point[2])

        # Model parameters
        healthcare_capacity = 90

        if new_run:

            #=====================================================================#
            # Design variables
            design_variables = [n_violators, SD, test_capacity]
            parameters = [healthcare_capacity]

            #=====================================================================#
            output_file_base = 'MCS_data_r%i' %run
            [infected_i,fatalities_i,GC_i,distance_i] = parallel_sampling(design_variables,parameters,output_file_base,n_samples)
            # [infected_i,fatalities_i,GC_i,distance_i] = serial_sampling(design_variables,parameters,output_file_base,n_samples)

            with open('data/MCS_data_r%i.pkl' %run,'wb') as fid:
                pickle.dump(infected_i,fid)
                pickle.dump(fatalities_i,fid)
                pickle.dump(GC_i,fid)
                pickle.dump(distance_i,fid)
                run += 1
                continue