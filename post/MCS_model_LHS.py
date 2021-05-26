import os
import numpy as np
import pickle

from functionsDesignSpace.stats_functions import LHS_sampling
from functionsUtilities.utils import serial_sampling,parallel_sampling
from functionsDOE.blackboxes import blackbox_COVID_SIM_UI

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #===================================================================#
    # LHS search

    # Model variables
    # bounds = np.array([[   16    , 101   ], # number of essential workers
    #                    [   0.0001, 0.15  ], # Social distancing factor
    #                    [   10    , 81    ]]) # Testing capacity

    bounds = np.array([[   16    , 101   ], # number of essential workers
                       [   0.0001, 0.15  ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    # bounds = np.array([[   10    , 51   ], # number of essential workers
    #                    [   0.0001, 0.15  ], # Social distancing factor
    #                    [   10    , 51    ]]) # Testing capacity

    # Points to plot
    lob_var = bounds[:,0] # lower bounds
    upb_var = bounds[:,1] # upper bounds
    
    new_LHS = True
    n_samples_LH = 300 # <-------------------------- adjust the number of latin hyper cube points (recommend 30 for quick results)

    # LHS distribution
    [_,_,_,points] = LHS_sampling(n_samples_LH,lob_var,upb_var,new_LHS=new_LHS)
    run = 0 # starting point

    #===================================================================#

    n_samples = 4 # <-------------------------- adjust the number of observations per latin hypercube point (recommend 50 for quick results)
    new_run = True

    #===================================================================#
    # Initialize

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
    # run = 0
    # points = points[run:]

    # # terminate MCS
    # run = 0
    # run_end = 299 + 1
    # points = points[run:run_end]

    for point in points:

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
            args = [design_variables] * n_samples # repeat variables by number of samples

            #=====================================================================#
            # Empty result lists
            infected_i = []; fatalities_i = []; GC_i = []; distance_i = []
            process_I = []; process_F = []; process_R = []; process_M = []; process_R0 = []

            # Blackbox set-up
            params_COVID_SIM_UI = [healthcare_capacity,]
            output_file_base = 'MCS_data_r%i' %run
            return_process = False
            params = [run, output_file_base, params_COVID_SIM_UI,return_process]

            #################################################################
            # Parallel sampling of blackbox (more intense, does not return stochastic disease profiles)s
            results = parallel_sampling(args,params,blackbox_COVID_SIM_UI)
            # Read results
            for result in results:
                
                [infected, fatalities, mean_GC, mean_distance] = result

                infected_i      += [infected]
                fatalities_i    += [fatalities]
                GC_i            += [mean_GC]
                distance_i      += [mean_distance]
            #################################################################
            # # Serial sampling of blackbox (less intense + return stochastic disease profiles)
            # results = serial_sampling(args,params,blackbox_COVID_SIM_UI)

            # # Read results
            # for result in results: 
            #     [infected, fatalities, mean_GC, mean_distance,I,F,R,M,run_data_R0] = result

            #     infected_i      += [infected]
            #     fatalities_i    += [fatalities]
            #     GC_i            += [mean_GC]
            #     distance_i      += [mean_distance]

            #     process_I   += [I]
            #     process_F   += [F]
            #     process_R   += [R]
            #     process_M   += [M]
            #     process_R0  += [run_data_R0]
            #################################################################

            # wipe log files
            for f in os.listdir(job_dir):
                dirname = os.path.join(job_dir, f)
                if dirname.endswith(".log"):
                    os.remove(dirname)

            with open('data/MCS_data_r%i.pkl' %run,'wb') as fid:
                pickle.dump(infected_i,fid)
                pickle.dump(fatalities_i,fid)
                pickle.dump(GC_i,fid)
                pickle.dump(distance_i,fid)
                run += 1
                continue