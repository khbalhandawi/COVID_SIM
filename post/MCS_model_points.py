import os
import numpy as np
import pickle

from functionsUtilities.utils import serial_sampling,parallel_sampling,scaling
from functionsDOE.blackboxes import blackbox_COVID_SIM_UI

#==============================================================================#
# Main execution
if __name__ == '__main__':

    #===================================================================#
    
    # Model variables
    bounds = np.array([[   16    , 101   ], # number of essential workers
                       [   0.0001, 0.15  ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    run = 0 # starting point

    # Points to plot
    # Demo points
    opts = np.array([[0.164705882 , 0.666444296 , 0.000000000 ],
                     [0.517647058 , 0.666444296 , 0.000000000 ],
                     [0.164705882 , 0.833222148 , 0.000000000 ],
                     [0.164705882 , 0.666444296 , 0.243902439 ]])

    # Points to plot
    # StoMADS V2
    # opts = np.array([[0.6203810000, 0.2954270000, 0.9158810000],
    #                  [0.6164750000, 0.2946950000, 0.9184600000],
    #                  [0.6360060000, 0.3064140000, 0.9809600000],
    #                  [0.6277050000, 0.3103200000, 0.9299350000],
    #                  [0.6194040000, 0.3039720000, 0.9028350000],
    #                  [0.6438180000, 0.3571950000, 0.8989290000],
    #                  [0.8664750000, 0.4196950000, 0.9809600000]])

    opts_unscaled = scaling(opts, bounds[:3,0], bounds[:3,1], 2)

    i = 0
    for point in opts_unscaled:
        print('point #%i: E = %f, S_D = %f, T = %f' %(i+1,point[0],point[1],point[2])); i+=1

    # Points to plot
    lob_var = bounds[:,0] # lower bounds
    upb_var = bounds[:,1] # upper bounds

    # save optimization points in LHS format file
    with open('data/points_opts.pkl','wb') as fid:
        pickle.dump(lob_var, fid)
        pickle.dump(upb_var, fid)
        pickle.dump(opts, fid)
        pickle.dump(opts_unscaled, fid)

    #===================================================================#
    n_samples = 1 # <<-------------------------- Edit the number of observations
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

        # Resume MCS
        # run = 1
        # opts_unscaled = opts_unscaled[run:]

        # # terminate MCS
        # run = 3
        # run_end = 3 + 1
        # opts_unscaled = opts_unscaled[run:run_end]

        for point in opts_unscaled:

            # Model variables
            n_violators = int(point[0])
            SD = point[1]
            test_capacity = int(point[2])

            # Model parameters
            healthcare_capacity = 90

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
            # return_process = False
            return_process = True
            params = [run, output_file_base, params_COVID_SIM_UI,return_process]

            #################################################################
            # Parallel sampling of blackbox (more intense, does not return stochastic disease profiles)s
            # results = parallel_sampling(args,params,blackbox_COVID_SIM_UI)
            # # Read results
            # for result in results:
                
            #     [infected, fatalities, mean_GC, mean_distance] = result

            #     infected_i      += [infected]
            #     fatalities_i    += [fatalities]
            #     GC_i            += [mean_GC]
            #     distance_i      += [mean_distance]
            #################################################################
            # Serial sampling of blackbox (less intense + return stochastic disease profiles)
            results = serial_sampling(args,params,blackbox_COVID_SIM_UI)

            # Read results
            for result in results: 
                [infected, fatalities, mean_GC, mean_distance,I,F,R,M,run_data_R0] = result

                infected_i      += [infected]
                fatalities_i    += [fatalities]
                GC_i            += [mean_GC]
                distance_i      += [mean_distance]

                process_I   += [I]
                process_F   += [F]
                process_R   += [R]
                process_M   += [M]
                process_R0  += [run_data_R0]
            #################################################################

            # wipe log files
            for f in os.listdir(job_dir):
                dirname = os.path.join(job_dir, f)
                if dirname.endswith(".log"):
                    os.remove(dirname)

            with open('data/MCS_process_data_r%i.pkl' %run,'wb') as fid:
                pickle.dump(process_I,fid)
                pickle.dump(process_F,fid)
                pickle.dump(process_R,fid)
                pickle.dump(process_M,fid)
                pickle.dump(process_R0,fid)

            with open('data/MCS_data_r%i.pkl' %run,'wb') as fid:
                pickle.dump(infected_i,fid)
                pickle.dump(fatalities_i,fid)
                pickle.dump(GC_i,fid)
                pickle.dump(distance_i,fid)
                run += 1
                continue
