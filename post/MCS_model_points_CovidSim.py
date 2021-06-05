import os
import numpy as np
import pickle

from functionsUtilities.utils import serial_sampling,scaling
from functionsDOE.blackboxes import blackbox_COVID_SIM_UI
from functionsDOE.blackbox_CovidSim import blackbox_CovidSim

#==============================================================================#
# Main execution
if __name__ == '__main__':

    #===================================================================#
    
    # Model variables
    bounds = np.array([[   16    , 101   ], # number of essential workers
                       [   0.0001, 0.15  ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    run = 0 # starting point

    #===================================================================#
    # trail points with CovidSim and COVID_SIM_UI

    ###################### UNITED KINGDOM ######################
    # bounds_CovidSim = np.array([[   1.0     , 0.0   ],      # compliance rate (inversely proportional to number of essential workers)
    #                             [   3.0     , 0.05  ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
    #                             [   0.1     , 0.9   ]])     # Testing capacity

    # opts_CovidSim = np.array([  [0.500000000 , 0.500000000 , 0.500000000 ],
    #                             [0.433109873 , 0.684745762 , 0.993190799 ],    # United Kingdom
    #                             [0.1164501187, 0.524594992 , 0.9041672301],    # United Kingdom
    #                             [0.3039501187, 0.441967874 , 0.6541672301]])   # United Kingdom

    ########################## CANADA ##########################
    # bounds_CovidSim = np.array([[   1.0     , 0.9   ],      # compliance rate (inversely proportional to number of essential workers)
    #                             [   5.0     , 1.0   ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
    #                             [   0.1     , 0.9   ]])     # Testing capacity

    # opts_CovidSim = np.array([  [0.5000000000, 0.5000000000, 0.5000000000],
    #                             [0.3762299506, 0.1908116385, 0.4284971336],     # Canada R1
    #                             [0.9218750000, 0.3437500000, 0.4375000000],     # Canada R2
    #                             [0.5425358063, 0.1014413537, 0.9212206925],     # Canada R3
    #                             [0.9158407368, 0.3206616619, 0.2405757265],     # Canada R4
    #                             [0.2491510932, 0.1319933987, 0.9961497223],     # Canada R5
    #                             [0.9696505299, 0.3888367454, 0.4676578774]])    # Canada R6

    # with different bounds
    # bounds_CovidSim = np.array([[   1.0         , 0.9   ],      # compliance rate (inversely proportional to number of essential workers)
    #                             [   6.243926142 , 1.0   ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
    #                             [   0.1         , 0.9   ]])     # Testing capacity

    # opts_CovidSim = np.array([  [0.5000000000, 0.618606375 , 0.5000000000],
    #                             [0.3762299506, 0.382761435 , 0.4284971336],     # Canada R1
    #                             [0.9218750000, 0.499420867 , 0.4375000000],     # Canada R2
    #                             [0.5425358063, 0.314590921 , 0.9212206925],     # Canada R3
    #                             [0.9158407368, 0.481809377 , 0.2405757265],     # Canada R4
    #                             [0.2491510932, 0.337895631 , 0.9961497223],     # Canada R5
    #                             [0.9696505299, 0.533812461 , 0.4676578774]])    # Canada R6

    # with different bounds
    bounds_CovidSim = np.array([[   1.0     , 0.9   ],      # compliance rate (inversely proportional to number of essential workers)
                                [   7.0     , 1.0   ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
                                [   0.1     , 0.9   ]])     # Testing capacity

    opts_CovidSim = np.array([  [0.5000000000, 0.666666667 , 0.5000000000],
                                [0.3762299506, 0.460541092 , 0.4284971336],     # Canada R1
                                [0.9218750000, 0.562500000 , 0.4375000000],     # Canada R2
                                [0.5425358063, 0.400960902 , 0.9212206925],     # Canada R3
                                [0.9158407368, 0.547107775 , 0.2405757265],     # Canada R4
                                [0.2491510932, 0.421328932 , 0.9961497223],     # Canada R5
                                [0.9696505299, 0.592557830 , 0.4676578774]])    # Canada R6

    # with different bounds
    # bounds_CovidSim = np.array([[   1.0     , 0.9   ],      # compliance rate (inversely proportional to number of essential workers)
    #                             [   8.0     , 1.0   ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
    #                             [   0.1     , 0.9   ]])     # Testing capacity

    # opts_CovidSim = np.array([  [0.5000000000, 0.714285714 , 0.5000000000],
    #                             [0.3762299506, 0.537606651 , 0.4284971336],     # Canada R1
    #                             [0.9218750000, 0.625000000 , 0.4375000000],     # Canada R2
    #                             [0.5425358063, 0.486537916 , 0.9212206925],     # Canada R3
    #                             [0.9158407368, 0.611806664 , 0.2405757265],     # Canada R4
    #                             [0.2491510932, 0.503996228 , 0.9961497223],     # Canada R5
    #                             [0.9696505299, 0.650763855 , 0.4676578774]])    # Canada R6

    # # with different bounds
    # bounds_CovidSim = np.array([[   1.0     , 0.9   ],      # compliance rate (inversely proportional to number of essential workers)
    #                             [   10.0    , 1.0   ],      # Contact rate given Social distancing (inversely proportional to Social distancing factor)
    #                             [   0.1     , 0.9   ]])     # Testing capacity

    # opts_CovidSim = np.array([  [0.5000000000, 0.777777777 , 0.5000000000],
    #                             [0.3762299506, 0.640360728 , 0.4284971336],     # Canada R1
    #                             [0.9218750000, 0.708333333 , 0.4375000000],     # Canada R2
    #                             [0.5425358063, 0.600640601 , 0.9212206925],     # Canada R3
    #                             [0.9158407368, 0.698071849 , 0.2405757265],     # Canada R4
    #                             [0.2491510932, 0.614219288 , 0.9961497223],     # Canada R5
    #                             [0.9696505299, 0.728371886 , 0.4676578774]])    # Canada R6

    # Algorithmic settings (index corresponds to array index)
    solution_dict = {
        0   : { "index" : 1,    "n_k" : 1,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 16, "group" : 0},
        1   : { "index" : 3,    "n_k" : 1,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 8,  "group" : 0},
        3   : { "index" : 5,    "n_k" : 2,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 8,  "group" : 1},
        2   : { "index" : 4,    "n_k" : 2,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 16, "group" : 1},
        5   : { "index" : 6,    "n_k" : 3,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 16, "group" : 2},
        4   : { "index" : 2,    "n_k" : 4,  "epsilon_f" : 0.01, "algo" : "StoMADS-PB",  "n_cores" : 16, "group" : 3},
    }

    # CovidSim
    opts_unscaled_CovidSim = scaling(opts_CovidSim, bounds_CovidSim[:3,0], bounds_CovidSim[:3,1], 2)

    i = 0
    for point in opts_unscaled_CovidSim:
        print('point #%i: Compliance = %f, Contact_rate = %f, Testing_capacity = %f' %(i+1,point[0],point[1],point[2])); i+=1

    # Points to plot
    lob_var_CovidSim = bounds_CovidSim[:,0] # lower bounds
    upb_var_CovidSim = bounds_CovidSim[:,1] # upper bounds

    # save optimization points in LHS format file
    with open('data/points_opts_CovidSim.pkl','wb') as fid:
        pickle.dump(lob_var_CovidSim, fid)
        pickle.dump(upb_var_CovidSim, fid)
        pickle.dump(opts_CovidSim, fid)
        pickle.dump(opts_unscaled_CovidSim, fid)
        pickle.dump(solution_dict, fid)

    # COVID_SIM_UI
    opts_unscaled = scaling(opts_CovidSim, bounds[:3,0], bounds[:3,1], 2)
    
    i = 0
    for point in opts_unscaled:
        print('point #%i: E = %f, S_D = %f, T = %f' %(i+1,point[0],point[1],point[2])); i+=1

    # Points to plot
    lob_var = bounds[:,0] # lower bounds
    upb_var = bounds[:,1] # upper bounds

    with open('data/points_opts.pkl','wb') as fid:
        pickle.dump(lob_var, fid)
        pickle.dump(upb_var, fid)
        pickle.dump(opts_CovidSim, fid)
        pickle.dump(opts_unscaled, fid)

    #===================================================================#
    n_samples = 10 # <<-------------------------- Edit the number of observations
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
        # run = 6
        # opts_unscaled = opts_unscaled[run:]
        # opts_unscaled_CovidSim = opts_unscaled_CovidSim[run:]

        # terminate MCS
        # run = 1
        # run_end = 1 + 1
        # opts_unscaled = opts_unscaled[run:run_end]
        # opts_unscaled_CovidSim = opts_unscaled_CovidSim[run:run_end]

        for point,point_CovidSim in zip(opts_unscaled,opts_unscaled_CovidSim):

            # Model variables
            # COVID_SIM_UI
            n_violators = round(point[0])
            SD = point[1]
            test_capacity = round(point[2])

            # CovidSim
            Compliance = point_CovidSim[0]
            Contact_rate = point_CovidSim[1]
            Testing_capacity = point_CovidSim[2]

            # Model parameters
            healthcare_capacity = 90
            healthcare_capacity_CovidSim = 0.09
            country = "Canada"
            pop_size_CovidSim = 36460098
            pop_size = 1000
            time_shift = 0

            #=====================================================================#
            # Design variables (COVID_SIM_UI)
            design_variables = [n_violators, SD, test_capacity]
            args = [design_variables] * n_samples # repeat variables by number of samples

            # Design variables (CovidSim)
            design_variables = [Compliance, Contact_rate, Testing_capacity]
            args_CovidSim = [design_variables] * n_samples # repeat variables by number of samples

            #=====================================================================#
            # Empty result lists
            infected_i = []; fatalities_i = []; GC_i = []; distance_i = []
            process_I = []; process_F = []; process_R = []; process_M = []; process_R0 = []

            infected_CovidSim = []; fatalities_CovidSim = []; process_I_CovidSim = []; process_F_CovidSim = []
            process_R_CovidSim = []; process_S_CovidSim = []; process_Critical_CovidSim = []

            # Blackbox set-up
            params_COVID_SIM_UI = [healthcare_capacity]
            params_CovidSim = [run, pop_size_CovidSim, healthcare_capacity_CovidSim, country]

            output_file_base = 'MCS_data_r%i' %run
            # return_process = False
            return_process = True
            params = [run, output_file_base, pop_size, time_shift, params_COVID_SIM_UI, return_process]

            ################################################################
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
            results = serial_sampling(args_CovidSim,params_CovidSim,blackbox_CovidSim)

            # Read results
            for result in results: 
                [infected, fatalities, I, F, R, S, Critical] = result

                infected_CovidSim           += [infected]
                fatalities_CovidSim         += [fatalities]

                process_I_CovidSim          += [I]
                process_F_CovidSim          += [F]
                process_R_CovidSim          += [R]
                process_S_CovidSim          += [S]
                process_Critical_CovidSim   += [Critical]

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

            with open('data/MCS_process_data_CovidSim_r%i.pkl' %run,'wb') as fid:    
                pickle.dump(process_I_CovidSim,fid)
                pickle.dump(process_F_CovidSim,fid)
                pickle.dump(process_R_CovidSim,fid)
                pickle.dump(process_S_CovidSim,fid)
                pickle.dump(process_Critical_CovidSim,fid)

            with open('data/MCS_data_CovidSim_r%i.pkl' %run,'wb') as fid:
                pickle.dump(infected_CovidSim,fid)
                pickle.dump(fatalities_CovidSim,fid)
            
            run += 1
            continue