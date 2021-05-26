import pickle

from functionsDesignSpace.stats_functions import LHS_sampling,statistics

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #===================================================================#
    # LHS search

    fit_cond = False # Do not fit data
    color_mode = 'color' # Choose color mode (black_White)
    run = 0 # starting point
    
    n_samples_LH = 300 # <-------------------------- adjust the number of observations according to MCS_model_LHS

    # LHS distribution
    [lob_var, upb_var, _,points] = LHS_sampling(n_samples_LH,new_LHS=False)

    run = 0 # starting point

    #===================================================================#
    mean_i_runs = []; std_i_runs = []; rel_i_runs = []; mean_f_runs = []; std_f_runs = []
    mean_d_runs = []; std_d_runs = []; mean_gc_runs = []; std_gc_runs = []

    #===================================================================#
    # Initialize

    # terminate MCS
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

        with open('data_vis/LHS/MCS_data_r%i.pkl' %run,'rb') as fid:
            infected_i = pickle.load(fid)
            fatalities_i = pickle.load(fid)
            GC_i = pickle.load(fid)
            distance_i = pickle.load(fid)
                
        # Infected plot
        data = infected_i

        [mean_i, std_i, rel_i] = statistics(data, constraint = healthcare_capacity)

        mean_i_runs += [mean_i]
        std_i_runs += [std_i]
        rel_i_runs += [rel_i]

        # Fatalities plot
        label_name = u'Number of fatalities $F(\mathbf{x})$'
        fun_name = 'fatalities'
        data = fatalities_i

        [mean_f, std_f, rel_f] = statistics(data)

        mean_f_runs += [mean_f]
        std_f_runs += [std_f]

        # Distance plot
        data = distance_i

        [mean_d, std_d, _] = statistics(data)

        mean_d_runs += [mean_d]
        std_d_runs += [std_d]

        # Ground covered plot
        data = GC_i

        [mean_gc, std_gc, _] = statistics(data)

        mean_gc_runs += [mean_gc]
        std_gc_runs += [std_gc]

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean infections: %f; std infections: %f; rel infections: %f' %(mean_i,std_i,rel_i))
        print('mean fatalities: %f; std fatalities: %f' %(mean_f,std_f))
        print('mean distance: %f; std distance: %f' %(mean_d,std_d))
        print('mean ground covered: %f; std ground covered: %f' %(mean_gc,std_gc))
        print('==============================================')
        run += 1

    with open('data_vis/LHS/MCS_data_stats.pkl','wb') as fid:
        pickle.dump(mean_i_runs,fid)
        pickle.dump(std_i_runs,fid)
        pickle.dump(rel_i_runs,fid)
        pickle.dump(mean_f_runs,fid)
        pickle.dump(std_f_runs,fid)
        pickle.dump(mean_d_runs,fid)
        pickle.dump(std_d_runs,fid)
        pickle.dump(mean_gc_runs,fid)
        pickle.dump(std_gc_runs,fid)