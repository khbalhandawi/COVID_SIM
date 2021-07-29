import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import matplotlib.patches as patches
import sys

from functionsDesignSpace.stats_functions import plot_distribution

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #===================================================================#
    fit_cond = False # Do not fit data
    color_mode = 'color' # Choose color mode (black_white)
    run_index = 0 # starting point

    # load optimization points in LHS format file
    with open('data_vis/opts/points_opts.pkl','rb') as fid:
            lob_var = pickle.load(fid)
            upb_var = pickle.load(fid)
            opts = pickle.load(fid)
            opts_unscaled = pickle.load(fid)

    #===================================================================#
    # n_bins = 30 # for continuous distributions
    # min_bin_width_i = 15 # for discrete distributions
    # min_bin_width_f = 5 # for discrete distributions

    n_bins = 10 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    same_axis = True
    if same_axis:
        fig_infections = plt.figure(figsize=(6,5))
        plt.subplots_adjust(top = 0.95, bottom = 0.12, right = 0.95, left = 0.13, 
            hspace = 0.0, wspace = 0.0)
        fig_fatalities = plt.figure(figsize=(6,5))
        plt.subplots_adjust(top = 0.95, bottom = 0.12, right = 0.99, left = 0.13, 
            hspace = 0.0, wspace = 0.0)
        fig_dist = plt.figure(figsize=(6,5))
        plt.subplots_adjust(top = 0.95, bottom = 0.12, right = 0.99, left = 0.13, 
            hspace = 0.0, wspace = 0.0)
        fig_GC = plt.figure(figsize=(6,5))
        plt.subplots_adjust(top = 0.95, bottom = 0.12, right = 0.99, left = 0.13, 
            hspace = 0.0, wspace = 0.0)
    else:
        fig_infections = fig_fatalities = fig_dist = fig_GC = None

    auto_limits = "fixed" # also can be "fixed", "load"
    if auto_limits == "determine":
        dataXLim_i = dataYLim_i = None
        dataXLim_f = dataYLim_f = None
        dataXLim_d = dataYLim_d = None
        dataXLim_GC = dataYLim_GC = None
    elif auto_limits == "load":
        with open('data_vis/MCS_data_limits.pkl','rb') as fid:
            dataXLim_i = pickle.load(fid)
            dataYLim_i = pickle.load(fid)
            dataXLim_f = pickle.load(fid)
            dataYLim_f = pickle.load(fid)
            dataXLim_d = pickle.load(fid)
            dataYLim_d = pickle.load(fid)
            dataXLim_GC = pickle.load(fid)
            dataYLim_GC = pickle.load(fid)
    elif auto_limits == "fixed":
        dataXLim_i  = ( 0 , 800 )
        dataYLim_i  = ( 0 , 0.0325 )
        dataXLim_f  = ( 0 , 250  )
        dataYLim_f  = ( 0 , 0.2  )
        dataXLim_d  = ( 75, 175  )
        dataYLim_d  = ( 0 , 0.4  )
        dataXLim_GC = ( 0.6 , 2.8  )
        dataYLim_GC = ( 0 , 15   )

    mean_i_runs = []; std_i_runs = []; rel_i_runs = []; mean_f_runs = []; std_f_runs = []
    mean_d_runs = []; std_d_runs = []; mean_gc_runs = []; std_gc_runs = []

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams['font.family'] = 'serif'
    #===================================================================#
    # Initialize
    handles_lgd = []; labels_lgd = [] # initialize legend

    if color_mode == 'color':
        # hatches = ['/'] * 10
        hatches = ['///','++','xx','o','|||']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
        colors = colors * 30 # repeat colors 30 times
    elif color_mode == 'black_white':
        hatches = ['/','//','x','o','|||']
        colors = ['#FFFFFF'] * 300

    mpl.rcParams['hatch.linewidth'] = 0.05  # previous pdf hatch linewidth

    # for plotting demo points
    # select_indices = [0,1,2,3]
    # zorders = [3,2,1,0]
    # prefixes = ['',] * len(select_indices)
    # y_label_add = ''
    # # y_label_add = ' ($\%$ population)'
    # healthcare_capacity = 90
    # runs = select_indices
    # discrete = True
    # scaling = 1000

    # for plotting best optimization results
    select_indices = [0,1,2]
    zorders = [3,2,1]
    prefixes = ['StoMADS-PB solution', 'GA solution', 'NOMAD solution']
    y_label_add = ''
    # y_label_add = ' ($\%$ population)'
    transparency = 0.4
    healthcare_capacity = 90
    runs = select_indices
    discrete = True
    scaling = 1000

    # for plotting CovidSim results
    # select_indices = [2,3,]
    # zorders = [1,2]
    # prefixes = ['',] * len(select_indices)
    # y_label_add = ' ($\%$ population)'
    # transparency = 0.4
    # healthcare_capacity = 0.09
    # runs = select_indices
    # discrete = False
    # scaling = 1

    opts_unscaled = [opts_unscaled[index] for index in select_indices]

    for point,run,prefix,zorder in zip(opts_unscaled,runs,prefixes,zorders):

        # Legend labels
        print('point #%i: E = %f, S_D = %f, T = %f' %(run+1,point[0],point[1],point[2]))
        # Legend labels
        # prefix = "Solution %i" %(run_index+1) # uncomment for generic legend labels
        # prefix = "Run %i" %(run+1) # uncomment for generic legend labels

        legend_label = "%s: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(prefix,round(point[0]),point[1],round(point[2])) # generic legend labels

        # Model variables
        n_violators = round(point[0])
        SD = point[1]
        test_capacity = round(point[2])

        with open('data_vis/opts/MCS_data_r%i.pkl' %run,'rb') as fid:
            infected_i = pickle.load(fid)
            fatalities_i = pickle.load(fid)
            GC_i = pickle.load(fid)
            distance_i = pickle.load(fid)

        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='k', facecolor=colors[run_index], fill=True ,hatch=hatches[run])

        handles_lgd += [a]
        labels_lgd += [legend_label]

        # Infected plot
        label_name = u'Maximum number of infections $n_{I,\mathrm{max}}$'
        fun_name = 'infections'
        data = infected_i

        # pass lists by value instead of by reference as python does by default
        handles_lgd_i = handles_lgd[:]
        labels_lgd_i = labels_lgd[:]

        [dataXLim_i_out, dataYLim_i_out, mean_i, std_i, rel_i] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = discrete, min_bin_width = min_bin_width_i, fig_swept = fig_infections, 
            run_label = legend_label, color = colors[run_index],  hatch_pattern = hatches[run], 
            dataXLim = dataXLim_i, dataYLim = dataYLim_i, constraint = healthcare_capacity,
            fit_distribution = fit_cond, handles = handles_lgd_i, labels = labels_lgd_i, 
            scaling = scaling, transparency = transparency, zorder = zorder)

        mean_i_runs += [mean_i]
        std_i_runs += [std_i]
        rel_i_runs += [rel_i]

        # Fatalities plot
        label_name = u'Number of fatalities $n_I^T$'
        fun_name = 'fatalities'
        data = fatalities_i
        
        [dataXLim_f_out, dataYLim_f_out, mean_f, std_f, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = discrete, min_bin_width = min_bin_width_f, fig_swept = fig_fatalities, 
            run_label = legend_label, color = colors[run_index], hatch_pattern = hatches[run], 
            dataXLim = dataXLim_f, dataYLim = dataYLim_f, constraint = None,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd, 
            scaling = scaling, transparency = transparency, zorder = zorder)

        mean_f_runs += [mean_f]
        std_f_runs += [std_f]

        # Distance plot
        label_name = u'Average cumulative distance travelled $D(\mathbf{x})$'
        fun_name = 'distance'
        data = distance_i

        [dataXLim_d_out, dataYLim_d_out, mean_d, std_d, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_dist, run_label = legend_label, color = colors[run_index], constraint = None,
            hatch_pattern = hatches[run], dataXLim = dataXLim_d, dataYLim = dataYLim_d,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd, 
            transparency = transparency, zorder = zorder)

        mean_d_runs += [mean_d]
        std_d_runs += [std_d]

        label_name = u'Mobility $M^T$'
        fun_name = 'ground covered'
        data = GC_i

        # Ground covered plot
        [dataXLim_GC_out, dataYLim_GC_out, mean_gc, std_gc, _] = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_GC, run_label = legend_label, color = colors[run_index], constraint = None, 
            hatch_pattern = hatches[run], dataXLim = dataXLim_GC, dataYLim = dataYLim_GC,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd, 
            transparency = transparency)

        mean_gc_runs += [mean_gc]
        std_gc_runs += [std_gc]

        run_index += 1
        continue
        if auto_limits != "determine":
            fig_infections.savefig('data_vis/%i_PDF_%s.eps' %(run , 'infections'), 
                                    format='pdf', dpi=100,bbox_inches='tight')
            fig_fatalities.savefig('data_vis/%i_PDF_%s.eps' %(run , 'fatalities'), 
                                format='pdf', dpi=100,bbox_inches='tight')
            fig_dist.savefig('data_vis/%i_PDF_%s.eps' %(run , 'distance'), 
                            format='pdf', dpi=100,bbox_inches='tight')
            fig_GC.savefig('data_vis/%i_PDF_%s.eps' %(run , 'ground_covered'), 
                            format='pdf', dpi=100,bbox_inches='tight')

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean infections: %f; std infections: %f; rel infections: %f' %(mean_i,std_i,rel_i))
        print('mean fatalities: %f; std fatalities: %f' %(mean_f,std_f))
        print('mean distance: %f; std distance: %f' %(mean_d,std_d))
        print('mean ground covered: %f; std ground covered: %f' %(mean_gc,std_gc))
        print('==============================================')

    with open('data_vis/MCS_data_limits.pkl','wb') as fid:
        pickle.dump(dataXLim_i_out,fid)
        pickle.dump(dataYLim_i_out,fid)
        pickle.dump(dataXLim_f_out,fid)
        pickle.dump(dataYLim_f_out,fid)
        pickle.dump(dataXLim_d_out,fid)
        pickle.dump(dataYLim_d_out,fid)
        pickle.dump(dataXLim_GC_out,fid)
        pickle.dump(dataYLim_GC_out,fid)

    with open('data_vis/opts/MCS_data_stats_opts.pkl','wb') as fid:
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
        fig_infections.savefig('data_vis/PDF_%s_opt.eps' %('infections'), 
                                format='eps', dpi=100,bbox_inches=None)
        fig_fatalities.savefig('data_vis/PDF_%s_opt.eps' %('fatalities'), 
                            format='eps', dpi=100,bbox_inches=None)
        fig_dist.savefig('data_vis/PDF_%s_opt.eps' %('distance'), 
                        format='eps', dpi=100,bbox_inches=None)
        fig_GC.savefig('data_vis/PDF_%s_opt.eps' %('ground_covered'), 
                       format='eps', dpi=100,bbox_inches=None)
        plt.show()