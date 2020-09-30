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

#==============================================================================#
# Scaling by a range
def scaling(x,l,u,operation):
    # scaling() scales or unscales the vector x according to the bounds
    # specified by u and l. The flag type indicates whether to scale (1) or
    # unscale (2) x. Vectors must all have the same dimension.
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

#==============================================================================#
# Execute system commands and return output to console
def system_command(command):

    #CREATE_NO_WINDOW = 0x08000000 # Create no console window flag

    p = subprocess.Popen(command,shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                         ) # disable windows errors

    for line in iter(p.stdout.readline, b''):
        line = line.decode('utf-8')
        print(line.rstrip()) # print line by line
        # rstrip() to remove \n separator

    output, error = p.communicate()
    if p.returncode != 0: # crash the program
        print("Exception during run: %s" %command)
        raise Exception("cpp failed %d %s %s" % (p.returncode, output, error))

#==============================================================================#
# C++ COMMAND
def cpp_application( i, design_variables, parameters, output_file_n, debug = False ):
    design_variables_str = ' '.join(map(str,design_variables)) # print variables as space delimited string
    parameters_str = ' '.join(map(str,parameters)) # print parameters as space delimited string
    command = "cpp_corona_simulation %i %s %s %s" %(i, design_variables_str, parameters_str, output_file_n)
    
    print(command)
    if not debug:
        system_command(command)

#==============================================================================#
# Function to parallelize
def processInput(procnum, design_variables, parameters, output_file_base):

    output_file_n = '%s_%i.log' %(output_file_base, procnum)
    cpp_application(procnum,design_variables,parameters,output_file_n)
    #--------------------------------------------------------------------------#
    # Get results
    output_file_path = "data/" + output_file_n

    with open(output_file_path, 'r') as fh:
        for line in fh:
            pass
        last = line

    values = last.split(',')
    [ _, _, _, _, _, infected, fatalities, mean_distance, mean_GC, _ ] = values

    return [int(infected), int(fatalities), float(mean_GC), float(mean_distance)]

#==============================================================================#
# Parallel sampling of blackbox
def parallel_sampling(design_variables,parameters,output_file_base,n_samples):
    from joblib import Parallel, delayed
    import multiprocessing
    from multiprocessing import Process, Pool
    import subprocess
    
    num_threads = multiprocessing.cpu_count() - 2

    # qout = multiprocessing.Queue()
    # processes = [multiprocessing.Process(target=processInput, args=(i, design_variables, parameters, output_file_base, qout)) for i in range(n_samples)]

    args = []
    for i in range(n_samples):
        args += [(i,design_variables,parameters,output_file_base)]

    with Pool(num_threads) as pool:
        results = pool.starmap(processInput, args)

    # for p in processes:
    #     p.start()

    # for p in processes:
    #     p.join()

    # results = [qout.get() for p in processes]

    infected_i = []; fatalities_i = []; GC_i = []; distance_i = []
    for result in results:

        infected_i += [result[0]]
        fatalities_i += [result[1]]
        GC_i += [result[2]]
        distance_i += [result[3]]

    return infected_i, fatalities_i, GC_i, distance_i

#==============================================================================#
# Serial sampling of blackbox
def serial_sampling(design_variables, parameters, output_file_base, n_samples):

    infected_i = []; fatalities_i = []; GC_i = []; distance_i = []

    for i in range(n_samples):  
        [infected, fatalities, mean_GC, mean_distance] = processInput(i, design_variables, parameters, output_file_base)

        infected_i += [infected]
        fatalities_i += [fatalities]
        GC_i += [mean_GC]
        distance_i += [mean_distance]

    return infected_i, fatalities_i, GC_i, distance_i

#==============================================================================#
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # DISTRIBUTIONS = [        
    #     st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #     st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #     st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #     st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #     st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #     st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
    #     st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #     st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #     st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #     st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    # ]

    # DISTRIBUTIONS = [     
    #     st.pearson3, st.johnsonsu, st.nct, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
    #     st.tukeylambda, st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
    #     st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
    #     st.alpha, st.norm
    # ]

    DISTRIBUTIONS = [     
        st.pearson3, st.johnsonsu, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
        st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
        st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
        st.alpha, st.norm
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    sse_d = []; name_d = []
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # print(distribution.name)
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)

                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                sse_d += [sse]
                name_d += [distribution.name]

        except Exception:
            pass
        
    sse_d, name_d = (list(t) for t in zip(*sorted(zip(sse_d, name_d))))
    
    return (best_distribution.name, best_params, name_d[:6])

#==============================================================================#
# Generate pdf functions
def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

#==============================================================================#
# Plot pdf function
def plot_distribution(data, fun_name, label_name, n_bins, run, 
                      discrete = False, min_bin_width = 0, 
                      fig_swept = None, run_label = 'PDF', color = u'b',
                      dataXLim = None, dataYLim = None, constraint = None,
                      fit_distribution = True, handles = [], labels = []):

    if constraint is not None:
        data_cstr = [d - constraint for d in data]
        mean_data = np.mean(data_cstr)
        std_data = np.std(data_cstr)
    else:
        mean_data = np.mean(data)
        std_data = np.std(data)

    # Plot raw data
    fig0 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        d = max(min(np.diff(np.unique(np.asarray(data)))), min_bin_width)
        left_of_first_bin = min(data) - float(d)/2
        right_of_last_bin = max(data) + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)

    ax = fig0.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Plot for comparison
    fig1 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)
    
    ax = fig1.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Display
    if fig_swept is None:
        fig2 = plt.figure(figsize=(6,5))
    else:
        fig2 = fig_swept
    
    ax2 = fig2.gca()

    if discrete:
        data_bins = bins
    else:
        data_bins = n_bins

    # Fit and plot distribution
    if fit_distribution:

        best_fit_name, best_fit_params, best_10_fits = best_fit_distribution(data, data_bins, ax)

        best_dist = getattr(st, best_fit_name)
        print('Best fit: %s' %(best_fit_name.upper()) )
        # Make PDF with best params 
        pdf = make_pdf(best_dist, best_fit_params)
        pdf.plot(lw=2, color = color, label=run_label, legend=True, ax=ax2)

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        handles = []; labels = []
    else:
        lgd = ax2.legend(handles, labels, fontsize = 9.0)

    if discrete:
        # discrete bin numbers
        ax2.hist(data, bins, color = color, alpha=0.5, label = 'data', density=True)
    else:
        ax2.hist(data, bins = n_bins, color = color, alpha=0.5, label = 'data', density=True)
    
    # plot constraint limits
    if constraint is not None:
        ax2.axvline(x=constraint, linestyle='--', linewidth='2', color='k')

    # Save plot limits
    if dataYLim is None and dataXLim is None:
        dataYLim = ax2.get_ylim()
        dataXLim = ax2.get_xlim()
    else:
        # Update plots
        ax2.set_xlim(dataXLim)
        ax2.set_ylim(dataYLim)

    ax2.tick_params(axis='both', which='major', labelsize=14) 
    ax2.set_xlabel(label_name, fontsize=14)
    ax2.set_ylabel('Probability density', fontsize=14)

    fig0.savefig('data/RAW_%s_r%i.pdf' %(fun_name,run), 
        format='pdf', dpi=100,bbox_inches='tight')
    
    if fig_swept is None:
        fig2.savefig('data/PDF_%s_r%i.pdf' %(fun_name,run), 
                format='pdf', dpi=100,bbox_inches='tight')

    if fig_swept is None:    
        plt.close('all')
    else:
        plt.close(fig0)
        plt.close(fig1)
    
    return dataXLim, dataYLim, mean_data, std_data

#==============================================================================#
# %% Main execution
if __name__ == '__main__':

    #=====================================================================#
    run = 0

    # n_samples = 1000
    # n_bins = 30 # for continuous distributions
    # min_bin_width_i = 15 # for discrete distributions
    # min_bin_width_f = 5 # for discrete distributions

    n_samples = 500
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = True

    # Model variables
    # bounds = np.array([[16      , 101 ],  # Essential workers
    #                    [0.0001  , 0.2 ],  # SD_factor
    #                    [10      , 51  ]]) # testing capacity)
    bounds = np.array([[16      , 151 ],  # Essential workers
                       [0.0001  , 0.2 ],  # SD_factor
                       [10      , 101 ]]) # testing capacity)

    fit_cond = True # Do not fit data
    run = 0 # starting point

    #===================================================================#
    # DOE levels
    n_var = 2; n_samples = 1000; n_steps = 5
    var_DOE = np.linspace(0.0,1.0,n_steps)
    var_DOE = scaling(var_DOE,bounds[n_var,0],bounds[n_var,1],2)

    same_axis = True
    if same_axis:
        fig_infections = plt.figure(figsize=(10,5))
        fig_fatalities = plt.figure(figsize=(10,5))
        fig_dist = plt.figure(figsize=(10,5))
        fig_GC = plt.figure(figsize=(10,5))
    else:
        fig_infections = fig_fatalities = fig_dist = fig_GC = None

    auto_limits = True
    if auto_limits:
        dataXLim_i = dataYLim_i = None
        dataXLim_f = dataYLim_f = None
        dataXLim_d = dataYLim_d = None
        dataXLim_GC = dataYLim_GC = None
    else:
        with open('data/MCS_data_limits.pkl','rb') as fid:
            dataXLim_i = pickle.load(fid)
            dataYLim_i = pickle.load(fid)
            dataXLim_f = pickle.load(fid)
            dataYLim_f = pickle.load(fid)
            dataXLim_d = pickle.load(fid)
            dataYLim_d = pickle.load(fid)
            dataXLim_GC = pickle.load(fid)
            dataYLim_GC = pickle.load(fid)

    mean_i_runs = []; std_i_runs = []; mean_f_runs = []; std_f_runs = []
    mean_d_runs = []; std_d_runs = []; mean_gc_runs = []; std_gc_runs = []

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]

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

    # Resume MCS
    # run = 1
    # var_DOE = var_DOE[run:]

    # terminate MCS
    # run = 3
    # run_end = 3 + 1
    # var_DOE = var_DOE[run:run_end]

    # design parameters
    healthcare_capacity = 150

    for var in var_DOE:

        legend_labels = ['Number of essential workers ($E$) = %i people' %(var),
                         'Social distancing factor ($S_D$)= %f' %(var),
                         'Testing capacity ($T$) = %i people' %(var)]

        legend_label = legend_labels[n_var] 

        if new_run:
            if n_var == 0:
                #=====================================================================#
                # Essential workers sweep
                SD = 0.1 # force amplitude
                test_capacity = 0 # number of people

                design_variables = [int(var), SD, test_capacity]
                parameters = [healthcare_capacity]

            elif n_var == 1:
                #=====================================================================#
                # SD sweep
                n_violators = 0 # number of people
                test_capacity = 0 # number of people

                design_variables = [n_violators, var, test_capacity]
                parameters = [healthcare_capacity]

            elif n_var == 2:
                #=====================================================================#
                # Testing sweep
                n_violators = 0 # number of people
                SD = 0.1 # force amplitude

                design_variables = [n_violators, SD, int(var)]
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
        else:
            with open('data/MCS_data_r%i.pkl' %run,'rb') as fid:
                infected_i = pickle.load(fid)
                fatalities_i = pickle.load(fid)
                GC_i = pickle.load(fid)
                distance_i = pickle.load(fid)

        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor=colors[run], facecolor=colors[run], fill='None' ,alpha=0.5)
        handles_lgd += [a]
        labels_lgd += [legend_label]

         # Infected plot
        label_name = u'Maximum number of infected $I(\mathbf{x})$'
        fun_name = 'infections'
        data = infected_i

        dataXLim_i_out, dataYLim_i_out, mean_i, std_i = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_i, fig_swept = fig_infections, 
            run_label = legend_label, color = colors[run], dataXLim = dataXLim_i, dataYLim = dataYLim_i,
            constraint = None, fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_i_runs += [mean_i]
        std_i_runs += [std_i]

        # Fatalities plot
        label_name = u'Number of fatalities $F(\mathbf{x})$'
        fun_name = 'fatalities'
        data = fatalities_i

        dataXLim_f_out, dataYLim_f_out, mean_f, std_f = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_f, fig_swept = fig_fatalities, 
            run_label = legend_label, color = colors[run], dataXLim = dataXLim_f, dataYLim = dataYLim_f,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_f_runs += [mean_f]
        std_f_runs += [std_f]

        # Distance plot
        label_name = u'Average distance travelled $D(\mathbf{x})$'
        fun_name = 'distance'
        data = distance_i

        dataXLim_d_out, dataYLim_d_out, mean_d, std_d = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_dist, run_label = legend_label, color = colors[run], 
            dataXLim = dataXLim_d, dataYLim = dataYLim_d,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)
        
        mean_d_runs += [mean_d]
        std_d_runs += [std_d]

        # Ground covered plot
        label_name = u'Mobility $D(\mathbf{x})$ ($\%$)'
        fun_name = 'ground covered'
        data = GC_i

        dataXLim_GC_out, dataYLim_GC_out, mean_gc, std_gc = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_GC, run_label = legend_label, color = colors[run], 
            dataXLim = dataXLim_GC, dataYLim = dataYLim_GC,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_gc_runs += [mean_gc]
        std_gc_runs += [std_gc]

        if not auto_limits:
            fig_infections.savefig('data/PDF_%s_r%i.pdf' %('infections', run + 1), 
                                    format='pdf', dpi=100,bbox_inches='tight')
            fig_fatalities.savefig('data/PDF_%s_r%i.pdf' %('fatalities', run + 1), 
                                format='pdf', dpi=100,bbox_inches='tight')
            fig_dist.savefig('data/PDF_%s_r%i.pdf' %('distance', run + 1), 
                            format='pdf', dpi=100,bbox_inches='tight')
            fig_GC.savefig('data/PDF_%s_r%i.pdf' %('ground_covered', run + 1), 
                            format='pdf', dpi=100,bbox_inches='tight')

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean infections: %f; std infections: %f' %(mean_i,std_i))
        print('mean fatalities: %f; std fatalities: %f' %(mean_f,std_f))
        print('mean distance: %f; std distance: %f' %(mean_d,std_d))
        print('mean ground covered: %f; std ground covered: %f' %(mean_gc,std_gc))
        print('==============================================')
        run += 1

    with open('data/MCS_data_limits.pkl','wb') as fid:
        pickle.dump(dataXLim_i_out,fid)
        pickle.dump(dataYLim_i_out,fid)
        pickle.dump(dataXLim_f_out,fid)
        pickle.dump(dataYLim_f_out,fid)
        pickle.dump(dataXLim_d_out,fid)
        pickle.dump(dataYLim_d_out,fid)
        pickle.dump(dataXLim_GC_out,fid)
        pickle.dump(dataYLim_GC_out,fid)

    with open('data/MCS_data_stats.pkl','wb') as fid:
        pickle.dump(mean_i_runs,fid)
        pickle.dump(std_i_runs,fid)
        pickle.dump(mean_f_runs,fid)
        pickle.dump(std_f_runs,fid)
        pickle.dump(mean_d_runs,fid)
        pickle.dump(std_d_runs,fid)
        pickle.dump(mean_gc_runs,fid)
        pickle.dump(std_gc_runs,fid)

    if same_axis:
        fig_infections.savefig('data/PDF_%s.pdf' %('infections'), 
                                format='pdf', dpi=100,bbox_inches='tight')
        fig_fatalities.savefig('data/PDF_%s.pdf' %('fatalities'), 
                            format='pdf', dpi=100,bbox_inches='tight')
        fig_dist.savefig('data/PDF_%s.pdf' %('distance'), 
                        format='pdf', dpi=100,bbox_inches='tight')
        fig_GC.savefig('data/PDF_%s.pdf' %('ground_covered'), 
                       format='pdf', dpi=100,bbox_inches='tight')
        plt.show()