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

#==============================================================================#
# SCALING BY A RANGE
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
# Load an output array from COVID application
def load_matrix(filename, folder='data_tstep'):
    '''loads tracking grid coordinates from disk

    Function that loads the tracking grid coordinates from specific files on the disk.
    Loads the state of the grid_coords matrix

    Keyword arguments
    -----------------

    tstep : int
        the timestep that will be saved
    ''' 
    matrix = np.loadtxt('%s/%s.bin' %(folder,filename))
    return matrix

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
        raise Exception("cpp failed %d %s %s" % (p.returncode, output, error))

#==============================================================================#
# C++ COMMAND
def cpp_application( i, design_variables, parameters, output_file_n, debug = False ):
    design_variables_str = ' '.join(map(str,design_variables)) # print variables as space delimited string
    parameters_str = ' '.join(map(str,parameters)) # print parameters as space delimited string
    command = "COVID_SIM_UI %i %s %s %s" %(i, design_variables_str, parameters_str, output_file_n)
    
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
    
    # num_threads = int(multiprocessing.cpu_count()/2)
    num_threads = 6

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
    process_I = []; process_F = []; process_R = []; process_M = []; process_R0 = []
    for i in range(n_samples):  
        [infected, fatalities, mean_GC, mean_distance] = processInput(i, design_variables, parameters, output_file_base)

        infected_i += [infected]
        fatalities_i += [fatalities]
        GC_i += [mean_GC]
        distance_i += [mean_distance]

        run_data = load_matrix('SIRF_data', folder='population')
        run_data_M = load_matrix('mean_GC_data', folder='population')
        run_data_R0 = load_matrix('mean_R0_data', folder='population')
        I = run_data[:,2]; R = run_data[:,3]; F = run_data[:,4]; M = [x * 100 for x in run_data_M[:,1]]

        process_I += [I]; process_F += [F]; process_R += [R]; process_M += [M]; process_R0 += [run_data_R0]

    return infected_i, fatalities_i, GC_i, distance_i, process_I, process_F, process_R, process_M, process_R0

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
                      fig_swept = None, run_label = 'PDF', color = u'b', hatch_pattern = u'',
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

        # plt.hist(data, bins, alpha=0.5, density=True)
        plt.hist(data, bins = n_bins, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)
    else:
        # plt.hist(data, bins = n_bins, alpha=0.5, density=True)
        plt.hist(data, bins = n_bins, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)

    ax = fig0.gca()

    # Save plot limits
    if dataYLim is None and dataXLim is None:
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    else:
        # Update plots
        ax.set_xlim(dataXLim)
        ax.set_ylim(dataYLim)

    # Update plots
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
        # ax2.hist(data, bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data, bins, linewidth=2, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)
    else:
        # ax2.hist(data, bins = n_bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data, bins = n_bins, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)
    
    # plot constraint limits
    if constraint is not None:
        ax2.axvline(x=constraint, linestyle='--', linewidth='2', color='r')
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

    fig0.savefig('data/%i_RAW_%s.pdf' %(run,fun_name), 
        format='pdf', dpi=100,bbox_inches='tight')
    
    if fig_swept is None:
        fig2.savefig('data/%i_PDF_%s.pdf' %(run,fun_name), 
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

    #===================================================================#
    
    # Model variables
    bounds = np.array([[   16    , 101   ], # number of essential workers
                       [   0.0001, 0.15  ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    run = 0 # starting point

    # Points to plot
    # Demo points
    # opts = np.array([[0.164705882 , 0.666444296 , 0.000000000 ],
    #                  [0.517647058 , 0.666444296 , 0.000000000 ],
    #                  [0.164705882 , 0.833222148 , 0.000000000 ],
    #                  [0.164705882 , 0.666444296 , 0.243902439 ]])

    # Points to plot
    # StoMADS V2
    opts = np.array([[0.6203810000, 0.2954270000, 0.9158810000],
                     [0.6164750000, 0.2946950000, 0.9184600000],
                     [0.6360060000, 0.3064140000, 0.9809600000],
                     [0.6277050000, 0.3103200000, 0.9299350000],
                     [0.6194040000, 0.3039720000, 0.9028350000],
                     [0.6438180000, 0.3571950000, 0.8989290000],
                     [0.8664750000, 0.4196950000, 0.9809600000]])

    # # NOMAD
    # opts = np.array([[ 0.998976, 0.0899023, 0.970503 ]])

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
    n_samples = 500
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
        run = 2
        opts_unscaled = opts_unscaled[run:]

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
            healthcare_capacity = 150

            #=====================================================================#
            # Design variables
            design_variables = [n_violators, SD, test_capacity]
            parameters = [healthcare_capacity]

            #=====================================================================#
            output_file_base = 'MCS_data_r%i' %run
            # [infected_i,fatalities_i,GC_i,distance_i] = parallel_sampling(design_variables,parameters,output_file_base,n_samples)
            [infected_i,fatalities_i,GC_i,distance_i,process_I,process_F,process_R,process_M,process_R0] = serial_sampling(design_variables,parameters,output_file_base,n_samples)

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
