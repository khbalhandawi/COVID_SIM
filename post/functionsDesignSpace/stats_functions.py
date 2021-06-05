from pyDOE import lhs
import numpy as np
import pickle
import matplotlib.pyplot as plt

from functionsUtilities.utils import scaling, check_folder

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
                      fit_distribution = True, handles = [], labels = [],
                      scaling=1,transparency=0.5,zorder=0):

    if constraint is not None:
        # data_f = [d*scaling - constraint for d in data] # for demo points only
        data_f = [d*scaling for d in data] # for everything else
        data_p = [d*scaling - constraint for d in data] # for everything else
        rel_data = sum(map(lambda x : x < 0, data_p)) / len(data_p)
        mean_data = np.mean(data_f)
        std_data = np.std(data_f,ddof=1)
    else:
        data_f = [d*scaling for d in data]
        rel_data = 0.0
        mean_data = np.mean(data_f)
        std_data = np.std(data_f,ddof=1)

    # Plot raw data
    fig0 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        d = max(min(np.diff(np.unique(np.asarray(data_f)))), min_bin_width)
        left_of_first_bin = min(data_f) - float(d)/2
        right_of_last_bin = max(data_f) + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        # plt.hist(data, bins, alpha=0.5, density=True)
        plt.hist(data_f, bins = n_bins, facecolor=color, alpha=0.5,
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True, linewidth=0.3)
    else:
        # plt.hist(data, bins = n_bins, alpha=0.5, density=True)
        plt.hist(data_f, bins = n_bins, facecolor=color, alpha=0.5,
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True, linewidth=0.3)

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
        plt.hist(data_f, bins, alpha=0.5, density=True)
    else:
        plt.hist(data_f, bins = n_bins, alpha=0.5, density=True)
    
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

    # plot constraint limits
    if constraint is not None:
        print("Healthcare capacity: %f" %constraint)
        a_cstr = ax2.axvline(x=constraint, linestyle='--', linewidth='2', color='k')
        handles += [a_cstr]; labels += ['Healthcare capacity $H_{\mathrm{max}}$']

    if discrete:
        data_bins = bins
    else:
        data_bins = n_bins

    # Fit and plot distribution
    if fit_distribution:

        best_fit_name, best_fit_params, best_10_fits = best_fit_distribution(data_f, data_bins, ax)

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
        if labels[0] != None:
            lgd = ax2.legend(handles, labels, fontsize = 9.0)

    if discrete:
        # discrete bin numbers
        # ax2.hist(data, bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data_f, bins, linewidth=0.5, facecolor=color, alpha=transparency,
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True,zorder=zorder)
    else:
        # ax2.hist(data, bins = n_bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data_f, bins = n_bins, linewidth=0.5, facecolor=color, alpha=transparency, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True,zorder=zorder)
    
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
    ax2.set_ylabel('Relative frequency', fontsize=14)

    fig0.savefig('data_vis/%i_RAW_%s.pdf' %(run,fun_name), 
        format='pdf', dpi=100,bbox_inches='tight')
    
    if fig_swept is None:
        fig2.savefig('data_vis/%i_PDF_%s.pdf' %(run,fun_name), 
                format='pdf', dpi=100,bbox_inches='tight')

    if fig_swept is None:    
        plt.close('all')
    else:
        plt.close(fig0)
        plt.close(fig1)
    
    return dataXLim, dataYLim, mean_data, std_data, rel_data

#==============================================================================
# Create or retrieve LHS data
def LHS_sampling(n_samples, lob_var=None, upb_var=None, folder='data/',base_name='LHS_points', new_LHS=False,):
    # LHS distribution
    if new_LHS and lob_var is not None and upb_var is not None :
        points = lhs(len(lob_var), samples=n_samples, criterion='maximin') # generate LHS grid (maximin criterion)
        points_us = scaling(points,lob_var,upb_var,2) # unscale latin hypercube points
        
        check_folder(folder)

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
# generate an n-dimensional grid
def gridsamp(bounds, q):
	import numpy as np
	'''
	GRIDSAMP  n-dimensional grid over given range

	Call:    S = gridsamp(bounds, q)

	bounds:  2*n matrix with lower and upper limits
	q     :  n-vector, q(j) is the number of points
	         in the j'th direction.
	         If q is a scalar, then all q(j) = q
	S     :  m*n array with points, m = prod(q)

	hbn@imm.dtu.dk  
	Last update June 25, 2002
	'''
	
	[mr,n] = np.shape(bounds);    dr = np.diff(bounds, axis=0)[0]; # difference across rows
	if  mr != 2 or any([item < 0 for item in dr]):
	  raise Exception('bounds must be an array with two rows and bounds(1,:) <= bounds(2,:)')
	 
	if  q.ndim > 1 or any([item <= 0 for item in q]):
	  raise Exception('q must be a vector with non-negative elements')
	
	p = len(q);   
	if  p == 1:
		q = np.tile(q, (1, n))[0]; 
	elif  p != n:
	  raise Exception(sprintf('length of q must be either 1 or %d',n))
	 
	
	# Check for degenerate intervals
	i = np.where(dr == 0)[0]
	if  i.size > 0:
		q[i] = 0*q[i]; 
	
	# Recursive computation
	if  n > 1:
		A = gridsamp(bounds[:,1::], q[1::]);  # Recursive call
		[m,p] = np.shape(A);
		q = q[0];
		S = np.concatenate((np.zeros((m*q,1)), np.tile(A, (q, 1))),axis=1);
		y = np.linspace(bounds[0,0],bounds[1,0], q);
		
		k = range(m);
		for i in range(q):
			aug = np.tile(y[i], (m, 1))
			aug = np.reshape(aug, S[k,0].shape)
			
			S[k,0] = aug;
			k = [item + m for item in k];
	else:
		S = np.linspace(bounds[0,0],bounds[1,0],q[-1])
		S = np.transpose([S])
		
	return S

#==============================================================================#
# Get summary statistics of MCS run
def statistics(data, constraint = None,):

    if constraint is not None:
        data_cstr = [d - constraint for d in data]
        rel_data = sum(map(lambda x : x < 0, data_cstr)) / len(data_cstr)
        mean_data = np.mean(data_cstr)
        std_data = np.std(data_cstr,ddof=1)
    else:
        rel_data = 0.0
        mean_data = np.mean(data)
        std_data = np.std(data,ddof=1)

    return mean_data, std_data, rel_data