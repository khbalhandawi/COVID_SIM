# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:35:50 2017

@author: Khalil
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.special import comb
from itertools import combinations
import matplotlib.gridspec as gridspec

from functionsUtilities.utils import scaling
from functionsDesignSpace.stats_functions import gridsamp
from functionsDesignSpace.SGTE_library import SGTE_server

#==============================================================================#
# Retrieve hyperparameters for an existing SGTE model
def get_SGTE_model(out_file):
	
	current_dir = os.getcwd()
	filepath = os.path.join(current_dir,'SGTE_matlab_server',out_file)
	
	# Get matrices names from the file
	NAMES = [];
	fileID = open(filepath,'r'); # Open file
	InputText = np.loadtxt(fileID,
					   delimiter = '\n',
					   dtype=np.str); # \n is the delimiter

	for n,line in enumerate(InputText): # Read line by line
		#Look for object
		if line.find('Surrogate: ') != -1:
			i = line.find(': ')
			line = line[i+2::];
			NAMES += [line];
	
	fileID.close()
	model = NAMES[0]
	
	return model

#==============================================================================#
# Get SGTE command based on existing or optimized hyperparameters
def define_SGTE_model(fit_type,run_type):

	fitting_types = [0,1,2,3,4,5]
	fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE']
	out_file = '%s.sgt' %fitting_names[fit_type]
	if os.path.exists(out_file):
		os.remove(out_file)
    	
	budget = 200
	
	if run_type == 1: # optimize fitting parameters
		if fit_type == fitting_types[0]:
				model = ("TYPE KRIGING RIDGE OPTIM DISTANCE_TYPE OPTIM METRIC "
						 "RMSECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[1]:
				model = ("TYPE LOWESS DEGREE OPTIM RIDGE OPTIM "
						 "KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM METRIC RMSECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[2]:
				model = ("TYPE KS KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM METRIC RMSECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[3]:
				model = ("TYPE RBF KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM RIDGE OPTIM METRIC RMSECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[4]:
				model = ("TYPE PRS DEGREE OPTIM RIDGE OPTIM "
						 "METRIC RMSECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[5]:
				model = ("TYPE ENSEMBLE WEIGHT OPTIM METRIC RMSECV "
						 "DISTANCE_TYPE OPTIM BUDGET %i OUTPUT %s" %(budget, out_file))
	elif run_type == 2: # Run existing SGTE model
		model = get_SGTE_model(out_file)
		
	return model,out_file

#==============================================================================#
# Define number of 2D projections needed based on n_dims
def hyperplane_SGTE_vis_norm(server,training_X,bounds,variable_lbls,nominal,training_Y,nn,fig,plt,threshold=None,
							 output_label="$\hat{f}(\mathbf{x})$",constraint_label="$\hat{g}(\mathbf{x})$",opts=None,
							 cmax = None, cmin = None):

	
	lob = bounds[:,0]
	upb = bounds[:,1]
	
	lob_n = np.zeros(np.size(lob)); upb_n = np.ones(np.size(upb))
	bounds_n = np.zeros(np.shape(bounds))
	bounds_n[:,0] = lob_n; bounds_n[:,1] = upb_n
	
	#======================== 2D GRID CONSTRUCTION ============================#
	# %% Activate 2 out 4 variables up to 4 variables
	d = len(training_X[0,:]); #<-------- Number of variables
	if d > 4: # maximum of four variables allowed
		d = 4
	    
	if d == 1:
		sp_shape = [1,1]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45    #<-------- Edit as necessary to fit figure properly
		fig.set_figheight(3); fig.set_figwidth(4.67)
	elif d == 2:
		sp_shape = [1,1]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45    #<-------- Edit as necessary to fit figure properly
		fig.set_figheight(3); fig.set_figwidth(4.67)
	elif d == 3:
		sp_shape = [1,3]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45    #<-------- Edit as necessary to fit figure properly
		fig.set_figheight(3.1); fig.set_figwidth(15)
	elif d == 4:
		sp_shape = [2,3]; ax_h = 0.1; ax_bot = 0.08; ax_left = 0.0; ann_pos = 0.45   #<-------- Edit as necessary to fit figure properly
		fig.set_figheight(5.8); fig.set_figwidth(15)
	
	gs = gridspec.GridSpec(sp_shape[0],sp_shape[1], 
						width_ratios = np.ones(sp_shape[1],dtype=int), 
						height_ratios = np.ones(sp_shape[0],dtype=int),
						left=0.15, right=0.85, wspace=0.2)
	
	
	q = combinations(range(d),2) # choose 2 out d variables
	ss = comb(d,2,exact=True)

	if d != 1:
		plot_countour_code(q,bounds,bounds_n,lob,upb,d,nominal,nn,
						   training_X,server,variable_lbls,gs,plt,fig,
						   threshold=threshold,output_label=output_label,constraint_label=constraint_label,
						   opts=opts, cmax = cmax, cmin = cmin)
	#===========================================================================
	# copyfile(sgt_file, os.path.join(os.getcwd(),'SGTE_matlab_server',sgt_file)) # backup hyperparameters
	#===========================================================================

#==============================================================================#
# Plot design space using 2D projections
def plot_countour_code(q,bounds,bounds_n,lob,upb,d,nominal,nn,training_X,server,variable_lbls,gs,plt,fig,
					   output_label="$\hat{f}(\mathbf{x})$",constraint_label="$\hat{g}(\mathbf{x})$",
					   threshold=None,n_rand=0,opts=None, cmax = None, cmin = None):

	iteraton = -1
	for par in q:
		iteraton += 1
		# Plot points
		i = par; # plot variable indices
		bounds_p = np.zeros(bounds.shape)
		bounds_p_n = np.zeros(bounds_n.shape)
		nn_vec = nn*np.ones(len(training_X[0,:]),dtype=int)
		for n in range(len(bounds)):
		    if n not in i:
		        lm = nominal[n]
		        fixed_value = scaling(lm,lob[n],upb[n],2) # Assign lambdas
		        
		        bounds_p[n,0] = fixed_value-0.0000001 # Set bounds equal to each other
		        bounds_p[n,1] = fixed_value+0.0000001 # Set bounds equal to each other
		        bounds_p_n[n,0] = lm # Nomalized bounds
		        bounds_p_n[n,1] = lm+0.01 # Nomalized bounds
		        nn_vec[n] = 1

		    else:
		        bounds_p[n,0] = bounds[n,0]
		        bounds_p[n,1] = bounds[n,1]
		        bounds_p_n[n,0] = bounds_n[n,0]
		        bounds_p_n[n,1] = bounds_n[n,1]
		
		X = gridsamp(bounds_p_n.T, nn_vec)
		# Prediction
		# YX = sm.predict_values(X)
		[YX, std, ei, cdf] = server.sgtelib_server_predict(X)
		
		#========================= DATA VISUALIZATION =============================#
		# %% Sensitivity plots
		YX_obj = YX[:,0]
		X = X[:,i]
		X1_norm = np.reshape(X[:,0],(nn,nn)); X2_norm = np.reshape(X[:,1],(nn,nn))
		X1 = scaling(X1_norm, lob[i[0]], upb[i[0]], 2) # Scale up plot variable
		X2 = scaling(X2_norm, lob[i[1]], upb[i[1]], 2) # Scale up plot variable
		YX_obj = np.reshape(YX_obj, np.shape(X1))
		
		ax = fig.add_subplot(gs[iteraton]) # subplot
		cf = ax.contourf( X1, X2, YX_obj, cmap=plt.cm.jet) # plot contour
		# cf = ax.contourf( X1, X2, YX_obj, vmin = cmin, vmax = cmax, cmap=plt.cm.jet); # plot contour
		ax.contour(cf, colors='k')
		
		cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
		cbar.set_array(YX_obj)

		if cmax is not None and cmin is not None:
			cmax = cmax; cmin = cmin # set colorbar limits
			boundaries = np.linspace(cmin, cmax, 51)
			cbar_h = fig.colorbar(cbar, boundaries=boundaries)
			cbar_h.set_label(output_label, rotation=90, labelpad=3)
		else:
			cbar_h = fig.colorbar(cbar)
			cbar_h.set_label(output_label, rotation=90, labelpad=3)

		artists, labels = cf.legend_elements()
		af = artists[0]
		
		handles = []; labels = []

		#======================== NONLINEAR CONSTRAINTS ============================#	
		# %% Nonlinear constraints
		if threshold:
			YX_cstr = YX[:,1] - threshold
			YX_cstr = np.reshape(YX_cstr, np.shape(X1))
			c1 = ax.contourf( X1, X2, YX_cstr, alpha=0.0, levels=[-20, 0, 20], colors=['#FF0000','#FF0000'], 
							hatches=['//', None])
			ax.contour(c1, colors='#FF0000', linewidths = 2.0, zorder=1)
			a1 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#FF0000', facecolor='none', fill='None', hatch='///')

			handles += [a1]
			labels += ["%s" %(constraint_label)] # for presentation
			
		#========================= OPTIMIZATION POINTS ============================#
		import matplotlib.lines as mlines

		if opts is not None:
			ax.plot(opts[:,i[0]],opts[:,i[1]],'*m', markersize = 10) # plot STOMADS points for surrogate
			a_opts = mlines.Line2D([], [], color='magenta', marker='*', markersize=7, linestyle='')

			handles += [a_opts]
			labels += ["StoMADS result"] # for presentation	

		#========================= MONTE CARLO POINTS =============================#

		ax.axis([lob[i[0]],upb[i[0]],lob[i[1]],upb[i[1]]]) # fix the axis limits

		ax.plot(training_X[:50,i[0]],training_X[:50,i[1]],'.k', markersize = 3) # plot DOE points for surrogate (first 50 only)
		a_MCI = mlines.Line2D([], [], color='black', marker='.', markersize=5, linestyle='')
		
		handles += [a_MCI]
		labels += ["Monte Carlo samples"] # for presentation	
		#============================ AXIS LABELS =================================#	

		ax.set_xlabel(variable_lbls[i[0]], labelpad=-1)
		ax.set_ylabel(variable_lbls[i[1]], labelpad=-1)

	lx = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize = 14)
	# Get the bounding box of the original legend
	bb = lx.get_bbox_to_anchor().transformed(ax.transAxes.inverted())

	# Change to location of the legend. 
	xOffset = 0.0; yOffset = 0.05
	bb.x0 += xOffset; bb.x1 += xOffset # Move anchor point x0 and x1 to gether
	bb.y0 += yOffset; bb.y1 += yOffset # Move anchor point y0 and y1 to gether
	lx.set_bbox_to_anchor(bb, transform = ax.transAxes)

#==============================================================================#
# Visualize design space (callable)
def visualize_surrogate(bounds,variable_lbls,server,training_X,training_Y,plt,current_path=os.getcwd(),
                        vis_output=0,threshold=None,resolution=20,
                        output_lbls=None,base_name="surrogate_model",
                        opts=None,cmax=None,cmin=None):

	mpl.rc('text', usetex = True)
	mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
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

	fig_name = '%s.png' %(base_name)
	fig_file_name = os.path.join(current_path,fig_name)
	fig.savefig(fig_file_name, bbox_inches='tight')

#==============================================================================#
# POSTPROCESS DOE DATA
def train_server(training_X, training_Y, bounds):
    
    #======================== SURROGATE META MODEL ============================#
    # %% SURROGATE modeling
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    Y = training_Y; S_n = scaling(training_X, lob, upb, 1)
    # fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE']
    # run_types = ['optimize hyperparameters','load hyperparameters'] (1 or 2)
    fit_type = 5; run_type = 1 # optimize all hyperparameters
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

#  MAIN FILE
def main():
	import os
	import numpy as np

	current_path = os.getcwd() # Working directory of file
	wokring_directory = os.path.join(current_path,'Job_results','BACKUP','Vf','Results_log') # print variables as space demilited string
	os.chdir(wokring_directory)

	# debug subroutines here ....
    
# Stuff to run when not called in import
if __name__ == "__main__":
	main()