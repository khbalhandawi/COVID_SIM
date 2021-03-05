import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from population import load_matrix
from utils import check_folder
import pickle
import numpy as np

def build_fig_SIR(figsize=(5,4), pop_size = 1000):

    fig = plt.figure(figsize=(5,4))
    spec = fig.add_gridspec(ncols=1, nrows=1)
    ax1 = fig.add_subplot(spec[0,0])

    ax1.set_xlabel('Time (days)', fontsize = 14)
    ax1.set_ylabel('Population size', fontsize = 14)

    #get color palettes
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444']
    
    return fig, spec, ax1

def build_fig_time_series(label, figsize=(5,4)):

    fig = plt.figure(figsize=(5,4))
    spec = fig.add_gridspec(ncols=1, nrows=1)

    ax1 = fig.add_subplot(spec[0,0])

    ax1.set_xlabel('Time (days)', fontsize = 14)
    ax1.set_ylabel(label, fontsize = 14)
    
    # handles, labels = [[a1,a2,a3,a4,a5], ['healthcare capacity','infectious','susceptible','recovered','fatalities']]
    # fig.legend(handles, labels, loc='upper center', ncol=5, fontsize = 10)

    #if 

    return fig, ax1

def draw_SIR(fig, ax1, leg=None, data=None, 
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    pop_size = 1000, healthcare_capacity = 300, plot_path = 'render/'):

    #get color palettes
    palette = palette

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()

    S = data[:,1]
    I = data[:,2]
    R = data[:,3]
    F = data[:,4]

    x_data = np.arange(len(I)) / 10 # time vector for plot
    infected_arr = np.asarray(I)
    indices = np.argwhere(infected_arr >= healthcare_capacity)

    ax1.plot(x_data, S, color=palette[0], label='susceptible')
    ax1.plot(x_data, I, color=palette[1], label='infectious')
    ax1.plot(x_data, R, color=palette[2], label='recovered')
    ax1.plot(x_data, F, color=palette[3], label='fatalities')

    ax1.plot(x_data, [healthcare_capacity for x in range(len(I))], 
                color=palette[1], linestyle='--', label='healthcare capacity')

    
    ax1.legend(loc='upper right', ncol=1, fontsize = 10)

    plt.draw()
        
    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/SIRF_plot.png' %(plot_path), dpi=1000, facecolor=bg_color, bbox_inches='tight')

def draw_SIR_compare(data, lb_data, ub_data, fig, ax1, leg=None, labels=None,
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    pop_size = 1000, plot_path = 'render/', save_name = 'X_compare',
    xlim = None, ylim = None, leg_location = 'center right',
    y_label = 'Population size', threshold=None, threshold_label=None):

    #get color palettes
    palette = palette

    if xlim is not None:
        ax1.set_xlim(0, xlim)
    if ylim is not None:
        ax1.set_ylim(0, ylim)

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()

    x_data = np.arange(len(data[0])) / 10 # time vector for plot

    n = 1; n_pts = len(data)
    for datum,lb,ub,label,color in zip(data,lb_data,ub_data,labels,palette):
        
        if n_pts > 1:
            transparency = (-0.15 / (n_pts - 1)) * n + (0.15/ (n_pts - 1)) + 0.25
        else:
            transparency = 0.25

        ax1.plot(x_data, datum, color=color, label=label, linewidth = 1)
        ax1.fill_between(x_data, lb, ub, color=color, alpha=transparency)
        print("%s: max = %f, cumilative = %f" %(save_name,max(datum),datum[-1]))
        n +=1 
    
    if (threshold is not None) and (threshold_label is not None):
        ax1.plot(x_data, [threshold for x in range(len(datum))], 
                 'k:', label=threshold_label)

    ax1.legend(loc=leg_location, ncol=1, fontsize = 8)
    ax1.set_ylabel(y_label, fontsize = 14)

    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/%s.png' %(plot_path,save_name), dpi=100, facecolor=bg_color, bbox_inches='tight')

def draw_R0_compare(data, data_x, lb_data, ub_data, fig, ax1, leg=None, labels=None,
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    pop_size = 1000, plot_path = 'render/', save_name = 'X_compare',
    xlim = None, ylim = None, leg_location = 'center right',
    y_label = 'Population size', line_label = None, threshold = None, 
    threshold_label= None):

    #get color palettes
    palette = palette

    if xlim is not None:
        ax1.set_xlim(0, xlim)
    if ylim is not None:
        ax1.set_ylim(0, ylim)

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()

    n = 1; n_pts = len(data)
    for datum,lb,ub,label,color in zip(data,lb_data,ub_data,labels,palette):

        if n_pts > 1:
            transparency = (-0.15 / (n_pts - 1)) * n + (0.15/ (n_pts - 1)) + 0.25
        else:
            transparency = 0.25

        x_data = [x/10 for x in data_x] # time vector for plot
        y_data = datum # R0 vector for plot
        ax1.plot(x_data, y_data, color=color, label=label, linewidth = 1)
        ax1.fill_between(x_data, lb, ub, color=color, alpha=transparency)
        epidemic = True; prev_R0 = 10.0
        for x,y in zip(x_data,y_data):
            if y <= 1.0 and prev_R0 > 1:
                x_endemic = x
            prev_R0 = y
        print("%s: t(R_0 == 1) = %f" %(save_name,x_endemic))
        n +=1 
    
    if (line_label is not None) and (threshold is not None) and (threshold_label is not None):
        ax1.plot(x_data, [threshold for x in range(len(datum))], 
                 'k:', label=threshold_label)

    n = 1; n_pts = len(data)
    axins = zoomed_inset_axes(ax1, 3, loc='center right')
    for datum,lb,ub,label,color in zip(data,lb_data,ub_data,labels,palette):
        
        if n_pts > 1:
            transparency = (-0.15 / (n_pts - 1)) * n + (0.15/ (n_pts - 1)) + 0.25
        else:
            transparency = 0.25

        x_data = [x/10 for x in data_x] # time vector for plot
        y_data = datum # R0 vector for plot
        axins.plot(x_data, y_data, color=color, label=label, linewidth = 1)
        axins.fill_between(x_data, lb, ub, color=color, alpha=transparency)
        n +=1 

    if (line_label is not None) and (threshold is not None) and (threshold_label is not None):
        axins.plot(x_data, [threshold for x in range(len(datum))], 
                 'k:', label=threshold_label)

    axins.set_xlim(75, 155)
    # axins.set_xlim(75 - 50, 155 - 50)
    axins.set_ylim(0.5, 1.5)
    # axins.xticks(visible=False)
    # axins.yticks(visible=False)
    # axins.tick_params(axis='x', tick1On=False, tick2On=False, label1On=False, label2On=False)
    axins.tick_params(axis='y', tick1On=False, tick2On=False, label1On=False, label2On=False)
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax1.legend(loc=leg_location, ncol=1, fontsize = 8)
    ax1.set_ylabel(y_label, fontsize = 14)

    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/%s.png' %(plot_path,save_name), dpi=1000, facecolor=bg_color, bbox_inches='tight')

def process_statistics(run):

    with open('population/MCS_process_data_r%i.pkl' %run,'rb') as fid:
        process_I = pickle.load(fid)
        process_F = pickle.load(fid)
        process_R = pickle.load(fid)
        process_M = pickle.load(fid)
        process_R0 = pickle.load(fid)

    n_increments = len(process_I[0])
    n_increments_R0 = len(process_R0[0])
    n_samples = len(process_I)

    sum_I = np.zeros(n_increments)
    sum_R = np.zeros(n_increments)
    sum_F = np.zeros(n_increments)
    sum_M = np.zeros(n_increments)
    sum_R0 = np.zeros(n_increments_R0)

    #================================================
    # Get distributions first (for SIRF and Mobility)

    ub_process_I = []
    lb_process_I = []
    ub_process_R = []
    lb_process_R = []
    ub_process_F = []
    lb_process_F = []
    ub_process_M = []
    lb_process_M = []
    
    for k in range(n_increments):
        
        dist_I = []; dist_R = []; dist_F = []; dist_M = []; dist_R0 = [];
        for I,R,F,M,R0 in zip(process_I,process_R,process_F,process_M,process_R0):

            dist_I += [I[k]]
            dist_R += [R[k]]
            dist_F += [F[k]]
            dist_M += [M[k]*(2000/3500)]

        ub_I = np.percentile(dist_I, 90); ub_process_I += [ub_I]
        lb_I = np.percentile(dist_I, 10); lb_process_I += [lb_I]
        ub_R = np.percentile(dist_R, 90); ub_process_R += [ub_R]
        lb_R = np.percentile(dist_R, 10); lb_process_R += [lb_R]
        ub_F = np.percentile(dist_F, 90); ub_process_F += [ub_F]
        lb_F = np.percentile(dist_F, 10); lb_process_F += [lb_F]
        ub_M = np.percentile(dist_M, 90); ub_process_M += [ub_M]
        lb_M = np.percentile(dist_M, 10); lb_process_M += [lb_M]

    #================================================
    # Get distributions first (for R0)

    ub_process_R0 = []
    lb_process_R0 = []

    for k in range(n_increments_R0):
        dist_R0 = [];
        for R0 in process_R0:
            dist_R0 += [R0[k,1]]
        
        ub_R0 = np.percentile(dist_R0, 90); ub_process_R0 += [ub_R0]
        lb_R0 = np.percentile(dist_R0, 10); lb_process_R0 += [lb_R0]

    #================================================
    # Get means

    for I,R,F,M,R0 in zip(process_I,process_R,process_F,process_M,process_R0):

        sum_I += I; sum_R += R; sum_F += F; sum_M += M; 
        sum_R0 += R0[:,1]

    mean_process_I = sum_I / n_samples
    mean_process_R = sum_R / n_samples
    mean_process_F = sum_F / n_samples
    mean_process_M = sum_M * (2000/3500) / n_samples 
    mean_process_R0 = sum_R0 / n_samples

    R0_time_axis = process_R0[0][:,0]

    #================================================
    # return stats data

    data = [mean_process_I, mean_process_F, mean_process_R, mean_process_M, mean_process_R0, 
            ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
            ub_process_M, lb_process_M, ub_process_R0, lb_process_R0, R0_time_axis]

    return data

#=============================================================================
# Main execution 
if __name__ == '__main__':

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'

    pop_size = 1000
    healthcare_capacity = 90

    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # load optimization points in LHS format file
    with open('population/points_opts.pkl','rb') as fid:
            lob_var = pickle.load(fid)
            upb_var = pickle.load(fid)
            points = pickle.load(fid)
            points_unscaled = pickle.load(fid)

    run = 0; data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []; labels = []
    lb_data_I = []; lb_data_F = []; lb_data_R = []; lb_data_M = []; lb_data_R0 = []
    ub_data_I = []; ub_data_F = []; ub_data_R = []; ub_data_M = []; ub_data_R0 = []

    # terminate
    select_indices = [0,1,2,3]
    # select_indices = [0,5,6]
    # select_indices = [0]
    points_unscaled = [points_unscaled[index] for index in select_indices]
    runs = select_indices
    palette = [palette[index] for index in select_indices]

    for point,run in zip(points_unscaled,runs):

        data = process_statistics(run)

        [mean_process_I, mean_process_F, mean_process_R, mean_process_M, mean_process_R0, 
            ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
            ub_process_M, lb_process_M, ub_process_R0, lb_process_R0, R0_time_data] = data

        data_I += [mean_process_I]; lb_data_I += [lb_process_I]; ub_data_I += [ub_process_I]
        data_F += [mean_process_F]; lb_data_F += [lb_process_F]; ub_data_F += [ub_process_F]
        data_R += [mean_process_R]; lb_data_R += [lb_process_R]; ub_data_R += [ub_process_R]
        data_M += [mean_process_M]; lb_data_M += [lb_process_M]; ub_data_M += [ub_process_M]
        data_R0 += [mean_process_R0]; lb_data_R0 += [lb_process_R0]; ub_data_R0 += [ub_process_R0]

        # Legend labels
        # legend_label = "Run %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))
        legend_label = "Solution %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))
        labels += [legend_label]

    fig_1, _, ax1_1 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_I, lb_data_I, ub_data_I, fig_1, ax1_1, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'I_compare', xlim = 350, leg_location = 'center right', 
        y_label = 'Number of infections $n_I^k$', threshold=90, threshold_label="Healthcare capacity $H_{\mathrm{max}}$")
    
    fig_2, _, ax1_2 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_F, lb_data_F, ub_data_F, fig_2, ax1_2, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'F_compare', xlim = 350, leg_location = 'upper left', 
        y_label = 'Number of fatalities $n_F^k$')

    fig_3, _, ax1_3 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_R, lb_data_R, ub_data_R, fig_3, ax1_3, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'R_compare', xlim = 350, leg_location = 'upper left', 
        y_label = 'Number of recoveries $n_R^k$')

    fig_4, _, ax1_4 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_M, lb_data_M, ub_data_M, fig_4, ax1_4, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'M_compare', xlim = 350, leg_location = 'lower right', 
        y_label = 'Mobility $M^k$')

    fig_5, _, ax1_5 = build_fig_SIR(pop_size = pop_size)
    draw_R0_compare(data_R0, R0_time_data, lb_data_R0, ub_data_R0, fig_5, ax1_5, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'R0_compare', xlim = 350, leg_location = 'upper right', 
        y_label = 'Basic reproductive number $R_0$', line_label = "$R_0$", threshold = 1, threshold_label='$R_0=1$')

    # fig_2AX, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('Time (days)', fontsize = 14)
    # ax1.set_ylabel('Number of infections $n_I^k$', fontsize = 14, color=color)
    
    # x_data = np.arange(len(data_I[0])) / 10 # time vector for plot

    # for datum,label,color in zip(data_I,labels,palette):
    #     ax1.plot(x_data, datum, color=color, label=label, linewidth = 1)

    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('Number of fatalities $n_F^k$', color=color)  # we already handled the x-label with ax1
    
    # for datum,label,color in zip(data_F,labels,palette):
    #     ax1.plot(x_data, datum, color=color, label=label, linewidth = 1)
    
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig_2AX.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()