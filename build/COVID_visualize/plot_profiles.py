import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from population import load_matrix
from utils import check_folder
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
    fig.savefig('%s/SIRF_plot.pdf' %(plot_path), dpi=1000, facecolor=bg_color, bbox_inches='tight')

def draw_SIR_compare(data, fig, ax1, leg=None, labels=None,
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    pop_size = 1000, plot_path = 'render/', save_name = 'X_compare',
    xlim = None, ylim = None, leg_location = 'center right',
    y_label = 'Population size'):

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

    for datum,label,color in zip(data,labels,palette):
        ax1.plot(x_data, datum, color=color, label=label, linewidth = 1)
        print("%s: max = %f, cumilative = %f" %(save_name,max(datum),datum[-1]))
    
    ax1.legend(loc=leg_location, ncol=1, fontsize = 8)
    ax1.set_ylabel(y_label, fontsize = 14)

    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/%s.pdf' %(plot_path,save_name), dpi=1000, facecolor=bg_color, bbox_inches='tight')

def draw_R0_compare(data, fig, ax1, leg=None, labels=None,
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

    for datum,label,color in zip(data,labels,palette):
        x_data = [x/10 for x in datum[:,0]] # time vector for plot
        y_data = datum[:,1] # R0 vector for plot
        ax1.plot(x_data, y_data, color=color, label=label, linewidth = 1)
        epidemic = True; prev_R0 = 10.0
        for x,y in zip(x_data,y_data):
            if y <= 1.0 and prev_R0 > 1:
                x_endemic = x
            prev_R0 = y
        print("%s: t(R_0 == 1) = %f" %(save_name,x_endemic))
    
    if (line_label is not None) and (threshold is not None) and (threshold_label is not None):
        ax1.plot(x_data, [threshold for x in range(len(datum[:,0]))], 
                 'k:', label=threshold_label)

    axins = zoomed_inset_axes(ax1, 2.5, loc='center right')
    for datum,label,color in zip(data,labels,palette):
        x_data = [x/10 for x in datum[:,0]] # time vector for plot
        y_data = datum[:,1] # R0 vector for plot
        axins.plot(x_data, y_data, color=color, label=label, linewidth = 1)

    if (line_label is not None) and (threshold is not None) and (threshold_label is not None):
        axins.plot(x_data, [threshold for x in range(len(datum[:,0]))], 
                 'k:', label=threshold_label)

    axins.set_xlim(85, 200)
    axins.set_ylim(0.1, 1.5)
    # axins.xticks(visible=False)
    # axins.yticks(visible=False)
    axins.tick_params(axis='x', tick1On=False, tick2On=False, label1On=False, label2On=False)
    axins.tick_params(axis='y', tick1On=False, tick2On=False, label1On=False, label2On=False)
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax1.legend(loc=leg_location, ncol=1, fontsize = 8)
    ax1.set_ylabel(y_label, fontsize = 14)

    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/%s.pdf' %(plot_path,save_name), dpi=1000, facecolor=bg_color, bbox_inches='tight')

#=============================================================================
# Main execution 
if __name__ == '__main__':

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'

    pop_size = 1000
    healthcare_capacity = 90

    # data = load_matrix('SIRF_data', folder='population')
    # fig_sir, spec_sir, ax1_sir = build_fig_SIR(pop_size = pop_size)
    # draw_SIR(fig_sir, ax1_sir, data=data, pop_size = pop_size, healthcare_capacity = healthcare_capacity)

    # data = load_matrix('dist_data', folder='population')
    # fig_D, ax1_D = build_fig_time_series(label = "Mean distance $D$")
    # draw_time_series(data[:,0], data[:,1], "D", fig_D, ax1_D)

    # data = load_matrix('mean_GC_data', folder='population')
    # fig_GC, ax1_GC = build_fig_time_series(label = "Mobility ($\%$)")
    # draw_time_series(data[:,0], data[:,1]*100, "GC", fig_GC, ax1_GC )

    # data = load_matrix('mean_R0_data', folder='population')
    # fig_R0, ax1_R0 = build_fig_time_series(label = "Basic reproductive number $R_0$")
    # draw_time_series(data[:,0], data[:,1], "R0", fig_R0, ax1_R0, line_label = "$R_0$", threshold = 1, threshold_label='$R_0=1$' )

    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    points = np.array( [[30.0, 0.100, 10],
                        [60.0, 0.100, 10],
                        [30.0, 0.125, 10],
                        [30.0, 0.100, 20]])

    run = 0; data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []; labels = []
    for point in points:

        run_data = load_matrix('SIRF_data', folder='population/run_%i' %(run+1))
        run_data_M = load_matrix('mean_GC_data', folder='population/run_%i' %(run+1))
        run_data_R0 = load_matrix('mean_R0_data', folder='population/run_%i' %(run+1))
        I = run_data[:,2]; R = run_data[:,3]; F = run_data[:,4]; M = [x * 100 for x in run_data_M[:,1]]

        data_I += [I]; data_F += [F]; data_R += [R]; data_M += [M]; data_R0 += [run_data_R0]

        # Legend labels
        legend_label = "Run %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,point[0],point[1],point[2])
        labels += [legend_label]

        run += 1

    fig_1, _, ax1_1 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_I, fig_1, ax1_1, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'I_compare', xlim = 350, leg_location = 'upper right', 
        y_label = 'Number of infections $n_I^k$')
    
    fig_2, _, ax1_2 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_F, fig_2, ax1_2, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'F_compare', xlim = 350, 
        y_label = 'Number of fatalities $n_F^k$')

    fig_3, _, ax1_3 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_R, fig_3, ax1_3, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'R_compare', xlim = 350, 
        y_label = 'Number of recoveries $n_R^k$')

    fig_4, _, ax1_4 = build_fig_SIR(pop_size = pop_size)
    draw_SIR_compare(data_M, fig_4, ax1_4, labels=labels, palette = palette, 
        pop_size = pop_size, save_name = 'M_compare', xlim = 350, leg_location = 'lower right', 
        y_label = 'Socio-economic impact $M^k$')

    fig_5, _, ax1_5 = build_fig_SIR(pop_size = pop_size)
    draw_R0_compare(data_R0, fig_5, ax1_5, labels=labels, palette = palette, 
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