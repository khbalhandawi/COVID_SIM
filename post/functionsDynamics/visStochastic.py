import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from functionsUtilities.utils import check_folder
import numpy as np

def build_fig_SIR(figsize=(6,5)):

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(ncols=1, nrows=1)
    ax1 = fig.add_subplot(spec[0,0])

    ax1.set_xlabel('Time (days)', fontsize = 14)
    ax1.set_ylabel('Population size', fontsize = 14)

    #get color palettes
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444']
    
    return fig, spec, ax1

def build_fig_time_series(label, figsize=(5,4)):

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(ncols=1, nrows=1)

    ax1 = fig.add_subplot(spec[0,0])

    ax1.set_xlabel('Time (days)', fontsize = 14)
    ax1.set_ylabel(label, fontsize = 14)
    
    # handles, labels = [[a1,a2,a3,a4,a5], ['healthcare capacity','infectious','susceptible','recovered','fatalities']]
    # fig.legend(handles, labels, loc='upper center', ncol=5, fontsize = 10)

    #if 

    return fig, ax1

def draw_SIR(fig, ax1, data=None, 
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    healthcare_capacity = 300, plot_path = 'render/'):

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

def draw_SIR_compare(data, time, lb_data, ub_data, fig, ax1, labels=None,
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    plot_path = 'render/', save_name = 'X_compare',
    xlim = None, ylim = None, leg_location = 'center right',
    y_label = 'Population size', threshold=None, threshold_label=None,styles=None):

    #get color palettes
    palette = palette
    if styles is None:
        styles = ["-"]*len(data)

    if xlim is not None:
        ax1.set_xlim(0, xlim)
    if ylim is not None:
        ax1.set_ylim(0, ylim)

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()

    n = 1; n_pts = len(data)
    legned_labels = []; legend_handles = []
    for x_data,datum,lb,ub,label,color,style in zip(time,data,lb_data,ub_data,labels,palette,styles):
        
        if n_pts > 1:
            transparency = (-0.15 / (n_pts - 1)) * n + (0.15/ (n_pts - 1)) + 0.25
        else:
            transparency = 0.25

        a1 = ax1.plot(x_data, datum, color=color, label=label, linewidth = 1.0,linestyle=style)
        # ax1.plot(x_data, lb, color=color, linewidth = 0.25, linestyle=(0, (20, 10, 20, 10)))
        # ax1.plot(x_data, ub, color=color, linewidth = 0.25, linestyle=(0, (20, 10, 20, 10)))
        ax1.plot(x_data, lb, color=color, linewidth = 0.1, linestyle=(0, (20, 0, 20, 0)))
        ax1.plot(x_data, ub, color=color, linewidth = 0.1, linestyle=(0, (20, 0, 20, 0)))
        ax1.fill_between(x_data, lb, ub, color=color, alpha=transparency)

        a2 = ax1.fill(np.NaN, np.NaN, color=color, alpha=transparency) # dummy actor

        print("%s: max = %f, cumilative = %f" %(save_name,max(datum),datum[-1]))
        n +=1 

        legend_handles += [(a1[0],a2[0])]
        legned_labels += [label]
    
    if (threshold is not None) and (threshold_label is not None):
        at = ax1.plot(time[0], [threshold for x in range(len(data[0]))], 
                 'k:', label=threshold_label,linestyle=(0, (5, 2, 5, 2)), linewidth = 0.7)

        legend_handles += [at[0]]
        legned_labels += [threshold_label]

    ax1.legend(legend_handles, legned_labels, loc=leg_location, ncol=1, fontsize = 8)
    ax1.set_ylabel(y_label, fontsize = 14)

    bg_color = 'w'

    check_folder(plot_path)
    fig.savefig('%s/%s.pdf' %(plot_path,save_name), dpi=100, facecolor=bg_color, bbox_inches='tight')

def draw_R0_compare(data, data_x, lb_data, ub_data, fig, ax1, labels=None,
    palette = ['#1C758A', '#CF5044', '#BBBBBB', '#444444'],
    plot_path = 'render/', save_name = 'X_compare',
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

        x_data = data_x # time vector for plot
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

        x_data = data_x # time vector for plot
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
    fig.savefig('%s/%s.pdf' %(plot_path,save_name), dpi=1000, facecolor=bg_color, bbox_inches='tight')