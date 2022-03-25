import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys

from functionsDynamics.visStochastic import build_fig_SIR,draw_SIR_compare
from functionsDynamics.statsStochastic import process_statistics, process_realizations


#=============================================================================
# Main execution 
if __name__ == '__main__':

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams['font.family'] = 'serif'

    pop_size = 1000
    healthcare_capacity = 90

    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palette_COVID_SIM_UI = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # load optimization points in terms of ABM solution space
    with open('data_dynamic/points_opts.pkl','rb') as fid:
            lob_var = pickle.load(fid)
            upb_var = pickle.load(fid)
            points = pickle.load(fid)
            points_unscaled = pickle.load(fid)

    run = 0;  time = []

    data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []

    # for plotting gerad demo
    select_indices = [0,]
    y_label_add = ''
    prefixes = ['',] * len(select_indices)
    y_scaling = 1
    threshold = 90
    ylims = [350,None,None,None,]
    xlims = [350,]*4
    runs = select_indices

    points_unscaled = [points_unscaled[index] for index in select_indices]


    ##########################   COVID_SIM_UI ##############################

    for realization in range(5):
        hatches = None
        data = process_realizations(run,x_scaling=1/10,y_scaling=y_scaling,n_realization=realization)

        [process_I, process_F, process_R, process_M, process_R0, R0_time_data, time_data] = data

        data_I += [process_I]
        data_F += [process_F]
        data_R += [process_R]
        data_M += [process_M]
        data_R0 += [process_R0]

        color = [palette[0],]*len(data_I)

        time += [time_data]

        ######################################################################

        fig_1, _, ax1_1 = build_fig_SIR()
        draw_SIR_compare(data_I, time, fig_1, ax1_1, palette = color, 
            save_name = 'I_compare_R_%i' %realization, xlim = xlims[0], ylim = ylims[0], leg_location = 'upper right', plot_path="data_dynamic",
            y_label = 'Infections $n_I^t$', threshold=threshold, threshold_label="Healthcare capacity $H_{\mathrm{max}}$",linewidth=0.2)

        fig_4, _, ax1_4 = build_fig_SIR()
        draw_SIR_compare(data_M, time, fig_4, ax1_4, palette = color, 
            save_name = 'M_compare_%i' %realization, xlim = xlims[3], ylim = ylims[3], leg_location = 'lower right', plot_path="data_dynamic",
            y_label = 'Mobility $M^t$',linewidth=0.2)

        # plt.show()


    ######################################################################
    # plot final distribution

    run = 0; labels = []; time = []; time_COVID_SIM_UI = []; labels_COVID_SIM_UI = []; styles = []

    data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []; 
    lb_data_I = []; lb_data_F = []; lb_data_R = []; lb_data_M = []; lb_data_R0 = []
    ub_data_I = []; ub_data_F = []; ub_data_R = []; ub_data_M = []; ub_data_R0 = []

    data = process_statistics(run=0,x_scaling=1/10,y_scaling=y_scaling,conf_interval=85,use_percentiles=True)
    # data = process_statistics(run,x_scaling=1/10,y_scaling=y_scaling,conf_interval=1,use_percentiles=False)

    [mean_process_I, mean_process_F, mean_process_R, mean_process_M, mean_process_R0, 
        ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
        ub_process_M, lb_process_M, ub_process_R0, lb_process_R0, R0_time_data, time_data] = data

    data_I += [mean_process_I]; lb_data_I += [lb_process_I]; ub_data_I += [ub_process_I]
    data_F += [mean_process_F]; lb_data_F += [lb_process_F]; ub_data_F += [ub_process_F]
    data_R += [mean_process_R]; lb_data_R += [lb_process_R]; ub_data_R += [ub_process_R]
    data_M += [mean_process_M]; lb_data_M += [lb_process_M]; ub_data_M += [ub_process_M]
    data_R0 += [mean_process_R0]; lb_data_R0 += [lb_process_R0]; ub_data_R0 += [ub_process_R0]

    time += [time_data]
    time_COVID_SIM_UI += [time_data]

    # Legend labels
    # prefix = "Solution %i" %(run+1) # uncomment for generic legend labels
    # prefix = "Scenario %i" %(run+1) # uncomment for generic legend labels

    legend_label = "inter-quartile range" # generic legend labels

    labels = [legend_label,]

    fig_1, _, ax1_1 = build_fig_SIR()
    draw_SIR_compare(data_I, time, fig_1, ax1_1, lb_data_I, ub_data_I, labels=labels, palette = palette, 
        save_name = 'I_compare_opt_%i' %run, xlim = xlims[0], ylim = ylims[0], leg_location = 'upper right', plot_path="data_dynamic",
        y_label = 'Infections $n_I^t$', threshold=threshold, threshold_label="Healthcare capacity $H_{\mathrm{max}}$")

    fig_4, _, ax1_4 = build_fig_SIR()
    draw_SIR_compare(data_M, time_COVID_SIM_UI, fig_4, ax1_4, lb_data_M, ub_data_M, labels=labels_COVID_SIM_UI, palette = palette_COVID_SIM_UI, 
        save_name = 'M_compare_opt_%i' %run, xlim = xlims[3], ylim = ylims[3], leg_location = 'lower right', plot_path="data_dynamic",
        y_label = 'Mobility $M^t$')