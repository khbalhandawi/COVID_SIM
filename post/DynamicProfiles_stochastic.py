import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys

from functionsDynamics.visStochastic import build_fig_SIR,draw_SIR_compare,draw_R0_compare
from functionsDynamics.statsStochastic import process_statistics


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

    run = 0; labels = []; time = []; time_COVID_SIM_UI = []; labels_COVID_SIM_UI = []; styles = []

    data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []; 
    lb_data_I = []; lb_data_F = []; lb_data_R = []; lb_data_M = []; lb_data_R0 = []
    ub_data_I = []; ub_data_F = []; ub_data_R = []; ub_data_M = []; ub_data_R0 = []

    data_S = []; data_Critical = []; 
    lb_data_S = []; lb_data_Critical = []
    ub_data_S = []; ub_data_Critical = []

    # for plotting best demo points
    # select_indices = [0,1,2,3]
    # y_label_add = ''
    # prefixes = ['',] * len(select_indices)
    # y_scaling = 1000
    # threshold = 90
    # ylims = [None]*4
    # runs = select_indices

    # for plotting best optimization results
    select_indices = [0,1,2]
    prefixes = ['StoMADS-PB solution', 'GA solution', 'NOMAD solution']
    y_label_add = ''
    y_scaling = 1000
    threshold = 90
    ylims = [120,None,None,2.5]
    xlims = [350,350,350,350]
    runs = select_indices
    xlim = 250
    palette = [y for x,y in sorted(zip(select_indices,palette))] 

    ########################## COVID_SIM_UI ##############################
    styles = [(0, (5, 0, 5, 0)),(0, (5, 5, 5, 5)),(0, (5, 1, 5, 1))]
    z_orders = [2,3,1]
    hatches = ["xxxx","++++","////"]
    hatches = None

    for point,run,prefix in zip(points_unscaled,runs,prefixes):

        data = process_statistics(run,x_scaling=1/10,y_scaling=y_scaling,conf_interval=85,use_percentiles=True)
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

        legend_label = "%s: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(prefix,round(point[0]),point[1],round(point[2])) # generic legend labels

        labels += [legend_label]
        # styles += ["-"]
        labels_COVID_SIM_UI += [legend_label]

        ######################################################################

        fig_1, _, ax1_1 = build_fig_SIR()
        draw_SIR_compare(data_I, time, lb_data_I, ub_data_I, fig_1, ax1_1, labels=labels, palette = palette, 
            save_name = 'I_compare_opt_%i' %run, xlim = xlims[0], ylim = ylims[0], leg_location = 'upper right', plot_path="data_dynamic",
            y_label = 'Infections%s $n_I^t$' %y_label_add, threshold=threshold, threshold_label="Healthcare capacity $H_{\mathrm{max}}$",
            styles=styles,z_orders=z_orders,hatch_patterns=hatches)

        fig_2, _, ax1_2 = build_fig_SIR()
        draw_SIR_compare(data_F, time, lb_data_F, ub_data_F, fig_2, ax1_2, labels=labels, palette = palette, 
            save_name = 'F_compare_opt_%i' %run, xlim = xlims[1], ylim = ylims[1], leg_location = 'upper left', plot_path="data_dynamic",
            y_label = 'Fatalities%s $n_F^t$' %y_label_add, styles=styles,z_orders=z_orders,hatch_patterns=hatches)

        fig_3, _, ax1_3 = build_fig_SIR()
        draw_SIR_compare(data_R, time, lb_data_R, ub_data_R, fig_3, ax1_3, labels=labels, palette = palette, 
            save_name = 'R_compare_opt_%i' %run, xlim = xlims[2], ylim = ylims[2], leg_location = 'upper left', plot_path="data_dynamic",
            y_label = 'Recoveries%s $n_R^t$' %y_label_add, styles=styles,z_orders=z_orders,hatch_patterns=hatches)

        fig_4, _, ax1_4 = build_fig_SIR()
        draw_SIR_compare(data_M, time_COVID_SIM_UI, lb_data_M, ub_data_M, fig_4, ax1_4, labels=labels_COVID_SIM_UI, palette = palette_COVID_SIM_UI, 
            save_name = 'M_compare_opt_%i' %run, xlim = xlims[3], ylim = ylims[3], leg_location = 'lower right', plot_path="data_dynamic",
            y_label = 'Mobility $M^t$', styles=styles,z_orders=z_orders,hatch_patterns=hatches)

        fig_5, _, ax1_5 = build_fig_SIR()
        draw_R0_compare(data_R0, R0_time_data, lb_data_R0, ub_data_R0, fig_5, ax1_5, labels=labels_COVID_SIM_UI, palette = palette_COVID_SIM_UI, 
            save_name = 'R0_compare_opt_%i' %run, xlim = 350, leg_location = 'upper right', plot_path="data_dynamic",
            y_label = 'Basic reproductive number $R_0$', line_label = "$R_0$", threshold = 1, threshold_label='$R_0=1$',
            styles=styles,z_orders=z_orders,hatch_patterns=hatches)

        plt.show()