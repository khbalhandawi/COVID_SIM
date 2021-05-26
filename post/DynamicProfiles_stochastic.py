import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

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

    # load optimization points in LHS format file
    with open('data_dynamic/points_opts.pkl','rb') as fid:
            lob_var = pickle.load(fid)
            upb_var = pickle.load(fid)
            points = pickle.load(fid)
            points_unscaled = pickle.load(fid)

    # other_points = np.array([[6.87323850e+01,4.43845073e-02,4.75511210e+01],
    #                         [7.07245300e+01,5.36435305e-02,4.68560890e+01],
    #                         [8.96503750e+01,6.30122805e-02,5.02193600e+01]])

    # points_unscaled = np.vstack((points_unscaled,other_points))

    # print(points_unscaled)

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
    
        # if run < 4:
        #     legend_label = "Run %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))
        # else:
        #     if run == 4:
        #         run = 0
        #     legend_label = "Solution %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,round(point[0]),point[1],round(point[2]))
       
        labels += [legend_label]

    fig_1, _, ax1_1 = build_fig_SIR()
    draw_SIR_compare(data_I, lb_data_I, ub_data_I, fig_1, ax1_1, labels=labels, palette = palette, 
        save_name = 'I_compare', xlim = 350, ylim = None, leg_location = 'center right', plot_path="data_dynamic",
        y_label = 'Number of infections $n_I^k$', threshold=90, threshold_label="Healthcare capacity $H_{\mathrm{max}}$")
    
    fig_2, _, ax1_2 = build_fig_SIR()
    draw_SIR_compare(data_F, lb_data_F, ub_data_F, fig_2, ax1_2, labels=labels, palette = palette, 
        save_name = 'F_compare', xlim = 350, leg_location = 'upper left', plot_path="data_dynamic",
        y_label = 'Number of fatalities $n_F^k$')

    fig_3, _, ax1_3 = build_fig_SIR()
    draw_SIR_compare(data_R, lb_data_R, ub_data_R, fig_3, ax1_3, labels=labels, palette = palette, 
        save_name = 'R_compare', xlim = 350, leg_location = 'upper left', plot_path="data_dynamic",
        y_label = 'Number of recoveries $n_R^k$')

    fig_4, _, ax1_4 = build_fig_SIR()
    draw_SIR_compare(data_M, lb_data_M, ub_data_M, fig_4, ax1_4, labels=labels, palette = palette, 
        save_name = 'M_compare', xlim = 350, ylim = 2.5, leg_location = 'lower right', plot_path="data_dynamic",
        y_label = 'Mobility $M^k$')

    fig_5, _, ax1_5 = build_fig_SIR()
    draw_R0_compare(data_R0, R0_time_data, lb_data_R0, ub_data_R0, fig_5, ax1_5, labels=labels, palette = palette, 
        save_name = 'R0_compare', xlim = 350, leg_location = 'upper right', plot_path="data_dynamic",
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