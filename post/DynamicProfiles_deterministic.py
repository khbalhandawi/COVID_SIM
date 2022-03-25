import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from functionsUtilities.utils import load_matrix
from functionsDynamics.visDeterministic import build_fig_SIR,draw_SIR_compare,draw_R0_compare

#=============================================================================
# Main execution 
if __name__ == '__main__':

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams['font.family'] = 'serif'

    pop_size = 1000
    healthcare_capacity = 90
    x_scaling = 0.1

    # data = load_matrix('SIRF_data', folder='population')
    # fig_sir, spec_sir, ax1_sir = build_fig_SIR()
    # draw_SIR(fig_sir, ax1_sir, data=data, , healthcare_capacity = healthcare_capacity)

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

        R0_time_axis = run_data_R0[:,0] * x_scaling
        time_data = np.arange(len(I)) * x_scaling # time vector for plot

        # Legend labels
        legend_label = "Run %i: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(run+1,point[0],point[1],point[2])
        labels += [legend_label]

        run += 1

    fig_1, _, ax1_1 = build_fig_SIR()
    draw_SIR_compare(data_I, fig_1, ax1_1, time=[time_data,], labels=labels, palette = palette, 
        save_name = 'I_compare', xlim = 350, leg_location = 'upper right', plot_path="data_dynamic",
        y_label = 'Number of infections $n_I^k$')
    
    fig_2, _, ax1_2 = build_fig_SIR()
    draw_SIR_compare(data_F, fig_2, ax1_2, time=[time_data,], labels=labels, palette = palette, 
        save_name = 'F_compare', xlim = 350, plot_path="data_dynamic",
        y_label = 'Number of fatalities $n_F^k$')

    fig_3, _, ax1_3 = build_fig_SIR()
    draw_SIR_compare(data_R, fig_3, ax1_3, time=[time_data,], labels=labels, palette = palette, 
        save_name = 'R_compare', xlim = 350, plot_path="data_dynamic",
        y_label = 'Number of recoveries $n_R^k$')

    fig_4, _, ax1_4 = build_fig_SIR()
    draw_SIR_compare(data_M, fig_4, ax1_4, time=[time_data,], labels=labels, palette = palette, 
        save_name = 'M_compare', xlim = 350, leg_location = 'lower right', plot_path="data_dynamic",
        y_label = 'Socio-economic impact $M^k$')

    fig_5, _, ax1_5 = build_fig_SIR()
    draw_R0_compare(data_R0, fig_5, ax1_5, time=[R0_time_axis,], labels=labels, palette = palette, 
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