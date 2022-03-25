import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from functionsDynamics.visStochastic import build_fig_SIR,draw_SIR_compare,draw_R0_compare
from functionsDynamics.statsStochastic import process_statistics, process_statistics_CovidSim


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

    run = 0; labels = []; time = []; time_COVID_SIM_UI = []; time_CovidSim = []; labels_COVID_SIM_UI = []; 
    styles = []; z_orders = []; hatches = []

    data_I = []; data_F = []; data_R = []; data_M = []; data_R0 = []; 
    lb_data_I = []; lb_data_F = []; lb_data_R = []; lb_data_M = []; lb_data_R0 = []
    ub_data_I = []; ub_data_F = []; ub_data_R = []; ub_data_M = []; ub_data_R0 = []

    data_S = []; data_Critical = []; 
    lb_data_S = []; lb_data_Critical = []
    ub_data_S = []; ub_data_Critical = []

    ############################ CovidSim ################################
    # for plotting CovidSim results
    # load optimization points in terms of CovidSim solution space
    with open('data_dynamic/points_opts_CovidSim.pkl','rb') as fid:
        lob_var_CovidSim = pickle.load(fid)
        upb_var_CovidSim = pickle.load(fid)
        points_CovidSim = pickle.load(fid)
        points_unscaled_CovidSim = pickle.load(fid)
        solution_dict = pickle.load(fid)

    # select_indices = [1,2,3,4,5,6]
    # select_indices = [2,4,6]
    # select_indices = [1,3,5]
    # select_indices = [2,3,5,6]
    select_indices = [2,3,]
    # select_indices = [1,2,]
    prefixes = ["ABM",] * len(select_indices)
    # prefixes_CovidSim = ["",] * len(select_indices)

    prefixes_CovidSim = []
    for i in select_indices:

        for key,value in solution_dict.items():
            if solution_dict[key]["index"] == i:
                stats = solution_dict[key]
                break
        
        # prefixes_CovidSim += [r"CovidSim %s $n^k=%i$" %(stats["algo"],stats["n_k"])]
        prefixes_CovidSim += [r"CovidSim"]

    y_label_add = ' ($\%$ population)'
    y_scaling = 1
    threshold = 0.09
    ylims = [0.135,0.22,0.6,None]
    xlims = [350,350,350,350]

    points_unscaled = [points_unscaled[index] for index in select_indices]
    runs = select_indices

    #[abm,abm,covidsim,covidsim,] --> [covidsim,abm,covidsim,abm,]
    m = list(range(2*len(select_indices)))
    for i in range(len(select_indices)):
        m[i] = 2*i+1
        m[len(select_indices)+i] = 2*i

    print(m)
    counts = [2] * len(palette) # repeat each color once
    palette = [item for item, count in zip(palette, counts) for i in range(count)]
    ######################################################################

    styles_COVID_SIM_UI = [(0, (5, 0, 5, 0)),(0, (5, 5, 5, 5))]
    z_orders_COVID_SIM_UI = [2,1]
    hatches_COVID_SIM_UI = ["+","////"]

    ########################## COVID_SIM_UI ##############################
    for point,run,prefix in zip(points_unscaled,runs,prefixes):

        data = process_statistics(run,x_scaling=1/10,y_scaling=y_scaling,conf_interval=50,use_percentiles=True)
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
    
    styles += styles_COVID_SIM_UI
    z_orders += z_orders_COVID_SIM_UI
    hatches += hatches_COVID_SIM_UI

    styles_CovidSim = [(0, (5, 1, 5, 1)),(0, (1, 1, 1, 1))]
    z_orders_CovidSim = [4,3]
    hatches_CovidSim = ["xxxx","\\"]

    ############################ CovidSim ################################
    for point,run,prefix in zip(points_unscaled,runs,prefixes_CovidSim):

        data = process_statistics_CovidSim(run,x_scaling=1/2,conf_interval=80,use_percentiles=True)
        # data = process_statistics_CovidSim(run,x_scaling=1/2,conf_interval=1,use_percentiles=False)

        [mean_process_I, mean_process_F, mean_process_R, mean_process_S, mean_process_Critical, 
            ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
            ub_process_S, lb_process_S, ub_process_Critical, lb_process_Critical, time_data] = data

        data_I += [mean_process_I]; lb_data_I += [lb_process_I]; ub_data_I += [ub_process_I]
        data_F += [mean_process_F]; lb_data_F += [lb_process_F]; ub_data_F += [ub_process_F]
        data_R += [mean_process_R]; lb_data_R += [lb_process_R]; ub_data_R += [ub_process_R]
        data_S += [mean_process_S]; lb_data_S += [lb_process_S]; ub_data_S += [ub_process_S]
        data_Critical += [mean_process_Critical]; lb_data_Critical += [lb_process_Critical]; ub_data_Critical += [ub_process_Critical]

        time += [time_data]
        time_CovidSim += [time_data]

        # Legend labels
        # prefix = "CovidSim Solution %i" %(run+1) # uncomment for generic legend labels
        # prefix = "CovidSim Run %i" %(run+1) # uncomment for generic legend labels

        legend_label = "%s: $n_E$ = %i, $S_D$ = %.3f, $n_T$ = %i" %(prefix,round(point[0]),point[1],round(point[2]))
       
        labels += [legend_label]
        # styles += [(0, (5, 5, 5, 5))]
    styles += styles_CovidSim
    z_orders += z_orders_CovidSim
    hatches += hatches_CovidSim

    data_I = [y for x,y in sorted(zip(m,data_I))] 
    data_F = [y for x,y in sorted(zip(m,data_F))] 
    data_R = [y for x,y in sorted(zip(m,data_R))] 
    # data_S = [y for x,y in sorted(zip(m,data_S))] 
    lb_data_I = [y for x,y in sorted(zip(m,lb_data_I))] 
    lb_data_F = [y for x,y in sorted(zip(m,lb_data_F))] 
    lb_data_R = [y for x,y in sorted(zip(m,lb_data_R))] 
    # lb_data_S = [y for x,y in sorted(zip(m,lb_data_S))] 
    ub_data_I = [y for x,y in sorted(zip(m,ub_data_I))] 
    ub_data_F = [y for x,y in sorted(zip(m,ub_data_F))] 
    ub_data_R = [y for x,y in sorted(zip(m,ub_data_R))] 
    # ub_data_S = [y for x,y in sorted(zip(m,ub_data_S))] 
    # data_Critical = [y for x,y in sorted(zip(m,data_Critical))] 
    # lb_data_Critical = [y for x,y in sorted(zip(m,lb_data_Critical))] 
    # ub_data_Critical = [y for x,y in sorted(zip(m,ub_data_Critical))] 
    labels = [y for x,y in sorted(zip(m,labels))] 
    time = [y for x,y in sorted(zip(m,time))] 
    styles = [y for x,y in sorted(zip(m,styles))] 
    z_orders = [y for x,y in sorted(zip(m,z_orders))] 
    hatches = [y for x,y in sorted(zip(m,hatches))] 
    ######################################################################

    fig_1, _, ax1_1 = build_fig_SIR()
    draw_SIR_compare(data_I, time, fig_1, ax1_1, lb_data_I, ub_data_I, labels=labels, palette = palette, 
        save_name = 'I_compare_CovidSim', xlim = xlims[0], ylim = ylims[0], leg_location = 'upper right', plot_path="data_dynamic",
        y_label = 'Infections%s $n_I^t$' %y_label_add, threshold=threshold, threshold_label="Healthcare capacity $H_{\mathrm{max}}$",
        styles=styles, z_orders=z_orders, hatch_patterns=hatches)

    fig_2, _, ax1_2 = build_fig_SIR() 
    draw_SIR_compare(data_F, time, fig_2, ax1_2, lb_data_F, ub_data_F, labels=labels, palette = palette, 
        save_name = 'F_compare_CovidSim', xlim = xlims[1], ylim = ylims[1], leg_location = 'upper left', plot_path="data_dynamic",
        y_label = 'Fatalities%s $n_F^t$' %y_label_add)

    fig_3, _, ax1_3 = build_fig_SIR()
    draw_SIR_compare(data_R, time, fig_3, ax1_3, lb_data_R, ub_data_R, labels=labels, palette = palette, 
        save_name = 'R_compare_CovidSim', xlim = xlims[2], ylim = ylims[2], leg_location = 'upper left', plot_path="data_dynamic",
        y_label = 'Recoveries%s $n_R^t$' %y_label_add)

    fig_4, _, ax1_4 = build_fig_SIR()
    draw_SIR_compare(data_M, time_COVID_SIM_UI, fig_4, ax1_4, lb_data_M, ub_data_M, labels=labels_COVID_SIM_UI, palette = palette_COVID_SIM_UI, 
        save_name = 'M_compare_CovidSim', xlim = xlims[3], ylim = ylims[3], leg_location = 'lower right', plot_path="data_dynamic",
        y_label = 'Mobility $M^t$', styles=styles_COVID_SIM_UI, z_orders=z_orders_COVID_SIM_UI, hatch_patterns=hatches_COVID_SIM_UI)

    fig_5, _, ax1_5 = build_fig_SIR()
    draw_R0_compare(data_R0, R0_time_data, fig_5, ax1_5, lb_data_R0, ub_data_R0, labels=labels_COVID_SIM_UI, palette = palette_COVID_SIM_UI, 
        save_name = 'R0_compare_CovidSim', xlim = 350, leg_location = 'upper right', plot_path="data_dynamic",
        y_label = 'Basic reproductive number $R_0$', line_label = "$R_0$", threshold = 1, threshold_label='$R_0=1$')

    plt.show()