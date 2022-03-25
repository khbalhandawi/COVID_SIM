
import pickle
import numpy as np

def process_realizations(run,folder='data_dynamic/',x_scaling=1,y_scaling=1,time_shift=0, n_realization=5):

    with open('%s/MCS_process_data_r%i.pkl' %(folder,run),'rb') as fid:
        process_I = pickle.load(fid)
        process_F = pickle.load(fid)
        process_R = pickle.load(fid)
        process_M = pickle.load(fid)
        process_R0 = pickle.load(fid)

    # Shift all data by a certain amount of time
    process_I_shift=[];process_R_shift=[];process_F_shift=[];process_M_shift=[];process_R0_shift=[];
    for I,R,F,M,R0 in zip(process_I,process_R,process_F,process_M,process_R0):

        I = [x * y_scaling for x in I]
        R = [x * y_scaling for x in R]
        F = [x * y_scaling for x in F]
        M = [x *(2000/3500) for x in M]

        I = [I[0]]*time_shift + I
        R = [R[0]]*time_shift + R
        F = [F[0]]*time_shift + F
        M = [M[0]]*time_shift + M
        R0 = np.vstack((np.repeat([R0[0,:],], repeats=int(time_shift/20), axis=0),R0))

        process_I_shift+=[I]
        process_R_shift+=[R]
        process_F_shift+=[F]
        process_M_shift+=[M]
        process_R0_shift+=[R0]

    process_I = process_I_shift[n_realization]
    process_F = process_R_shift[n_realization]
    process_R = process_F_shift[n_realization]
    process_M = process_M_shift[n_realization]
    process_R0 = process_R0_shift[n_realization]

    #================================================
    # Get simulation time
    R0_time_axis = process_R0_shift[0][:,0] * x_scaling
    time_data = np.arange(len(process_I_shift[0])) * x_scaling # time vector for plot

    #================================================
    # return stats data

    data = [process_I, process_F, process_R, process_M, process_R0, R0_time_axis, time_data]

    return data

def process_statistics(run,folder='data_dynamic/',x_scaling=1,y_scaling=1,time_shift=0,conf_interval=50,use_percentiles=True):

    with open('%s/MCS_process_data_r%i.pkl' %(folder,run),'rb') as fid:
        process_I = pickle.load(fid)
        process_F = pickle.load(fid)
        process_R = pickle.load(fid)
        process_M = pickle.load(fid)
        process_R0 = pickle.load(fid)

    # Shift all data by a certain amount of time
    process_I_shift=[];process_R_shift=[];process_F_shift=[];process_M_shift=[];process_R0_shift=[];
    for I,R,F,M,R0 in zip(process_I,process_R,process_F,process_M,process_R0):

        I = [x * y_scaling for x in I]
        R = [x * y_scaling for x in R]
        F = [x * y_scaling for x in F]

        I = [I[0]]*time_shift + I
        R = [R[0]]*time_shift + R
        F = [F[0]]*time_shift + F
        M = [M[0]]*time_shift + M
        R0 = np.vstack((np.repeat([R0[0,:],], repeats=int(time_shift/20), axis=0),R0))

        process_I_shift+=[I]
        process_R_shift+=[R]
        process_F_shift+=[F]
        process_M_shift+=[M]
        process_R0_shift+=[R0]

    process_I = process_I_shift
    process_F = process_R_shift
    process_R = process_F_shift
    process_M = process_M_shift
    process_R0 = process_R0_shift

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
    ub_process_R = []
    ub_process_F = []
    ub_process_M = []
    lb_process_I = []
    lb_process_R = []
    lb_process_F = []
    lb_process_M = []
    median_process_I = []
    median_process_R = []
    median_process_F = []
    median_process_M = []

    for k in range(n_increments):
        
        dist_I = []; dist_R = []; dist_F = []; dist_M = []; dist_R0 = [];
        for I,R,F,M,_ in zip(process_I,process_R,process_F,process_M,process_R0):

            dist_I += [I[k]]
            dist_R += [R[k]]
            dist_F += [F[k]]
            dist_M += [M[k]*(2000/3500)]

        if use_percentiles:
            ub_I = np.percentile(dist_I, 50+conf_interval/2); ub_process_I += [ub_I]
            ub_R = np.percentile(dist_R, 50+conf_interval/2); ub_process_R += [ub_R]
            ub_F = np.percentile(dist_F, 50+conf_interval/2); ub_process_F += [ub_F]
            ub_M = np.percentile(dist_M, 50+conf_interval/2); ub_process_M += [ub_M]
            lb_I = np.percentile(dist_I, 50-conf_interval/2); lb_process_I += [lb_I]
            lb_R = np.percentile(dist_R, 50-conf_interval/2); lb_process_R += [lb_R]
            lb_F = np.percentile(dist_F, 50-conf_interval/2); lb_process_F += [lb_F]
            lb_M = np.percentile(dist_M, 50-conf_interval/2); lb_process_M += [lb_M]

            median_I = np.percentile(dist_I, 50); median_process_I += [median_I]
            median_R = np.percentile(dist_R, 50); median_process_R += [median_R]
            median_F = np.percentile(dist_F, 50); median_process_F += [median_F]
            median_M = np.percentile(dist_M, 50); median_process_M += [median_M]
        else:
            ub_I = np.mean(dist_I) + conf_interval*np.std(dist_I,ddof=1); ub_process_I += [ub_I]
            ub_R = np.mean(dist_R) + conf_interval*np.std(dist_R,ddof=1); ub_process_R += [ub_R]
            ub_F = np.mean(dist_F) + conf_interval*np.std(dist_F,ddof=1); ub_process_F += [ub_F]
            ub_M = np.mean(dist_M) + conf_interval*np.std(dist_M,ddof=1); ub_process_M += [ub_M]
            lb_I = np.mean(dist_I) - conf_interval*np.std(dist_I,ddof=1); lb_process_I += [lb_I]
            lb_R = np.mean(dist_R) - conf_interval*np.std(dist_R,ddof=1); lb_process_R += [lb_R]
            lb_F = np.mean(dist_F) - conf_interval*np.std(dist_F,ddof=1); lb_process_F += [lb_F]
            lb_M = np.mean(dist_M) - conf_interval*np.std(dist_M,ddof=1); lb_process_M += [lb_M]

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

    R0_time_axis = process_R0[0][:,0] * x_scaling

    #================================================
    # Get simulation time

    time_data = np.arange(len(mean_process_I)) * x_scaling # time vector for plot

    #================================================
    # return stats data

    if use_percentiles:
        data = [median_process_I, median_process_F, median_process_R, median_process_M, mean_process_R0, 
                ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
                ub_process_M, lb_process_M, ub_process_R0, lb_process_R0, R0_time_axis,time_data]
    else:
        data = [mean_process_I, mean_process_F, mean_process_R, mean_process_M, mean_process_R0, 
                ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
                ub_process_M, lb_process_M, ub_process_R0, lb_process_R0, R0_time_axis,time_data]

    return data

    
def process_statistics_CovidSim(run,folder='data_dynamic/',x_scaling=1,conf_interval=50,use_percentiles=True):
    
    with open('%s/MCS_process_data_CovidSim_r%i.pkl' %(folder,run),'rb') as fid:
        process_I = pickle.load(fid)
        process_F = pickle.load(fid)
        process_R = pickle.load(fid)
        process_S = pickle.load(fid)
        process_Critical = pickle.load(fid)

    n_increments = len(process_I[0])
    n_samples = len(process_I)

    sum_I = np.zeros(n_increments)
    sum_R = np.zeros(n_increments)
    sum_F = np.zeros(n_increments)
    sum_S = np.zeros(n_increments)
    sum_Critical = np.zeros(n_increments)

    #================================================
    # Get distributions first (for SIRF and Mobility)

    ub_process_I = []
    ub_process_R = []
    ub_process_F = []
    ub_process_S = []
    ub_process_Critical = []
    lb_process_I = []
    lb_process_R = []
    lb_process_F = []
    lb_process_S = []
    lb_process_Critical = []
    median_process_I = []
    median_process_R = []
    median_process_F = []
    median_process_S = []
    median_process_Critical = []

    for k in range(n_increments):
        
        dist_I = []; dist_F = []; dist_R = []; dist_S = []; dist_Critical = [];
        for I,R,F,S,Critical in zip(process_I,process_R,process_F,process_S,process_Critical):

            dist_I += [I[k]]
            dist_R += [R[k]]
            dist_F += [F[k]]
            dist_S += [S[k]]
            dist_Critical += [Critical[k]]

        if use_percentiles:
            ub_I =          np.percentile(dist_I, 50+conf_interval/2);          ub_process_I +=         [ub_I]
            ub_R =          np.percentile(dist_R, 50+conf_interval/2);          ub_process_R +=         [ub_R]
            ub_F =          np.percentile(dist_F, 50+conf_interval/2);          ub_process_F +=         [ub_F]
            ub_S =          np.percentile(dist_S, 50+conf_interval/2);          ub_process_S +=         [ub_S]
            ub_Critical =   np.percentile(dist_Critical, 50+conf_interval/2);   ub_process_Critical +=  [ub_Critical]
            lb_I =          np.percentile(dist_I, 50-conf_interval/2);          lb_process_I +=         [lb_I]
            lb_R =          np.percentile(dist_R, 50-conf_interval/2);          lb_process_R +=         [lb_R]
            lb_F =          np.percentile(dist_F, 50-conf_interval/2);          lb_process_F +=         [lb_F]
            lb_S =          np.percentile(dist_S, 50-conf_interval/2);          lb_process_S +=         [lb_S]
            lb_Critical =   np.percentile(dist_Critical, 50-conf_interval/2);   lb_process_Critical +=  [lb_Critical]

            median_I =          np.percentile(dist_I, 50);          median_process_I +=         [median_I]
            median_R =          np.percentile(dist_R, 50);          median_process_R +=         [median_R]
            median_F =          np.percentile(dist_F, 50);          median_process_F +=         [median_F]
            median_S =          np.percentile(dist_S, 50);          median_process_S +=         [median_S]
            median_Critical =   np.percentile(dist_Critical, 50);   median_process_Critical +=  [median_Critical]

        else:
            ub_I =          np.mean(dist_I) + conf_interval*np.std(dist_I,ddof=1); ub_process_I += [ub_I]
            ub_R =          np.mean(dist_R) + conf_interval*np.std(dist_R,ddof=1); ub_process_R += [ub_R]
            ub_F =          np.mean(dist_F) + conf_interval*np.std(dist_F,ddof=1); ub_process_F += [ub_F]
            ub_S =          np.mean(dist_S) + conf_interval*np.std(dist_S,ddof=1); ub_process_S += [ub_S]
            ub_Critical =   np.mean(dist_Critical) + conf_interval*np.std(dist_Critical,ddof=1); ub_process_Critical += [ub_Critical]
            lb_I =          np.mean(dist_I) - conf_interval*np.std(dist_I,ddof=1); lb_process_I += [lb_I]
            lb_R =          np.mean(dist_R) - conf_interval*np.std(dist_R,ddof=1); lb_process_R += [lb_R]
            lb_F =          np.mean(dist_F) - conf_interval*np.std(dist_F,ddof=1); lb_process_F += [lb_F]
            lb_S =          np.mean(dist_S) - conf_interval*np.std(dist_S,ddof=1); lb_process_S += [lb_S]
            lb_Critical =   np.mean(dist_Critical) - conf_interval*np.std(dist_Critical,ddof=1); lb_process_Critical += [lb_Critical]

    #================================================
    # Get means

    for I,R,F,M,Critical in zip(process_I,process_R,process_F,process_S,process_Critical):

        sum_I += I; sum_R += R; sum_F += F; sum_S += M; sum_Critical += Critical; 

    mean_process_I = sum_I / n_samples
    mean_process_R = sum_R / n_samples
    mean_process_F = sum_F / n_samples
    mean_process_S = sum_S / n_samples 
    mean_process_Critical = sum_Critical / n_samples

    #================================================
    # Get simulation time

    time_data = np.arange(len(mean_process_I)) * x_scaling # time vector for plot

    #================================================
    # return stats data

    if use_percentiles:
        data = [median_process_I, median_process_F, median_process_R, median_process_S, median_process_Critical, 
            ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
            ub_process_S, lb_process_S, ub_process_Critical, lb_process_Critical,time_data]
    else:
        data = [mean_process_I, mean_process_F, mean_process_R, mean_process_S, mean_process_Critical, 
                ub_process_I, lb_process_I, ub_process_R, lb_process_R, ub_process_F, lb_process_F, 
                ub_process_S, lb_process_S, ub_process_Critical, lb_process_Critical,time_data]

    return data
