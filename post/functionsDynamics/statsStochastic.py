
import pickle
import numpy as np

def process_statistics(run,folder='data_dynamic/'):

    with open('%s/MCS_process_data_r%i.pkl' %(folder,run),'rb') as fid:
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