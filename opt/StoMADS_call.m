function StoMADS_call(nprob,n_k,epsilon_f,Delta,gamma,tau,max_bb_eval,nb_proc,n_eval_k_success,folder)

    addpath MATLAB_blackbox
    addpath MATLAB_blackbox\support_functions
    addpath StoMADS_V2

    global index
    index = 0;

    % clear history
    d = dir('./data');
    filenames = {d.name};
    dirFlags = [d.isdir];
    % Loop through the filenames
    for i = 1:numel(filenames)
        fn = filenames{i};
        if ~dirFlags(i)
            delete([d(1).folder,'\',fn])
        end
    end
    
    ceal = v_deux_Sto_Param(2,n_k,epsilon_f,Delta,gamma,tau,max_bb_eval,nb_proc,n_eval_k_success,folder);
    V2_StoMADS_PB(nprob,ceal)
    
end