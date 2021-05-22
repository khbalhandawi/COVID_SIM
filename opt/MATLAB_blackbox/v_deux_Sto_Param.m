function z = v_deux_Sto_Param(call_type,n_k,epsilon_f,Delta,gamma,tau,max_bb_eval,nb_proc,n_eval_k_success,folder)

switch call_type
    
%---------------------------------------------------------------------------------------------------    

    case 1      % default values
        obj_is_noisy = 1;                           % 1
        constr_vec_is_noisy = 1;                    % 2
        dimension = 3;                              % 3
        number_of_constr = 1;                       % 4
        x_0_infeas = [0.5 0.5 0.5]';                % 5       
        x_0_feas = [];                              % 6
        lower_bound = (0) * zeros(dimension,1);     % 7  
        upper_bound = (1) * ones(dimension,1);      % 8
        obj_sampling_rate = 1;                      % 9
        constr_sampling_rate = 1;                   % 10
        max_bb_eval = 10000;                        % 11          1000 * (dimension + 1); 
        Delta = 1;                                  % 12
        epsilon_f = 0.1;                            % 13
        gamma = 5;                                  % 14
        tau = 1/2;                                  % 15
        number_of_iterations = 10^10;               % 16
        nb_proc = 1;                                % 17
        n_eval_k_success = 10;                      % 18
        folder = 'StoMADS_exp/Run_1';               % 19

        z = {obj_is_noisy; constr_vec_is_noisy; dimension; number_of_constr; ...
            x_0_infeas; x_0_feas; lower_bound; upper_bound; obj_sampling_rate; ...
            constr_sampling_rate; max_bb_eval; Delta; epsilon_f; gamma; tau; number_of_iterations; ...
            nb_proc; n_eval_k_success ; folder};  

%---------------------------------------------------------------------------------------------------
    case 2      % external function call
        obj_is_noisy = 1;                           % 1
        constr_vec_is_noisy = 1;                    % 2
        dimension = 3;                              % 3
        number_of_constr = 1;                       % 4
        x_0_infeas = [0.5 0.5 0.5]';                % 5       
        x_0_feas = [];                              % 6
        lower_bound = (0) * zeros(dimension,1);           
        upper_bound = (1) * ones(dimension,1);      % 8
        obj_sampling_rate = n_k;                    % 9
        constr_sampling_rate = n_k;                 % 10
        number_of_iterations = 10^10;               % 16

        z = {obj_is_noisy; constr_vec_is_noisy; dimension; number_of_constr; ...
            x_0_infeas; x_0_feas; lower_bound; upper_bound; obj_sampling_rate; ...
            constr_sampling_rate; max_bb_eval; Delta; epsilon_f; gamma; tau; number_of_iterations; ...
            nb_proc; n_eval_k_success; folder};  
        
end