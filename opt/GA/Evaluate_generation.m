function Evaluate_generation(state,fitness,constraint,ceal)
    
    global n_sub_generations cumilative_f_evals f_best h_max stall_G_sub
    global x_feas
    %-------------------------------------------------------------%
    % Reset sub generation variables after an iteration is completed
    G = state.Generation;
    
    if isempty(state.Best) == 0
        
        %---------------------------------------------------------%
        % get current tolerance
        tol = abs(diff(state.Best));
        if isempty(tol) == 0
            current_tol = tol(end);
        else
            current_tol = 1e9;
        end

        %---------------------------------------------------------%
        % Extract optimum
        stall_G_sub = G - state.LastImprovement;
        n_sub_generations = G;
        
        x_best = state.Population(state.Score == state.Best(end),:);
        x_best = x_best(end,:);
        best_fitnesses = fitness(state.Score == state.Best(end),:);
        f_best = best_fitnesses(end,:);
        h_max = max(max(constraint, 0));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                         SUCCESS                         %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if stall_G_sub == 0

            %---------------------------------------------------------%
            % Print current generation results if new optimum is found
            x_string = repmat('%-.6f   ',1,ceal{3});
            fprintf(['        %-3d       %-5d       %-+012.6f  %-+012.3f %-3d       %-.6e   ( ',x_string,' )\n'],...
                n_sub_generations,cumilative_f_evals,f_best,h_max,stall_G_sub,current_tol,x_best)
            
            %---------------------------------------------------------%
            % Compute true objective and constraint values
            eval_k_success = ceal{18};
            x_feas = x_best;
            eval_matrix_success(x_best, eval_k_success, ceal)
            
            %---------------------------------------------------------%
            % Show progress
            shg(); % show current figure

        end

    else
                    
        % Initialize global variables
        h_max = 1e9; stall_G_sub = 0;
            
    end
end