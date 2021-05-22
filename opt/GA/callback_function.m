%=========================================================================%
%            GA OUTPUT FUNCTION (called after each generation)            %
%=========================================================================%
function [state,options,optchanged] = callback_function(options,state,flag,ceal)
    %callback_function Plots the mean and the range of the population.
    %   [state,options,optchanged] = callback_function(OPTIONS,STATE,FLAG) plots the mean and the range
    %   (highest and the lowest distance) of individuals.  
    %   (highest and the lowest score) of individuals. 
    %   Copyright 2021-2020 Khalil Al Handawi
    global fig ax2 ax3 % axis handles for plotting progress
    global cumilative_f_evals % variables for controlling total computational budget
    global max_generations_per_it max_stall_generations_per_it % variables for controlling augmented lagrange step
    global h_max stall_G_sub f_best n_sub_generations % variables for keeping track of augmented lagrange step
    global GA_hist
    optchanged = false;
    
    G = state.Generation;
    population = state.Population;
    score = state.Score;
    
    % Retrieve fitness and constraints off population
    for p_i = 1:1:size(population,1)
        RowIdx(p_i) = find(ismember(GA_hist(:,1:ceal{3}), population(p_i,:),'rows'));
    end
    fitness = GA_hist(RowIdx, ceal{3}+1);
    constraint = GA_hist(RowIdx, ceal{3}+ceal{4}+1);

    % Score mean
    smean = nanmean(fitness); % Ignore infeasible individuals with a NaN score
    Y = smean;
    L = min(fitness);
    U = max(fitness);
    
    % Distance mean
    population_center = mean(population,1); % mean along rows
    % Subtract population center from all the columns.
    differences = bsxfun(@minus,population,population_center);
    distances = vecnorm(differences,2,2);
    Y_m = mean(distances);
    L_m = min(distances);
    U_m = max(distances);

    state.FunEval = cumilative_f_evals;
    
    switch flag

        case 'init'
            fprintf('        G_sub     F_evals     f_best        H_max        G_stall   Tol            (  x_best                )\n')
            fprintf('        ---------------------------------------------------------------------------------------------------\n')
            
            % Progress plots
            split_str = split(ceal{19},'_');
            run_number = str2num(split_str{end}); % get run number
            
            [fig,ax2,ax3] = build_fig(run_number);
            plotRange = errorbar(ax2,cumilative_f_evals,Y_m, Y_m - L_m, U_m - Y_m);
            score_plot = errorbar(ax3,cumilative_f_evals,Y, Y - L, U - Y);
            
            offset = 0.2*max(abs(U_m - L_m));
            set(ax2,'ylim',[min(L_m) - offset,max(U_m) + offset])
            offset = 0.2*max(abs(U - L));
            set(ax3,'ylim',[min(L) - offset,max(U) + offset])
            
            set(plotRange,'Tag','plot1drange');
            set(score_plot,'Tag','score_plot');
            
            %-------------------------------------------------------------%
            % Evaluate last generation results
            state.LastImprovement = 0;
            Evaluate_generation(state,fitness,constraint,ceal);
            
        case 'iter'
            if isempty(state.Best) == 0
                stall_G = G - state.LastImprovement;    
                fprintf('=============================================================================================================\n')
                fprintf('Iteration     G_sub     F_evals     f_best        H_max           S G      S G_sub   ( x_best                 )\n')
                
                % Extract optimum
                x_best = population(score == state.Best(end),:);
                x_best = x_best(end,:);
                
                % Print current generation results
                x_string = repmat('%-.6f   ',1,ceal{3});
                fprintf(['%-3d           %-3d       %-5d       %-+012.6f  %-+012.3f    %-3d      %-3d       ( ',x_string,' )\n'],...
                    G,n_sub_generations,cumilative_f_evals,f_best,h_max,stall_G,stall_G_sub,x_best)
                fprintf('=============================================================================================================\n')
            end
            
            % for sub iterations                                              
            fprintf('        G_sub     F_evals     f_best        H_max        G_stall   Tol            (  x_best                )\n')
            fprintf('        ---------------------------------------------------------------------------------------------------\n')
            
        case 'interrupt'
            
            %-------------------------------------------------------------%
            % Progress plots
            
            plotRange = findobj(get(ax2,'Children'),'Tag','plot1drange');
            newX = [get(plotRange,'Xdata') cumilative_f_evals];
            newY = [get(plotRange,'Ydata') Y_m];
            newL = [get(plotRange,'Ldata') (Y_m - L_m)];
            newU = [get(plotRange,'Udata') (U_m - Y_m)];   
            set(get(ax2,'xlabel'),'String','Function evaluations');
            set(get(ax2,'ylabel'),'String','Average population distance');
            set(plotRange,'Xdata',newX,'Ydata',newY,'Ldata',newL,'Udata',newU);
            lower_abs = newY - newL; upper_abs = newY + newU;
            offset = 0.2*max(abs(upper_abs - lower_abs));
            set(ax2,'ylim',[min(lower_abs) - offset,max(upper_abs) + offset]);
            
            if isempty(state.Best) == 0 && ~isnan(Y)
                score_plot = findobj(get(ax3,'Children'),'Tag','score_plot');
                newX = [get(score_plot,'Xdata') cumilative_f_evals];
                newY = [get(score_plot,'Ydata') Y];
                newL = [get(score_plot,'Ldata') (Y - L)];
                newU = [get(score_plot,'Udata') (U - Y)];     
                set(get(ax3,'xlabel'),'String','Function evaluations');
                set(get(ax3,'ylabel'),'String','Score');
                set(score_plot,'Xdata',newX,'Ydata',newY,'Ldata',newL,'Udata',newU);
                lower_abs = newY - newL; upper_abs = newY + newU;
                offset = 0.2*max(abs(upper_abs - lower_abs));
                set(ax3,'ylim',[min(lower_abs) - offset,max(upper_abs) + offset]);
            end
            
            %-------------------------------------------------------------%
            % Evaluate last generation results
            log_generation(Y,L,U,Y_m,L_m,U_m,ceal)
            Evaluate_generation(state,fitness,constraint,ceal);

            %-------------------------------------------------------------%
            % Stopping criteria 
            budget = ceal{11};
            if cumilative_f_evals >= budget
                state.StopFlag = ['computational budget of ',num2str(budget), ' reached'];
            end
            
            % if G > 0
            %     if G > max_generations_per_it || stall_G_sub > max_stall_generations_per_it
            %         state.Generation = G + options.MaxStallGenerations;
            %         state.Best(state.Generation) = state.Best(end);
            %     end
            % end
            
    end
    
end