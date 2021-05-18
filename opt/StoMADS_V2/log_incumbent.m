function log_incumbent(x_infeas, x_feas, history, eval_k, nprob, epsilon, Lambda, hmax)
% update feasible and infeasible solutions logs
global hist f_hist i_hist ceal

for i = 1:eval_k
    
    % bbe = size(history,1)+1 - eval_k + i; 
    bbe = hist(end,1) - eval_k + i;
    
    % store feasible improvement history
    if isempty(x_feas) == 0
        [f_feas,c_feas,p_feas,h_feas,u_feas] = get_updated_estimates(history,x_feas,nprob,epsilon,Lambda);
        f_hist = [f_hist; [bbe, x_feas', f_feas, c_feas, p_feas, h_feas] ];
    else
        [f_feas,c_feas,p_feas,h_feas,u_feas] = get_updated_estimates(history,x_infeas,nprob,epsilon,Lambda);
        f_hist = [f_hist; [bbe, x_infeas', f_feas, c_feas, p_feas, h_feas] ];
    end

    f_hist_file = fopen(['f_hist_long' num2str(nprob) '.txt'], 'w');
    fprintf(f_hist_file, [repmat('%12.20f ',1, ceal{3} + ceal{4} + 4) '\n'],f_hist');
    fclose(f_hist_file);

    % store infeasible improvement history
    [f_infeas,c_infeas,p_infeas,h_infeas,u_infeas] = get_updated_estimates(history,x_infeas,nprob,epsilon,Lambda);
    i_hist = [i_hist; [bbe, x_infeas', f_infeas, c_infeas, p_infeas, h_infeas, u_infeas, hmax] ];

    i_hist_file = fopen(['i_hist_long' num2str(nprob) '.txt'], 'w');
    fprintf(i_hist_file, [repmat('%12.20f ',1, ceal{3} + ceal{4} + 6) '\n'],i_hist');
    fclose(i_hist_file);
end

end

    