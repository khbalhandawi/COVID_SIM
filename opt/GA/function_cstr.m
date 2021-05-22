%=========================================================================%
%                      'SMART' CONSTRAINT FUNCTION                        %
%=========================================================================%
function [c,ceq] = function_cstr(x,ceal)
    global cumilative_f_evals GA_hist x_feas
    eval_k = ceal{10};
    
    if isempty(GA_hist) == 0 % check if histroy exists
    
        RowIdx = find(ismember(GA_hist(:,1:ceal{3}), x,'rows'));

        if isempty(RowIdx) == 0 % If point has been previously evaluated

            f = GA_hist(RowIdx(end), ceal{3}+1);       
            c = GA_hist(RowIdx(end), ceal{3}+ceal{4}+1); 
            
        else  % Compute point
            % fprintf('cstr \n')
            z = eval_matrix(x, eval_k, x_feas, ceal);
            mean_z = mean(z,1);
                        
            % Update GA lookup table and counter
            GA_hist = [GA_hist; x mean_z];
            cumilative_f_evals = eval_k + cumilative_f_evals;
            
            mean_c = mean_z(2:ceal{4}+2-1);
            c = mean_c;
            
        end

    else  % Compute point

        z = eval_matrix(x, eval_k, x_feas, ceal);
        mean_z = mean(z,1);
        
        % Update GA lookup table and counter
        GA_hist = [GA_hist; x mean_z];
        cumilative_f_evals = eval_k + cumilative_f_evals;
        
        mean_c = mean_z(2:ceal{4}+2-1);
        c = mean_c;
        
    end
    
    ceq = [];
    
end