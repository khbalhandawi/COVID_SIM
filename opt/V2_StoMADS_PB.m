
% Basic version of StoMADS_PB, without Search Step, adapted for the
% theoretical paper

function V2_StoMADS_PB(nprob,ceal_in)
tic()

fprintf('StoMADS run {\n')
fprintf('\n')
title_string = '           BBE      (         SOL          )          OBJ\n';
fprintf(title_string)
fprintf('\n')

% PB-StoMADS
addpath StoMADS_V2
format long g

global ceal
ceal = ceal_in;

%% ===================================================================== %%
% Clear past runs

folder = ceal{19};

hist_file_name = [folder,'/STOMADS_hist' num2str(nprob) '.txt'];  % Check if hist_file_name already exists in the directory
if isfile(hist_file_name)
    delete(hist_file_name);
end
hist_file = fopen(hist_file_name, 'w');
fprintf(hist_file, 'bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(hist_file); 

f_hist_file_name = [folder,'/f_hist_STOMADS' num2str(nprob) '.txt'];  % Check if f_hist_file_name already exists in the directory
if isfile(f_hist_file_name)
    delete(f_hist_file_name);
end
f_hist_file = fopen(f_hist_file_name, 'w');
fprintf(f_hist_file, 'bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(f_hist_file); 

i_hist_file_name = [folder,'/i_hist_STOMADS' num2str(nprob) '.txt'];  % Check if i_hist_file_name already exists in the directory
if isfile(i_hist_file_name)
    delete(i_hist_file_name);
end
i_hist_file = fopen(i_hist_file_name, 'w');
fprintf(i_hist_file, 'bbe,x1,x2,x3,f,cstr,p_value,h,u\n');
fclose(i_hist_file); 

f_hist_file_name = [folder,'/f_progress_STOMADS' num2str(nprob) '.txt'];  % Check if f_hist_file_name already exists in the directory
if isfile(f_hist_file_name)
    delete(f_hist_file_name);
end
f_hist_file = fopen(f_hist_file_name, 'w');
fprintf(f_hist_file, 'n_success,bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(f_hist_file); 

i_hist_file_name = [folder,'/i_progress_STOMADS' num2str(nprob) '.txt'];  % Check if i_hist_file_name already exists in the directory
if isfile(i_hist_file_name)
    delete(i_hist_file_name);
end
i_hist_file = fopen(i_hist_file_name, 'w');
fprintf(i_hist_file, 'n_success,bbe,x1,x2,x3,f,cstr,p_value,h,u\n');
fclose(i_hist_file); 

%% ===================================================================== %%
% Initialization
global hist hist_avg f_hist i_hist f_progress i_progress;

hist = []; % this includes the raw blackbox evaluations (used for logging only)
hist_avg = []; % this includes the running average of blackbox evaluations (used for logging only)
f_hist = []; % this includes the running average of feasible incumbent (used for logging only)
i_hist = []; % this includes the running average of infeasible incumbent (used for logging only)
f_progress = []; % this includes past feasible success (computed for eval_k_success samples)
i_progress = []; % this includes past infeasible success (computed for eval_k_success samples)

history = []; % this includes the raw blackbox evaluations (used by the search algorithm)
profile_feasible = [];
profile_infeasible = [];

epsilon = ceal{13};
C = ceal{14};
tau = ceal{15};
eval_k = ceal{9};
eval_k_success = ceal{18};
N = ceal{16};
Delta = ceal{12};

Lambda = Delta;
%hmax = Inf;
x_infeas = ceal{5};
if iscolumn(x_infeas')
    x_infeas=x_infeas';
end
x_feas = [];
%Infeas_matrix = [];
ds = [];
xf_ds = [];
xi_ds = [];
%hmatrix = [];
flag_feas = 0;
%failure = 0;
% f_Dom = 0;
% h_Dom = 0;
% Imp = 0;

%% ===================================================================== %%
% MADS iterations
for i = 1:N
    
    %disp(['i = ' num2str(i)])
    %disp(['Delta = ' num2str(Delta)])
    
    if isempty(hist) == 1 % log initial guess
        eval_matrix_success(x_infeas, eval_k_success, nprob, 1)
        i_progress = f_progress;
        hist_file = fopen([folder,'/i_progress_STOMADS' num2str(nprob) '.txt'], 'a');
        formatting = [repmat('%12.20f,',1, length(x_infeas) + ceal{4} + 5) '\n']; formatting(end-2) = ''; % remove last delimliter
        fprintf(hist_file, formatting, i_progress(end,:)');
        fclose(hist_file); 
    end
    
    if Delta < 10^(-100)
        break
    end
    
    delta = min(Delta, Delta^2);
    %% ================================================================= %%
    %----------------------- Warning messags -----------------------------%
    
    if sum(length(ceal{5}) ~= length(ceal{7})) == 1
        warning('lower_bound is not the same length as x_0_infeas! Ensure they are both Column Vectors')
        warning('Error in Sto_Param.m')
        break
    end
    if sum(length(ceal{5}) ~= length(ceal{8})) == 1
        warning('upper_bound is not the same length as x_0_infeas! Ensure they are both Column Vectors')
        warning('Error in Sto_Param.m')
        break
    end
    
    if sum(x_infeas < ceal{7}) > 0
        warning('Error using StoMADS-PB')
        warning('StoMADS-PB Parameter Error: ')
        warning('Invalid Parameter (Sto_Param.m): x_0_infeas < lower_bound')
        break
    end
    if sum(x_infeas > ceal{8}) > 0
        warning('Error using StoMADS-PB ')
        warning('StoMADS-PB Parameter Error: ')
        warning('Invalid Parameter (Sto_Param.m): x_0_infeas > upper_bound')
        break
    end
    %% ================================================================= %%
    %-------------------- Estimates computation --------------------------%
    
    history = [history; [repmat(x_infeas',eval_k,1), eval_matrix(x_infeas, eval_k, nprob, x_feas, x_infeas)]];
    
    RowIdx = find(ismember(history(:,1:ceal{3}), x_infeas','rows'));
    Infeas_matrix = history(RowIdx, (ceal{3} + 1):end);    
    Infeas_mean = mean(Infeas_matrix, 1);     
    fok_infeas = Infeas_mean(1);
    hok_infeas = sum(max(Infeas_mean(2:end), 0));
    hmax = sum(max(Infeas_mean(2:end) + epsilon*Lambda^2, 0));
    
    if isempty(x_feas) == 0
        history = [history; [repmat(x_feas',eval_k,1), eval_matrix(x_feas, eval_k, nprob, x_feas, x_infeas)]];
        
        FIdx = find(ismember(history(:,1:ceal{3}), x_feas','rows'));
        Feas_matrix = history(FIdx, (ceal{3} + 1):end);
        Feas_mean = mean(Feas_matrix, 1);      
        fok_feas = Feas_mean(1);
    end    
    %% ================================================================= %%
    %------------------------- Search Step -------------------------------%

    ampl = 10;
    lb = ceal{7}; ub = ceal{8};
    for b_it = 1:length(ub)
        if (lb(b_it) == -Inf) && (ub(b_it) == Inf)
            if isempty(x_feas) == 0
                lb(b_it) = -ampl-abs(x_feas(b_it)); ub(b_it) = ampl+abs(x_feas(b_it)); 
            else
                lb(b_it) = -ampl-abs(x_infeas(b_it)); ub(b_it) = ampl+abs(x_infeas(b_it)); 
            end            
        elseif (lb(b_it) == -Inf) && (ub(b_it) ~= Inf)
            if isempty(x_feas) == 0
                lb(b_it) = -ampl-abs(x_feas(b_it));
            else
                lb(b_it) = -ampl-abs(x_infeas(b_it));
            end            
        elseif (lb(b_it) ~= -Inf) && (ub(b_it) == Inf)
            if isempty(x_feas) == 0
                ub(b_it) = ampl+abs(x_feas(b_it)); 
            else
                ub(b_it) = ampl+abs(x_infeas(b_it)); 
            end            
        end
    end
    t = unifrnd(lb,ub); % this is essentially a grid search
    history = [history; [repmat(t',eval_k,1), eval_matrix(t, eval_k, nprob, x_feas, x_infeas)]];
    
    SIdt = find(ismember(history(:,1:ceal{3}), t','rows'));
    Pt_matrix = history(SIdt, (ceal{3} + 1):end);
    Pt_mean = mean(Pt_matrix, 1);              
    usk = sum(max(Pt_mean(2:end) + epsilon*Lambda^2, 0));
    fsk = Pt_mean(1);
    hsk = sum(max(Pt_mean(2:end), 0));    

    %----------------------------------------------------------------------
    % Succesful
    % feasible (f-dominating)
    if ((flag_feas == 0) && (usk == 0)) || ((flag_feas == 1) && (usk == 0) && (fsk - fok_feas <= -C*epsilon*Lambda^2)) % f-dominating
        x_feas = t;
        eval_matrix_success(t, eval_k_success, nprob, 1); % log success
        
        flag_feas = 1;
        Delta = (1/tau^2)* Delta;
        if (isempty(xf_ds) == 0)
            ds = x_feas - xf_ds;
        end
        xf_ds = x_feas;
        continue
        
    % infeasible (h-dominating)
    elseif (hsk - hok_infeas <= -C * ceal{4} * epsilon * Lambda^2) && (fsk - fok_infeas <= -C*epsilon*Lambda^2) && (usk <= hmax) % h-dominating
        % && (usk > 0)
        x_infeas = t;
        eval_matrix_success(t, ceil(eval_k_success / 5), nprob, 2); % log success
        
        Delta = (1/tau^2) * Delta;
        if (isempty(xi_ds) == 0)
            ds = x_infeas - xi_ds;
        end
        xi_ds = x_infeas;
        continue
    end    
    
    %----------------------------------------------------------------------
    % Unsuccesful
    %% ================================================================= %%
    %----------------------- Poll construction ---------------------------%
    
    v = unifrnd(-10,10,ceal{3},1);
    
    if (isempty(ds) == 1)
        P_infeas = diag(x_infeas') * ones(length(x_infeas), 2 * length(x_infeas))...
            + delta * Poll(delta, Delta,v);
        
        if (isempty(x_feas) == 0)
            P_feas = diag(x_feas') * ones(length(x_feas), 2 * length(x_feas))...
                + delta * Poll(delta, Delta,v);
        else
            P_feas = [];
        end
    else
        
        P_infeas = diag(x_infeas') * ones(length(x_infeas), 2 * length(x_infeas))...
            + delta * order_last(Poll(delta, Delta,v), ds);
        
        if (isempty(x_feas) == 0)
            P_feas = diag(x_feas') * ones(length(x_feas), 2 * length(x_feas))...
                + delta * order_last(Poll(delta, Delta,v), ds);
        else
            P_feas = [];
        end
    end
    %     if (isempty(P_feas) == 0) && (isempty(P_infeas) == 0)
    %         P = union(P_feas', P_infeas', 'rows', 'stable')';
    %     else
    %         P = P_infeas;
    %     end
    %------------------------ Bounds control ------------------------------
    
    ipsi = 1;
    Psi = [];
    for w = 1:length(P_infeas(1,:))
        if (sum(P_infeas(:,w) >= ceal{7}) == length(ceal{7}) && sum(P_infeas(:,w) <= ceal{8}) == length(ceal{8}))
            Psi(:,ipsi)=P_infeas(:,w);
            ipsi = ipsi + 1;
        end
    end
    Psf = [];
    if (isempty(P_feas) == 0)
        ipsf = 1;
        for wf = 1:length(P_feas(1,:))
            if (sum(P_feas(:,wf) >= ceal{7}) == length(ceal{7}) && sum(P_feas(:,wf) <= ceal{8}) == length(ceal{8}))
                Psf(:,ipsf)=P_feas(:,wf);
                ipsf = ipsf + 1;
            end
        end
    end
    
    
    if isempty([Psi, Psf]) == 1
        Delta = tau^2 * Delta;
        continue
    end
    %----------------------------------------------------------------------
    % here

    %----------------------------------------------------------------------
    p_bool = 0;
    if (hmax == 0) && (isempty(x_feas) == 0)
        if (fok_infeas - fok_feas < -(2*epsilon*Lambda^2))
            if isempty(Psi) == 0
                P = Psi;
            else
                P = Psf;
            end
        else
            if isempty(Psf) == 0
                P = Psf;
            else
                P = Psi;
            end
        end
        p_bool = 1;
    end
    if (hmax > 0) && (isempty(x_feas) == 0)
        if (isempty(Psi) == 0) && (isempty(Psf) == 0)
            if (fok_infeas - fok_feas < -(2*epsilon*Lambda^2))
                if (sum(ismember(Psf', -Psf(:,1)', 'rows'))>0)
                    P = [Psi, Psf(:,1), -Psf(:,1)];
                else
                    P = [Psi, Psf(:,1)];
                end
            else
                if (sum(ismember(Psi', -Psi(:,1)', 'rows'))>0)
                    P = [Psf, Psi(:,1), -Psi(:,1)];
                else
                    P = [Psf, Psi(:,1)];
                end
            end
        else
            P = [Psi, Psf];
        end
        p_bool = 2;
    end
    if (p_bool == 0)
        P = [Psf, Psi];
    end
    
    %% ================================================================= %%
    %--------------------------- Poll Step -------------------------------%
    
    u_matrix = [];
    v_matrix = [];
    P_cell = [];
    i_u = 1;
    success = 0;
    %     failure = 0;
    %     f_Dom = 0;
    %     h_Dom = 0;
    %     Imp = 0;
    hf_Dom = 0;
    %     hf_Imp = 0;
    %Lambda = Delta;
    for j = 1: length(P(1,:))
        eval_poll = eval_k;
        %eval_poll =1+ floor(j/2);
        %eval_poll = 1;
        history = [history; [repmat(P(:,j)',eval_poll,1), eval_matrix(P(:,j), eval_poll, nprob, x_feas, x_infeas)]];
        
        SIdx = find(ismember(history(:,1:ceal{3}), P(:,j)','rows'));
        P_matrix = history(SIdx, (ceal{3} + 1):end);
        P_mean = mean(P_matrix, 1);           
        usk = sum(max(P_mean(2:end) + epsilon*Lambda^2, 0));
        fsk = P_mean(1);
        hsk = sum(max(P_mean(2:end), 0));
        
        if (hsk - hok_infeas <= -C * ceal{4} * epsilon * Lambda^2) && (fsk - fok_infeas <= -C*epsilon*Lambda^2) && (usk <= hmax) % (h-dominating)
            y_infeas = P(:,j);
            hf_Dom = 1;
        elseif (hsk - hok_infeas <= -C * ceal{4} * epsilon * Lambda^2) && (usk <= hmax) % (improving)
            v_matrix = [v_matrix; P(:,j)', usk];
            %                 hf_Imp = 1;
        end
        
        %----------------------------------------------------------------------
        % Succesful
        % feasible (f-dominating)
        if ((flag_feas == 0) && (usk == 0)) || ((flag_feas == 1) && (usk == 0) && (fsk - fok_feas <= -C*epsilon*Lambda^2))
            x_feas = P(:,j);
            eval_matrix_success(P(:,j), eval_k_success, nprob, 1); % log success
            
            flag_feas = 1;
            success = 1;
            Delta = (1/tau^2)* Delta;
            if (isempty(xf_ds) == 0)
                ds = x_feas - xf_ds;
            end
            xf_ds = x_feas;
            if (hf_Dom == 1) % also update infeasible incumbent if new point is h-dominating
                x_infeas = y_infeas;
                eval_matrix_success(y_infeas, ceil(eval_k_success / 5), nprob, 2, false); % log success (use last evaluation)
            end
            break % oppurtunistic poll
        
        % infeasible
        elseif (sum(ismember(P_infeas', P(:,j)', 'rows'))>0) && (hsk - hok_infeas <= -C * ceal{4} * epsilon * Lambda^2)
            u_matrix = [u_matrix; P(:,j)', usk];
            P_cell{i_u} = P_matrix;
            i_u = i_u + 1;
            
            % (h-dominating)
            if (fsk - fok_infeas <= -C*epsilon*Lambda^2) && (usk <= hmax)% && (usk > 0)
                x_infeas = P(:,j);
                eval_matrix_success(P(:,j), ceil(eval_k_success / 5), nprob, 2); % log success
                
                Delta = (1/tau^2) * Delta;
                success = 1;
                if (isempty(xi_ds) == 0)
                    ds = x_infeas - xi_ds;
                end
                xi_ds = x_infeas;
                break % oppurtunistic poll
            end
        end
    end
    
    %----------------------------------------------------------------------
    % Unsuccesful
    if (success == 0)
        % improving
        if (isempty(u_matrix) == 0)
            [~, idx] = min(u_matrix(:, end));
            x_infeas = u_matrix(idx, 1:ceal{3})';
            eval_matrix_success(u_matrix(idx, 1:ceal{3})', ceil(eval_k_success / 5), nprob, 2); % log success
            
            Delta = (1/tau^2)*Delta;
            %             Imp = 1;
        else
            Delta = tau^2 * Delta;
            Lambda = tau^2 * Lambda;
            %             failure = 1;
            
            %eval_k = min([eval_k + 1, 4]);
        end
    end
    
    %% ================================================================= %%
    %------------------- Log end of MADS iteration -----------------------%
    
    % disp(['hmax = ' num2str(hmax)])
    % if isempty(x_feas) == 0 % if x_feas is found (incumbent)
    %     disp(['x_feas = ' num2str(x_feas')])
    % end
    % 
    % disp(['x_infeas = ' num2str(x_infeas')])
    
    if (isempty(hist) == 0)
        if (size(hist, 1) > ceal{11})
            break
        end
    end
    %--------------------------- End -------------------------------------%
end
fprintf('\n')
title_string = '} end of run\n';
fprintf(title_string)
toc()
%clear
end