function z = eval_matrix(x, eval_k, nprob, x_feas, x_infeas)

global hist hist_avg f_hist i_hist ceal;
z = zeros(eval_k, ceal{4}+1);
z_sur = zeros(eval_k, ceal{4}+1);
bbe = size(hist,1)+1; 
folder = ceal{19};
    
%% Evaluate blackbox eval_k times

if nprob == 34 % COVID problem only
    d = dir('./data');
    filenames = {d.name};
    % Loop through the filenames
    for i = 1:numel(filenames)
        fn = filenames{i};
        [num, cnt] = sscanf(fn(find(fn == '-', 1, 'last')+1:end-4), '%s');
        if cnt == 1 && strcmp(num,'matlab_out_Blackbox')
            % fprintf('deleting ...')
            % disp(fn) 
            delete([d(1).folder,'\',fn])
        end
    end
end

nb_proc = ceal{17};

if nb_proc ~= 1
    parfor (e = 1:eval_k, nb_proc)
        [z(e,:),z_sur(e,:)] = v_deux_black_box(x, nprob, bbe+e-1, e);
    end
else
    for e = 1:eval_k
        [z(e,:),z_sur(e,:)] = v_deux_black_box(x, nprob, bbe+e-1);
    end
end

%% Logging and update blackbox outputs
% Update history cache (ON by default)

% loop over results
for i = 1:size(z,1)

    % Update raw history
    bbe = size(hist,1)+1;   
    hist = [hist; bbe x' z(i,:) z_sur(i,:)];
end

% loop over results
for i = 1:size(z,1)

    bbe = hist(end,1) - eval_k + i;

    % Compute running average of raw history
    [fsk,csk,psk,hsk,~,fsk_sur,csk_sur,psk_sur,hsk_sur,~] = get_updated_estimates(hist(:,2:end),x,nprob);
    hist_avg = [hist_avg; [bbe, x', fsk, csk, psk, hsk, fsk_sur, csk_sur, psk_sur, hsk_sur] ];

    hist_file = fopen([folder,'/STOMADS_hist' num2str(nprob) '.txt'], 'a');
    format = [repmat('%12.20f,',1, ceal{3} + 2*ceal{4} + 7) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(hist_file, format, hist_avg(end,:)');
    fclose(hist_file); 

    % store feasible improvement history
    if isempty(x_feas) == 0
        [f_feas,c_feas,p_feas,h_feas,~,f_sur_feas,c_sur_feas,p_sur_feas,h_sur_feas,~] = get_updated_estimates(hist(:,2:end),x_feas,nprob);
        f_hist = [f_hist; [bbe, x_feas', f_feas, c_feas, p_feas, h_feas, f_sur_feas, c_sur_feas, p_sur_feas, h_sur_feas] ];
    else
        [f_feas,c_feas,p_feas,h_feas,~,f_sur_feas,c_sur_feas,p_sur_feas,h_sur_feas,~] = get_updated_estimates(hist(:,2:end),x_infeas,nprob);
        f_hist = [f_hist; [bbe, x_infeas', f_feas, c_feas, p_feas, h_feas, f_sur_feas, c_sur_feas, p_sur_feas, h_sur_feas] ];
    end

    f_hist_file = fopen([folder,'/f_hist_STOMADS' num2str(nprob) '.txt'], 'a');
    format = [repmat('%12.20f,',1, ceal{3} + 2*ceal{4} + 7) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(f_hist_file, format, f_hist(end,:)');
    fclose(f_hist_file);

    % store infeasible improvement history
    [f_infeas,c_infeas,p_infeas,h_infeas,u_infeas,f_sur_infeas,c_sur_infeas,p_sur_infeas,h_sur_infeas,u_sur_infeas] = get_updated_estimates(hist(:,2:end),x_infeas,nprob);
    i_hist = [i_hist; [bbe, x_infeas', f_infeas, c_infeas, p_infeas, h_infeas, u_infeas, f_sur_infeas, c_sur_infeas, p_sur_infeas, h_sur_infeas, u_sur_infeas] ];

    i_hist_file = fopen([folder,'/i_hist_STOMADS' num2str(nprob) '.txt'], 'a');
    format = [repmat('%12.20f,',1, ceal{3} + 2*ceal{4} + 9) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(i_hist_file, format, i_hist(end,:)');
    fclose(i_hist_file);
end
        