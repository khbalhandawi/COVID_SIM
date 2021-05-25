function eval_matrix_success(x, eval_k, nprob, success_type, re_evaluate)

if (nargin==4)
    re_evaluate=true; % reevaluate by default
end

global hist f_progress i_progress ceal;
size_empty = max(1,eval_k); % in case eval_k == 0 

z = zeros(size_empty, ceal{4}+1);
z_sur = zeros(size_empty, ceal{4}+1);
bbe = size(hist,1); 
folder = ceal{19};
    
%% Display success
if isempty(hist) == 0 && success_type == 1
    [fsk,csk,psk,hsk,~,~,~,~,~,~] = get_updated_estimates(hist(:,2:end),x,nprob);
    format_bbe = '          %4d      (         ';
    format_x = [repmat('%12.10f ',1, length(x))]; format_x(end)='';
    format_obj = '  )     %12.10f';
    format = [format_bbe format_x format_obj '\n'];
    fprintf(format, [bbe, x', fsk])
end
%% Evaluate blackbox eval_k times
if re_evaluate

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
    
end

%% Logging and update blackbox outputs
% Update success history cache

if re_evaluate
    P_matrix = [];
    % loop over results
    for i = 1:size(z,1)
        % Update evaluation matrix
        P_matrix = [P_matrix; x' z(i,:) z_sur(i,:)];
    end
    [fsk,csk,psk,hsk,~,~,~,~,~,~] = get_updated_estimates(P_matrix,x,nprob);
else
    % use last known result instead
    if success_type == 1
        data = f_progress(end,length(x) + 3:end);
        fsk = data(1); csk = data(2); psk = data(3); hsk = data(4);
    elseif success_type == 2
        data = i_progress(end,length(x) + 3:end);
        fsk = data(1); csk = data(2); psk = data(3); hsk = data(4);
    end
end

n_success_f = size(f_progress,1) + 1; 
n_success_i = size(i_progress,1) + 1; 

if success_type == 1
    % Compute running average of raw history
    f_progress = [f_progress; [n_success_f, bbe, x', fsk, csk, psk, hsk] ];

    hist_file = fopen([folder,'/f_progress_STOMADS' num2str(nprob) '.txt'], 'a');
    format = [repmat('%12.20f,',1, length(x) + ceal{4} + 5) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(hist_file, format, f_progress(end,:)');
    fclose(hist_file); 
elseif success_type == 2
    % Compute running average of raw history
    i_progress = [i_progress; [n_success_i, bbe, x', fsk, csk, psk, hsk] ];

    hist_file = fopen([folder,'/i_progress_STOMADS' num2str(nprob) '.txt'], 'a');
    format = [repmat('%12.20f,',1, length(x) + ceal{4} + 5) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(hist_file, format, i_progress(end,:)');
    fclose(hist_file); 
end
        