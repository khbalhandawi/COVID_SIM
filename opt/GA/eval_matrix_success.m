function eval_matrix_success(x, eval_k, ceal, re_evaluate)

if (nargin==3)
    re_evaluate=true; % reevaluate by default
end

global hist f_progress;
z = zeros(eval_k, ceal{4}+1);
bbe = size(hist,1); 
folder = ceal{19};
    
%% Display success
% if isempty(hist) == 0
%     [fsk,csk,psk,hsk,usk] = get_updated_estimates(hist(:,2:end), x, ceal);
%     format_bbe = '          %4d      (         ';
%     format_x = [repmat('%12.10f ',1, length(x))]; format_x(end)='';
%     format_obj = '  )     %12.10f';
%     format = [format_bbe format_x format_obj '\n'];
%     fprintf(format, [bbe, x', fsk])
% end
%% Evaluate blackbox eval_k times
if re_evaluate

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

    nb_proc = ceal{17};

    if nb_proc ~= 1
        parfor (e = 1:eval_k, nb_proc)
            z(e,:) = covid_V2_one(x, bbe+e-1, e); % TODO
        end
    else
        for e = 1:eval_k
            z(e,:) = covid_V2_one(x, bbe+e-1); % TODO
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
        P_matrix = [P_matrix; x z(i,:)];
    end
    [fsk,csk,psk,hsk,usk] = get_updated_estimates(P_matrix, x, ceal); % just get the average essentially
else
    % use last known result instead
    data = f_progress(end,length(x) + 3:end);
    fsk = data(1); csk = data(2); psk = data(3); hsk = data(4);
end

n_success_f = size(f_progress,1) + 1; 

% Compute running average of raw history
f_progress = [f_progress; [n_success_f, bbe, x, fsk, csk, psk, hsk] ];

hist_file = fopen([folder,'/f_progress_GA.txt'], 'a');
format = [repmat('%12.20f,',1, ceal{3} + ceal{4} + 5) '\n']; format(end-2) = ''; % remove last delimliter
fprintf(hist_file, format, f_progress(end,:)');
fclose(hist_file); 
        