function z = eval_matrix(x, eval_k, x_feas, ceal)

global hist hist_avg f_hist;
z = zeros(eval_k, ceal{4}+1);
bbe = size(hist,1)+1; 
folder = ceal{19};
    
%% Evaluate blackbox eval_k times

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

%% Logging and update blackbox outputs
% Update history cache (ON by default)

% loop over results
for i = 1:size(z,1)

    % Update raw history
    bbe = size(hist,1)+1;   
    hist = [hist; bbe x z(i,:)];
end

% loop over results
for i = 1:size(z,1)

    bbe = hist(end,1) - eval_k + i;

    % Compute running average of raw history
    
    [fsk,csk,psk,hsk,~] = get_updated_estimates(hist(:,2:end), x, ceal);
    hist_avg = [hist_avg; [bbe, x, fsk, csk, psk, hsk] ];

    hist_file = fopen([folder,'/GA_hist.txt'], 'a');
    format = [repmat('%12.20f,',1, ceal{3} + ceal{4} + 4) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(hist_file, format, hist_avg(end,:)');
    fclose(hist_file); 

    % store feasible improvement history
    if isempty(x_feas) == 0
        [f_feas,c_feas,p_feas,h_feas,~] = get_updated_estimates(hist(:,2:end), x_feas, ceal);
        f_hist = [f_hist; [bbe, x_feas, f_feas, c_feas, p_feas, h_feas] ];
    
        f_hist_file = fopen([folder,'/f_hist_GA.txt'], 'a');
        format = [repmat('%12.20f,',1, ceal{3} + ceal{4} + 4) '\n']; format(end-2) = ''; % remove last delimliter
        fprintf(f_hist_file, format, f_hist(end,:)');
        fclose(f_hist_file);
    end

end
        