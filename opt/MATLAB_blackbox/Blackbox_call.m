function [f] = Blackbox_call(d,bbe,e,sur)

    if (nargin==2)
        e=1;
        sur=false;
    elseif (nargin==3)
        sur=false;
    end
    
    % [folder_exe, ~, ~] = fileparts(which(mfilename)); % get blackbox_call folder
    folder_exe = '.';
    %% Simulation Paramters
    % Model variables
    bounds = [16      , 101   ;... % number of essential workers
              0.0001  , 0.15  ;... % Social distancing factor
              10      , 51    ];   % Testing capacity

    lob = bounds(:,1)'; upb = bounds(:,2)';
    
    healthcare_capacity = 90;
    
    %% Scale variables
    d = scaling(d, lob, upb, 2); % Normalize variables for optimization (StoMADS)
    % d = scaling(d', lob, upb, 2); % Normalize variables for optimization (NOMAD)
    d(1) = round(d(1)); d(3) = round(d(3));
    
    %% Input variables
    Net_results = sprintf('%.4f ' , d);
    Net_results = Net_results(1:end-1);% strip final comma
    variable_str = [num2str(bbe),' ',num2str(e),' ',Net_results];

    %% Input parameters
    Net_results = sprintf('%0.2f ' , [healthcare_capacity]);
    Net_results = Net_results(1:end-1);% strip final comma
    parameter_str = Net_results;
    % parameter_str = '';
    
    %% Delete output files
    output_filename = sprintf('%i-matlab_out_Blackbox.log', e);
    out_full_filename = [folder_exe,'/data/',output_filename];
    if exist(out_full_filename, 'file') == 2
      delete(out_full_filename)
    end

    err_filename = sprintf('%i-err_out_Blackbox_%i.log', e,bbe);
    
    %% Run Blackbox
    command = ['cd ', folder_exe, ' & COVID_SIM_UI ',variable_str, ' ', parameter_str];
    
    %%%%%%%%%%%%%%%%%%%%%
    if ~(sur)
        %%%%%%%%%%%%%%%%%%%%%
        % Real model
        %%%%%%%%%%%%%%%%%%%%%
        status = system(command);
        if exist(out_full_filename, 'file') == 2
            out_exist = 1;
        else
            out_exist = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%
        % SAO only
        %%%%%%%%%%%%%%%%%%%%%
        out_exist = 1;
        status = 0;
        fprintf('no surrogate provided\n')
        %%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%
    if status == 0 & out_exist == 1 % REMOVE CSTR FOR BB OPT
        %% Obtain output
        if ~(sur)
            %%%%%%%%%%%%%%%%%%%%%
            % Real model Only
            %%%%%%%%%%%%%%%%%%%%%
            fileID_out = fopen(out_full_filename,'r');
            f = textscan(fileID_out,'%f %f %f', 'Delimiter', ',');
            f = cell2mat(f);
            % f = [f(1), f(3)]; % for NOMAD
            fclose(fileID_out);
            fclose('all');
            %%%%%%%%%%%%%%%%%%%%%
        end
    elseif status == 1 | out_exist == 0
        %% Error execution
        
        fileID_err = fopen([folder_exe,'/data/',err_filename],'at');
        Net_results = sprintf('%f,' , d);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_err, '%i,%s', [bbe,Net_results]);
        fprintf(fileID_err,'\n');
        fclose('all');

        msg = 'Error: Invalid point';
        
        f = [NaN, NaN, NaN];
        % f = [NaN, NaN]; % for NOMAD
        warning(msg)
    end
end