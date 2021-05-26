function [f] = Blackbox_call_CovidSim(d,bbe,e,sur)

    if (nargin==2)
        e=1;
        sur=false;
    elseif (nargin==3)
        sur=false;
    end
    
    % [folder_exe, ~, ~] = fileparts(which(mfilename)); % get blackbox_call folder
    folder_exe = "";
    suppress = ">nul 2>&1"; 
    % suppress = ""; % uncomment to view console output of CovidSim
    %% Simulation Paramters
    % Model variables
    % bounds = [1.0	, 0.0   ;... % complaince rate (inversely propotional to number of essential workers)
    %           0.99	, 0.05  ;... % Contact rate given Social distancing (inversely propotional to Social distancing factor)
    %           0.1   , 0.9   ];   % Testing capacity
  
    bounds = [1.0	, 0.0   ;... % complaince rate (inversely propotional to number of essential workers)
              1     , 720  ;... % Social distancing duration (0 - 720 days)
              0.1   , 0.9   ];   % Testing capacity
          
    lob = bounds(:,1)'; upb = bounds(:,2)';
    
    %% Scale variables
    d = scaling(d, lob, upb, 2); % Normalize variables for optimization (StoMADS, GA)
    % if optimizing duration then round to nearest integer
    d(2) = round(d(2));
    
    % d = scaling(d', lob, upb, 2); % Normalize variables for optimization (NOMAD)
    
    %% Input variables
    variable_str = sprintf("/CLP1:%.4f /CLP2:%.4f /CLP3:%.4f" ,d(1),d(2),d(3));

    %% Input parameters
    pop_surrogate = 1000;
    healthcare_capacity = 0.09;
    r = 3.0;
    rs = r/2;
    threads = 8;
    
    working_directory = pwd;
    % country = "United_Kingdom";
    country = "Canada";
    root = "PC7_CI_HQ_SD";
    root = "PC7_CI_HQ_SD_duration";
    outdir = working_directory+"\covid_sim\output_files\";
    paramdir = working_directory+"\covid_sim\param_files\";
    networkdir = working_directory+"\covid_sim\network_files\";
    admindir = working_directory+"\covid_sim\admin_units\";
    
    switch country 
        case "Canada"
            wpop_file_root = "usacan";
            pp_file = paramdir+"preUK_R0=2.0.txt";
            pop = 36460098;
        case "usa_territories"
            wpop_file_root = "us_terr";
            pp_file = paramdir+"preUS_R0=2.0.txt";
            pop = 1;
        case "nigeria"
            wpop_file_root = "nga_adm1";
            pp_file = paramdir+"preNGA_R0=2.0.txt";
            pop = 1;
    	case "United_Kingdom"
            wpop_file_root = "eur";
            pp_file = paramdir+"preUK_R0=2.0.txt";
            pop = 66777534;
    end
    
    cf = paramdir+"p_"+root+".txt";
    out_file = outdir+num2str(e)+"-"+country+"_"+root+"_R0="+num2str(r,"%4.1f");
    wpop_file = networkdir+"wpop_"+wpop_file_root+".txt";
    wpop_bin = networkdir+country+"_pop_density.bin";
    network_bin = networkdir+"Network_"+country+"_T"+num2str(threads)+"_R"+num2str(r,"%4.1f")+".bin";
    admin_file = admindir+country+"_admin.txt";

    
    %random_seed_1 = [randi([1e6,9e6]),randi([1e6,9e6])];
    random_seed_1 = [5218636, 2181088]; % Setup seed for network (keep fixed)
    random_seed_2 = [randi([1e6,9e6]),randi([1e6,9e6])];
    
    % First time setup
    if bbe == 1                                                             % DEBUG: Suppress
        
        try_remove(network_bin);
        try_remove(wpop_bin);
    
        network_str =   "/D:" + wpop_file+" "+... % Input (this time text) pop density
                        "/M:" + wpop_bin+" "+... % Where to save binary pop density
                        "/S:" + network_bin; % Where to save binary net setup
    else
        network_str =   "/D:"   + wpop_bin+" "+... % Binary pop density file (speedup)
                        "/L:"   + network_bin; % Network to load
    end

    parameter_str =     "/PP:"  + pp_file+" "+...
                        "/P:"   + cf+" "+...
                        variable_str+" "+...
                        "/O:"   + out_file+" "+...
                        network_str+" "+... % Network parameters
                        "/NR:"  + num2str(1)+" "+...
                        "/R:"   + num2str(rs)+" "+...
                        num2str(random_seed_1(1))+" "+... % These four numbers are RNG seeds
                        num2str(random_seed_1(2))+" "+...
                        num2str(random_seed_2(1))+" "+...
                        num2str(random_seed_2(2));
                    
    %% Delete output files
    
    % True model outputs
    out_full_filename = out_file + ".avNE.severity.xls";
    try_remove(out_full_filename);                                          % DEBUG: Suppress
    
    % Surrogate model outputs
    output_SAO_filename = sprintf("%i-matlab_out_Blackbox.log", e);
    out_SAO_filename = [folder_exe,"/data/",output_SAO_filename];
    
    % Error logging file
    err_filename = "err_out_Blackbox_CovidSim.log";

    %% Run Blackbox
    
    exe = "CovidSim.exe";
    
    cmd =   exe+" "+...
            "/c:"   + num2str(threads)+" "+...
            "/A:"   + admin_file+" "+...
            parameter_str+" "+...
            suppress;
        
    command = "cd " + folder_exe + suppress + " & " + cmd;
    
    %%%%%%%%%%%%%%%%%%%%%
    if ~(sur)
        %%%%%%%%%%%%%%%%%%%%%
        % Real model
        %%%%%%%%%%%%%%%%%%%%%
        status = system(command);                                           % DEBUG: Suppress
        
        % % DEBUG ONLY
        % % override success flag every fourth call (for debugging only)
        % if mod(bbe,4) == 0
        %     status = 1;
        % else
        %     status = 0;
        % end
        
        if exist(out_full_filename, "file") == 2
            out_exist = 1;
        else
            out_exist = 0;
        end
        
        % % DEBUG ONLY
        % % override output found flag every fourth call (for debugging only)
        % if mod(bbe,4) == 0
        %     out_exist = 0;
        % end
        
        %%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%
        % SAO only
        %%%%%%%%%%%%%%%%%%%%%
        if exist(out_SAO_filename, "file") == 2
            out_exist = 1;
        else
            out_exist = 0;
        end
        status = 0;
        fprintf("WARNING: Surrogate call made\n")
        %%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%
    if out_exist == 1 % REMOVE CSTR FOR BB OPT
        %% Obtain output
        if ~(sur)
            %%%%%%%%%%%%%%%%%%%%%
            % Real model Only
            %%%%%%%%%%%%%%%%%%%%%
            T = readmatrix(out_full_filename,"FileType","text");
            max_I = max(T(:,7)); % retrieve maximum number of infections
            f = (max_I/pop) - healthcare_capacity;
            fclose("all");
            %%%%%%%%%%%%%%%%%%%%%
        else
            %%%%%%%%%%%%%%%%%%%%%
            % SAO only (Read COVID_SIM_UI result instead)
            %%%%%%%%%%%%%%%%%%%%%
            fileID_out = fopen(out_SAO_filename,"r");
            f = textscan(fileID_out,"%f %f %f", "Delimiter", ",");
            f = cell2mat(f);
            f = (f(3)/pop_surrogate); % already has healthcare_capacity = 90 subtracted (inside cpp program)
            fclose(fileID_out);
            fclose("all");
            %%%%%%%%%%%%%%%%%%%%%
        end
    end
    
    if status ~= 0 && out_exist == 1
        %% Failed execution only
        fprintf("command: %s\nout_exist: %i\nstatus: %i\n",command,out_exist,status)
        fileID_err = fopen("./"+err_filename,"at");
        Net_results = sprintf("%f," , d);
        
        fprintf(fileID_err, "%i,%s%s,%s,%i,failed_exe",bbe,Net_results,command,out_full_filename,status);
        fprintf(fileID_err,"\n");
        fclose(fileID_err);

        msg = "Warning: Failed executation";
        warning(msg)
    end
    
    if out_exist == 0
        %% Error execution
        fprintf("command: %s\nout_exist: %i\nstatus: %i\n",command,out_exist,status)
        fileID_err = fopen("./"+err_filename,"at");
        Net_results = sprintf("%f," , d);
        
        fprintf(fileID_err, "%i,%s%s,%s,%i,no_output", bbe,Net_results,command,out_full_filename,status);
        fprintf(fileID_err,"\n");
        fclose(fileID_err);

        msg = "Warning: No blackbox output found";
        
        f = NaN;
        warning(msg)
    end
end

%% Inline functions
function try_remove(filename)
    % Delete files
    if exist(filename, "file") == 2
      delete(filename)
    end
end