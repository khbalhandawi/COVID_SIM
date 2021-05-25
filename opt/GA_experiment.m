clearvars
format compact
clc
close('all')
fclose('all');
diary off

%=========================================================%
%                          MAIN                           %
%=========================================================%

%==========================================================
% input arguments   

% GA_settings
n_k                     = 4;
MAX_BB_EVAL             = 10000;
nb_proc                 = 8;
n_k_success             = 100;
n_population            = 50;
R_initial               = 'auto'; % can be 'auto'
R_factor                = 100;
max_stall_G             = 50;
config                  = 'default';

GA_settings = { n_k, MAX_BB_EVAL, nb_proc, n_k_success, n_population, ...
    R_initial, R_factor, max_stall_G, config };
GA_settings_text = { 'n_k', 'MAX_BB_EVAL', 'nb_proc', 'n_k_success', ...
    'n_population', 'R_initial', 'R_factor', 'max_stall_G', 'config'};

n_runs = 4;

folder = 'GA_exp';
machine = 'WORKSTATION';
check_folder(folder)

for i = 1:1:n_runs

    fprintf('\n+=======================================================+\n')
    fprintf('|                    EXPERIMENT %04d                    |\n',i)
    fprintf('+=======================================================+\n')

    run_folder = ['./',folder,'/Run_',num2str(i)];
    check_folder(run_folder)
    clear_folder(run_folder)

    settings_file = [run_folder,'/','settings.txt'];
    machine_file = [run_folder,'/',machine];
    diary_file = [run_folder,'/','output.txt'];
    
    % Print machine name
    file = fopen(machine_file, 'w');
    fclose(file);
    
    % Print StoMADS settings to file
    file = fopen(settings_file, 'w');
    for k = 1:1:length(GA_settings)
        formatting = [GA_settings_text{k}, ' : %s\n'];
        fprintf(file, formatting, mat2str(GA_settings{k})); 
    end
    fclose(file);
        
    diary(diary_file) % save console output to window
    diary on
    GA_call(n_k,MAX_BB_EVAL,nb_proc,n_k_success,n_population,R_initial,...
        R_factor,max_stall_G,config,run_folder)
    diary off

end

%=========================================================%
%                     Clear directory                     %
%=========================================================%
function clear_folder(folder)
    % clear all files inside folder %
    delete([folder,'\*'])
end
            
%=========================================================%
%                 Create empty directory                  %
%=========================================================%
function check_folder(folder)
	% check if folder exists, make if not present %
    if not(isfolder(folder))
        mkdir(folder)
    end
end