clearvars
format compact
clc
diary off

%=========================================================%
%                          MAIN                           %
%=========================================================%

%==========================================================
% input arguments   

% StoMADS_settings
n_k                     = 1;
epsilon_f               = 0.2;
Delta                   = 1.0;
gamma                   = 5.0;
tau                     = 0.5;
OPPORTUNISTIC_EVAL      = true;
MAX_BB_EVAL             = 10000;
nb_proc                 = 8;
n_k_success             = 100;

nprob = 34; % Problem using COVID_SIM_UI for both objective and constraint
% nprob = 35; % Problem including CovidSim as a constraint

StoMADS_settings = { n_k, epsilon_f, Delta, gamma, tau, OPPORTUNISTIC_EVAL ,...
    MAX_BB_EVAL, nb_proc, n_k_success };
StoMADS_settings_text = { 'n_k', 'epsilon_f', 'Delta', 'gamma', 'tau', 'OPPORTUNISTIC_EVAL' ,...
    'MAX_BB_EVAL', 'nb_proc', 'n_k_success' };

n_runs = 3;

folder = 'StoMADS_exp';
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
    for k = 1:1:length(StoMADS_settings)
        formatting = [StoMADS_settings_text{k}, ' : %s\n'];
        fprintf(file, formatting, mat2str(StoMADS_settings{k})); 
    end
    fclose(file);
        
    diary(diary_file) % save console output to window
    diary on
    StoMADS_call(nprob,n_k,epsilon_f,Delta,gamma,tau,MAX_BB_EVAL,nb_proc,n_k_success,run_folder)
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