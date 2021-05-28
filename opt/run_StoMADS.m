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
epsilon_f               = 0.01;
Delta                   = 1.0;
gamma                   = 5.0;
tau                     = 0.5;
OPPORTUNISTIC_EVAL      = true;
MAX_BB_EVAL             = 250;
nb_proc                 = 1;
n_k_success             = 0; % must be greater than 0 to evaluate true objective and constraint values

% nprob = 34; % Problem using COVID_SIM_UI for both objective and constraint
nprob = 35; % Problem including CovidSim as a constraint

StoMADS_settings = { n_k, epsilon_f, Delta, gamma, tau, OPPORTUNISTIC_EVAL ,...
    MAX_BB_EVAL, nb_proc, n_k_success };
StoMADS_settings_text = { 'n_k', 'epsilon_f', 'Delta', 'gamma', 'tau', 'OPPORTUNISTIC_EVAL' ,...
    'MAX_BB_EVAL', 'nb_proc', 'n_k_success' };

folder = 'StoMADS_exp';
machine = 'WORKSTATION';
check_folder(folder)

% Only needed If using CovidSim
error_log = "err_out_Blackbox_CovidSim.log";
CovidSim_output_folder = '.\covid_sim\output_files';
check_folder(CovidSim_output_folder)
clear_folder(CovidSim_output_folder)
try_remove(error_log)

run_folder = ['./',folder,'/Run_',num2str(1)];
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

%% Utility functions
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

%=========================================================%
%           Try to delete a file if it exists             %
%=========================================================%
function try_remove(filename)
    % Delete files
    if exist(filename, "file") == 2
      delete(filename)
    end
end