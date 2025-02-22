%% Preamble
clearvars
close all
clc
format compact

addpath GA
addpath MATLAB_blackbox
addpath MATLAB_blackbox\support_functions
% rng default % For reproducibility (only works if problem is deterministic)

%% Problem definition
n_k                     = 4;
MAX_BB_EVAL             = 10000;
nb_proc                 = 8;
n_k_success             = 20;
n_population            = 30;
R_initial               = 'auto'; % can be 'auto'
R_factor                = 100;
max_stall_G             = 50;
config = 'default';

folder = 'GA_exp';
machine = 'WORKSTATION';
check_folder(folder)

run_folder = ['./',folder,'/Run_1'];
check_folder(run_folder)
clear_folder(run_folder)

ceal = v_deux_Sto_Param(2,n_k,0,0,0,0,MAX_BB_EVAL,nb_proc,n_k_success,run_folder);

%% clear history

folder = ceal{19}; % Careful here (redefinition)

hist_file_name = [folder,'/GA_hist.txt'];  % Check if hist_file_name already exists in the directory
if isfile(hist_file_name)
    delete(hist_file_name);
end
hist_file = fopen(hist_file_name, 'w');
fprintf(hist_file, 'bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(hist_file); 

f_hist_file_name = [folder,'/f_hist_GA.txt'];  % Check if f_hist_file_name already exists in the directory
if isfile(f_hist_file_name)
    delete(f_hist_file_name);
end
f_hist_file = fopen(f_hist_file_name, 'w');
fprintf(f_hist_file, 'bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(f_hist_file); 

f_hist_file_name = [folder,'/f_progress_GA.txt'];  % Check if f_hist_file_name already exists in the directory
if isfile(f_hist_file_name)
    delete(f_hist_file_name);
end
f_hist_file = fopen(f_hist_file_name, 'w');
fprintf(f_hist_file, 'n_success,bbe,x1,x2,x3,f,cstr,p_value,h\n');
fclose(f_hist_file); 

G_hist_file_name = [folder,'/G_stats_GA.txt'];  % Check if i_hist_file_name already exists in the directory
if isfile(G_hist_file_name)
    delete(G_hist_file_name);
end
G_hist_file = fopen(G_hist_file_name, 'w');
fprintf(G_hist_file, 'bbe,f_best,mean_f,min_f,max_f,mean_d,min_d,max_d\n');
fclose(G_hist_file); 

%% Set up augmented Lagrangian GA optimization
global cumilative_f_evals GA_hist x_feas
global hist hist_avg f_hist f_progress;
global max_generations_per_it max_stall_generations_per_it min_tol_per_it
global fig

cumilative_f_evals = 0; GA_hist = []; x_feas = [];
hist = []; hist_avg = []; f_hist = []; f_progress = [];

max_generations_per_it = 300; max_stall_generations_per_it = 10; min_tol_per_it = 1e-3;
nvars = ceal{3}; % Number of variables

%% (optional) Initialize population and compute initial penalty factor
LB = ceal{7}';
UB = ceal{8}';
[R,initial_population] = Initialize_pop(n_population,nvars,LB,UB,ceal);

if strcmp(R_initial,'auto')
    % R = max(R*50,1); % must be at least 1
    R = R * 50;
else
    R = R_initial;
end

fprintf('INITIAL PENALTY R = %-012.6f\n',R)

%% GA Options
if strcmp(config,'custom')

    % Non-default options
    options = optimoptions('ga','Display','off',...
        'MaxStallGenerations',max_stall_G,'FunctionTolerance',1e-6,'ConstraintTolerance',0,...
        'NonlinearConstraintAlgorithm','auglag','PopulationSize',n_population,...
        'InitialPopulationMatrix',initial_population,'InitialPenalty',R,...
        'PenaltyFactor',R_factor,...
        'CrossoverFraction',0.1,'MutationFcn',{@mutationadaptfeasible,1,.5},...
        'OutputFcn',@(options,state,flag) callback_function(options,state,flag,ceal));

elseif strcmp(config,'default')

    % Default options
    options = optimoptions('ga','Display','off',...
        'MaxStallGenerations',max_stall_G,...
        'NonlinearConstraintAlgorithm','auglag','PopulationSize',n_population,...
        'InitialPopulationMatrix',initial_population,'InitialPenalty',R,...
        'PenaltyFactor',R_factor,...
        'OutputFcn',@(options,state,flag) callback_function(options,state,flag,ceal));

end

%% Use GA optimization
[x,fval] = ga({@function_obj,ceal},nvars,[],[],[],[],ceal{7}',ceal{8}',{@function_cstr,ceal},options);

% Report final results
fprintf('================================\n')
fprintf('SOLUTION\n')
x_string = repmat('%-012.6f   ',1,ceal{3});
fprintf(['The optimizer is at x = [   ',x_string,']\n'],x)
fprintf('The optimum function value is f = %-012.6f \n',fval)
fprintf('================================\n')

% Save figure of progress to file and close it
set(fig,'color','w');
savefig(fig,[folder,'/progress.fig']);
print(fig,[folder,'/progress.pdf'],'-dpdf','-r300');
close(fig);

%% Inline functions
%=========================================================%
%             Initial population statistics               %
%=========================================================%
function [R,population] = Initialize_pop(n_population,n_vars,LB,UB,ceal)
    % Initiliazes population of first generation %

    population = lhsdesign(n_population,n_vars,'criterion','maximin');
    population = scaling(population,LB,UB,2);
    
    % Evaluate population
    for i = 1:1:n_population
        individual = population(i,:);
        [c(i,:),~] = function_cstr(individual,ceal);
        f(i) = function_obj(individual,ceal);
    end
    %----------------------------------------------------%
    % Compute initial penalty

    CV = sum(max(c,0),2); % sum of constraint functions along columns
    R = sum(abs(f))/sum(CV);
    
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