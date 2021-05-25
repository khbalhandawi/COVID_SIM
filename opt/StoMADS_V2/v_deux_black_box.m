function [z,z_sur] = v_deux_black_box(x, nprob, bbe, e)

format long g

if (nargin==3)
    e=1; % sample rate index for objective and constraints
end

if iscolumn(x')
    x=x';
end

%------------------------------------------------------------------------------------------------------------------


switch nprob
        
    case 34  % Khalil; covid_V2_one: first objective subject to the constraint
        z = covid_V2_one(x,bbe,e);
        z_sur = z;
        
    case 35  % Khalil; covid_V2_two: first objective from COVID_SIM_UI, constraint from CovidSim
        [z,z_sur] = covid_V2_two(x,bbe,e);
        
end