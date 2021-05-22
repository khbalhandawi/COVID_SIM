function [z] = v_deux_black_box(x, nprob, bbe, e)

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
        
end