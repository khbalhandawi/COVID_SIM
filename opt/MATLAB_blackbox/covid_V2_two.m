function [z,z_sur] = covid_V2_two(x,bbe,e)
if (nargin==2)
    e=1; % sample rate index for objective and constraints
end
if iscolumn(x)
    x = x';
end

% COVID_SIM_UI blackbox (objective)
u = Blackbox_call(x,bbe,e);
z(1) = u(1);
z_sur(1) = u(1);
z_sur(2) = u(3)/1000;

% CovidSim blackbox (constraint)
u_covidsim = Blackbox_call_CovidSim(x,bbe,e);       % uncomment to use CovidSim for constraint
z(2) = u_covidsim;                                  % uncomment to use CovidSim for constraint
% z(2) = u(3)/1000;                                 % uncomment to use COVID_SIM_UI for constraint

if isrow(z)
    z = z';
end
end