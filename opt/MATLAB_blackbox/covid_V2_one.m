function z = covid_V2_one(x,bbe,e)
if (nargin==2)
    e=1; % sample rate index for objective and constraints
end
if iscolumn(x)
    x = x';
end
u = Blackbox_call(x,bbe,e);
z(1) = u(1);
z(2) = u(3);
if isrow(z)
    z = z';
end
end