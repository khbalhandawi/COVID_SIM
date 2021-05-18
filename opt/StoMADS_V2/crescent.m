function z = crescent(x)

% Objective function
z(1) = x(10);

% Constraints

z(2) = sum((x-1).^2)- 10^2 ;
z(3) = 10^2  - (sum((x+1).^2));

end