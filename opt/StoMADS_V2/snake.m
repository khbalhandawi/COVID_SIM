function z = snake(x)


% objective function
z(1) = sqrt((x(1) - 20)^2 + (x(2) - 1)^2);

% constraints
z(2) = sin(x(1)) - 1/10 - x(2) ;
z(3) = x(2) - sin(x(1)) ;

end