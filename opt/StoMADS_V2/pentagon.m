function z = pentagon(x)



% objective function

f1_pent = -sqrt((x(1) - x(3))^2 + (x(2) - x(4))^2);
f2_pent = -sqrt((x(3) - x(5))^2 + (x(4) - x(6))^2);
f3_pent = -sqrt((x(5) - x(1))^2 + (x(6) - x(2))^2);
        
z(1) = max([f1_pent, f2_pent, f3_pent]);

% constraints
z(2) = x(1) * cos(2*pi*0/5) + x(2) * sin(2*pi*0/5) - 1 ;
z(3) = x(1) * cos(2*pi*1/5) + x(2) * sin(2*pi*1/5) - 1 ;
z(4) = x(1) * cos(2*pi*2/5) + x(2) * sin(2*pi*2/5) - 1 ;
z(5) = x(1) * cos(2*pi*3/5) + x(2) * sin(2*pi*3/5) - 1 ;
z(6) = x(1) * cos(2*pi*4/5) + x(2) * sin(2*pi*4/5) - 1 ;
z(7) = x(3) * cos(2*pi*0/5) + x(4) * sin(2*pi*0/5) - 1 ;
z(8) = x(3) * cos(2*pi*1/5) + x(4) * sin(2*pi*1/5) - 1 ;
z(9) = x(3) * cos(2*pi*2/5) + x(4) * sin(2*pi*2/5) - 1 ;
z(10) = x(3) * cos(2*pi*3/5) + x(4) * sin(2*pi*3/5) - 1 ;
z(11) = x(3) * cos(2*pi*4/5) + x(4) * sin(2*pi*4/5) - 1 ;
z(12) = x(5) * cos(2*pi*0/5) + x(6) * sin(2*pi*0/5) - 1 ;
z(13) = x(5) * cos(2*pi*1/5) + x(6) * sin(2*pi*1/5) - 1 ;
z(14) = x(5) * cos(2*pi*2/5) + x(6) * sin(2*pi*2/5) - 1 ;
z(15) = x(5) * cos(2*pi*3/5) + x(6) * sin(2*pi*3/5) - 1 ;
z(16) = x(5) * cos(2*pi*4/5) + x(6) * sin(2*pi*4/5) - 1 ;





end