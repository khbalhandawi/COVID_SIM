function z = Poll(delta, Delta, v)
format long g
nn = length(v); 
In = eye(nn, nn);
H = In - 2 * (v / norm(v)) * (v' / norm(v));
B = zeros(nn,nn);
    for i = 1:nn
        B(:,i) = round((Delta/delta) * (H(:,i) / max(abs(H(:,i)))));
    end
    z = [B -B];
end 