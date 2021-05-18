function [fsk,csk,psk,hsk,usk] = get_updated_estimates(history,x,nprob,epsilon,Lambda)
% Retrieve updated estimates from history

global ceal
SIdx = find(ismember(history(:,1:ceal{3}), x','rows'));
P_matrix = history(SIdx, (ceal{3} + 1):end);
P_mean = mean(P_matrix, 1);            

fsk = P_mean(1);
csk = P_mean(2:end);
psk = sum(P_matrix(:,2) <= 0) / length(P_matrix(:,2)); % reliability
hsk = sum(max(P_mean(2:end), 0));

% compute U_max
if (nargin==3)
    usk=0;
elseif (nargin==5)
    usk = sum(max(P_mean(2:end) + epsilon*Lambda^2, 0));
end

end

