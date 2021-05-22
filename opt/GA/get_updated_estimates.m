function [fsk,csk,psk,hsk,usk] = get_updated_estimates(history,x,ceal)
% Retrieve updated estimates from history

SIdx = find(ismember(history(:,1:ceal{3}), x,'rows'));
P_matrix = history(SIdx, (ceal{3} + 1):end);
P_mean = mean(P_matrix, 1);            

fsk = P_mean(1);
csk = P_mean(2:end);
psk = sum(P_matrix(:,2) <= 0) / length(P_matrix(:,2)); % reliability
hsk = sum(max(P_mean(2:end), 0));

% compute U_max (not applicable for GAs)
usk=0;

end

