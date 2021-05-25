function [fsk,csk,psk,hsk,usk,fsk_sur,csk_sur,psk_sur,hsk_sur,usk_sur] = get_updated_estimates(history,x,nprob,epsilon,Lambda)
% Retrieve updated estimates from history

global ceal

SIdx = find(ismember(history(:,1:ceal{3}), x','rows'));
P_matrix = history(SIdx, (ceal{3} + 1):end);
P_mean = nanmean(P_matrix, 1); % Triple check : in case CovidSim crashed simply ignore result and continue         

fsk = P_mean(1);
csk = P_mean(2:2+ceal{4}-1);
psk = sum(P_matrix(:,2) <= 0) / length(P_matrix(:,2)); % reliability (for first constraint only)
hsk = sum(max(csk, 0));

fsk_sur = P_mean(2+ceal{4});
csk_sur = P_mean(2+ceal{4}+1:2+ceal{4}+1+ceal{4}-1);
psk_sur = sum(P_matrix(:,2+ceal{4}+1) <= 0) / length(P_matrix(:,2+ceal{4}+1)); % reliability (for first constraint only)
hsk_sur = sum(max(csk_sur, 0));

% compute U_max
if (nargin==3)
    usk=0;
    usk_sur = 0;
elseif (nargin==5)
    usk = sum(max(csk + epsilon*Lambda^2, 0));
    usk_sur = sum(max(csk_sur + epsilon*Lambda^2, 0));
end

end

