%=========================================================%
%             Initial population statistics               %
%=========================================================%
function [R,population] = Initialize_pop(n_population,n_vars,LB,UB,ceal)
    % Initiliazes population of first generation %

    population = lhsdesign(n_population,n_vars,'criterion','maximin');
    population = scaling(population,LB,UB,2);
    
    % Evaluate population
    for i = 1:1:n_population
        individual = population(i,:);
        [c(i,:),~] = function_cstr(individual,ceal);
        f(i) = function_obj(individual,ceal);
    end
    %----------------------------------------------------%
    % Compute initial penalty

    CV = sum(max(c,0),2); % sum of constraint functions along columns
    R = sum(abs(f))/sum(CV);
    
end