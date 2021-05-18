function [z] = v_deux_black_box(x, nprob, bbe, e)

format long g


global bb_history ceal

if (nargin==3)
    e=1; % sample rate index for objective and constraints
end

if iscolumn(x')
    x=x';
end

%------------------------------------------------------------------------------------------------------------------
ka = 3*0.001;

f_min = [-9, -17.320505062987, 0.0809767250669388, -1.85961866517433, -0.389659516097185,...
           -0.330357142857196, 24.7463262223885, -0.738032947883234, -0.742347776503948, -0.333177938041652,...
           9214.83613116, 14.4089265775702, 306.499999999887, 1, -22.6274169970728,...
           -0.671826579984121, -44.0000000000003, 680.653616867073, -15.0000000000003, 0.126800335858373,...
           -1.66101448991706, 13.590841691859, 26340.5139373854, 7202.85145097776, -7950.96189432334,...
           2.00000000000001, -0.0958250414180358, 1.58707751214324, 10855.2848139146, 2994.47106614609,...
           0.0126715348232979, 2.14305947964041, -0.748308310898478, 3.05421843912888, -0.971104067282413,...
           -1.04739389109279, -3.35985630839095, -0.688383458323536, -15.3327693358917, -3.93429519399679,...
           3.29873508584955, 0.435688086692635];

%------------------------------------------------------------------------------------------------------------------


switch nprob
    
    

    
    case 1  % crescent
        u = crescent(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end        
        
    case 2 % disk
        u = disk(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));     
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end            

    case 3  % snake
        u = snake(x);
        z = u;% + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           %(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end      

    case 4  % pentagon
        u = pentagon(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end
        
    case 5  % mad1 --> mad_un
        u = mad_un(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 6  % mad2 --> mad_deux
        u = mad_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 7  % wong_deux
        u = wong_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 8  % G2_10 --> G_deux_dix
        u = G_deux_dix(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 9  % G2_20 --> G_deux_vingt
        u = G_deux_vingt(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 10  % G2_3 --> G_deux_trois
        u = G_deux_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

      case 11  % chenwang_f_deux
        u = chenwang_f_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 12 %chenwang_f3 --> chenwang_f_trois
        u = chenwang_f_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 13 % hs_quinze
        u = hs_quinze(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 14 % hs_vingt_deux
        u = hs_vingt_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 15 % hs_vingt_neuf
        u = hs_vingt_neuf(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 16 % hs_cent_huit
        u = hs_cent_huit(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 17 % hs_quarante_trois
        u = hs_quarante_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 18 % taowang_f_deux
        u = taowang_f_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        
    case 19 % zhaowang_f_cinq
        u = zhaowang_f_cinq(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 20 % spring
        u = spring(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 21 % barnes
        u = barnes(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 22 % taowang_f_un
        u = taowang_f_un(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 23 % hs_cent_quatorze
        u = hs_cent_quatorze(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 24  % dembo
        u = dembo(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        
     case 25  % hs_dix_neuf
        u = hs_dix_neuf(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
      
     case 26  % hs_vingt_trois
        u = hs_vingt_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 27   % mezmontes
        u = mezmontes(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 28   % welded_beam
        u = welded_beam(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

     case 29   % pressure_vessel
        u = pressure_vessel(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          

    case 30  % Khalil
        z = TRS_BB_simple(x')'; 

    case 31  % Khalil; covid_one: first objective subject to the constraint
        z = covid_one(x);

    case 32  % Khalil; covid_two: second objective subject to the constraint
        z = covid_two(x);
        
    case 33  % Khalil; covid_three: first objective no constraint
        z = [covid_three(x); -10];
        
    case 34  % Khalil; covid_V2_one: first objective subject to the constraint
        z = covid_V2_one(x,bbe,e);
        
    case 35  % Khalil; HPO: first objective subject no constraint
        z = HPO_one(x);
        
end