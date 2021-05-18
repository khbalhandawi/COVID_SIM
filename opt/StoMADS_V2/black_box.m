function z = black_box(x, nprob)

format long g


global hist;


ceal = Sto_Param(nprob);
p_it_vec = repmat(1:42,1,1000);

if iscolumn(x')
    x=x';
end
%------------------------------------------------------------------------------------------------------------------
ka = 3*0.001;

prob_min = [-9, -17.320505062987, 0.0809767250669388, -1.85961866517433, -0.389659516097185,...
           -0.330357142857196, 24.7463262223885, -0.738032947883234, -0.742347776503948, -0.333177938041652,...
           9214.83613116, 14.4089265775702, 306.499999999887, 1, -22.6274169970728,...
           -0.671826579984121, -44.0000000000003, 680.653616867073, -15.0000000000003, 0.126800335858373,...
           -1.66101448991706, 13.590841691859, 26340.5139373854, 7202.85145097776, -7950.96189432334,...
           2.00000000000001, -0.0958250414180358, 1.58707751214324, 10855.2848139146, 2994.47106614609,...
           0.0126715348232979, 2.14305947964041, -0.748308310898478, 3.05421843912888, -0.971104067282413,...
           -1.04739389109279, -3.35985630839095, -0.688383458323536, -15.3327693358917, -3.93429519399679,...
           3.29873508584955, 0.435688086692635];
f_min = repmat(prob_min, 1,1000);
%------------------------------------------------------------------------------------------------------------------

switch p_it_vec(nprob)
    
    

    
    case 1  % crescent
        bbe = size(hist,1)+1;  
        u = crescent(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end        
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file); 
        
        
    case 2 % disk
        bbe = size(hist,1)+1;   
        u = disk(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));     
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end            
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
        
    case 3  % snake
        bbe = size(hist,1)+1;   
        u = snake(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];  
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end      
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);   
        
        
    case 4  % pentagon
        bbe = size(hist,1)+1;  
        u = pentagon(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);      
        
        
    case 5  % mad1 --> mad_un
        bbe = size(hist,1)+1;  
        u = mad_un(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
    case 6  % mad2 --> mad_deux
        bbe = size(hist,1)+1; 
        u = mad_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
    case 7  % wong_deux
        bbe = size(hist,1)+1; 
        u = wong_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 8  % G2_10 --> G_deux_dix
        bbe = size(hist,1)+1;   
        u = G_deux_dix(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];    
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 9  % G2_20 --> G_deux_vingt
        bbe = size(hist,1)+1;   
        u = G_deux_vingt(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);       
        
     case 10  % G2_3 --> G_deux_trois
        bbe = size(hist,1)+1;   
        u = G_deux_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);       
        
      case 11  % chenwang_f_deux
        bbe = size(hist,1)+1;   
        u = chenwang_f_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 12 %chenwang_f3 --> chenwang_f_trois
        bbe = size(hist,1)+1;   
        u = chenwang_f_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
    
    case 13 % hs_quinze
        bbe = size(hist,1)+1;   
        u = hs_quinze(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
    case 14 % hs_vingt_deux
        bbe = size(hist,1)+1;   
        u = hs_vingt_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
    case 15 % hs_vingt_neuf
        bbe = size(hist,1)+1;   
        u = hs_vingt_neuf(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 16 % hs_cent_huit
        bbe = size(hist,1)+1;   
        u = hs_cent_huit(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 17 % hs_quarante_trois
        bbe = size(hist,1)+1;   
        u = hs_quarante_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
    case 18 % taowang_f_deux
        bbe = size(hist,1)+1;   
        u = taowang_f_deux(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
    case 19 % zhaowang_f_cinq
        bbe = size(hist,1)+1;   
        u = zhaowang_f_cinq(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);       
        
     case 20 % spring
        bbe = size(hist,1)+1;   
        u = spring(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
     case 21 % barnes
        bbe = size(hist,1)+1;   
        u = barnes(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
     case 22 % taowang_f_un
        bbe = size(hist,1)+1;   
        u = taowang_f_un(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));        
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
     case 23 % hs_cent_quatorze
        bbe = size(hist,1)+1;   
        u = hs_cent_quatorze(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
     case 24  % dembo
        bbe = size(hist,1)+1;   
        u = dembo(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
     case 25  % hs_dix_neuf
        bbe = size(hist,1)+1;   
        u = hs_dix_neuf(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
                
     case 26  % hs_vingt_trois
        bbe = size(hist,1)+1;   
        u = hs_vingt_trois(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
     case 27   % mezmontes
        bbe = size(hist,1)+1;   
        u = mezmontes(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
     case 28   % welded_beam
        bbe = size(hist,1)+1;   
        u = welded_beam(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);      
        
     case 29   % pressure_vessel
        bbe = size(hist,1)+1;   
        u = pressure_vessel(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);                 
        
     case 30   % speed_reducer
        bbe = size(hist,1)+1;   
        u = speed_reducer(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);               
        
      case 31   % opteng_rbf 
        bbe = size(hist,1)+1;   
        u = opteng_rbf(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
      case 32   % opteng_benchmark_quatre
        bbe = size(hist,1)+1;   
        u = opteng_benchmark_quatre(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));      
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
        
      case 33   % opteng_benchmark_cinq
        bbe = size(hist,1)+1;   
        u = opteng_benchmark_cinq(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
       case 34   % bertsimas_f_poly
        bbe = size(hist,1)+1;   
        u = bertsimas_f_poly(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
       case 35   % gomez
        bbe = size(hist,1)+1;   
        u = gomez(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
        
       case 36   % constr_branin
        bbe = size(hist,1)+1;   
        u = constr_branin(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));      
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
       case 37   % new_branin
        bbe = size(hist,1)+1;   
        u = new_branin(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);       
        
       case 38   % sasena
        bbe = size(hist,1)+1;   
        u = sasena(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
       case 39   % zilong_g_quatre
        bbe = size(hist,1)+1;   
        u = zilong_g_quatre(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
        
        
       case 40   % zilong_g_vingt_quatre
        bbe = size(hist,1)+1;   
        u = zilong_g_vingt_quatre(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);        
        
        case 41   % angun
        bbe = size(hist,1)+1;   
        u = angun(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));      
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
        case 42   % mad_six
        bbe = size(hist,1)+1;   
        u = mad_six(x);
        z = u + unifrnd(-(ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])),...
           (ka*abs(Det_bb_call(ceal{5}, nprob)-[f_min(nprob), zeros(1, ceal{4})])));       
        hist = [hist; bbe x' u h(x, nprob)];
        if h(x, nprob) == 0
            disp([bbe u(1) h(x, nprob)])%-------------------
        end          
        hist_file = fopen(['hist' num2str(nprob) '.txt'], 'w');
        fprintf(hist_file, [repmat(' %12.20f ',1, length(x) + ceal{4} + 3) '\n'],hist');
        fclose(hist_file);         
        
end