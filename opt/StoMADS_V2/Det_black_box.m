function z = Det_black_box(x, nprob)

format long g


if iscolumn(x')
    x=x';
end

switch nprob
    
    

    
    case 1  % crescent
        z = crescent(x);
        
    case 2 % disk  
        z = disk(x);

    case 3  % snake
        z = snake(x);

    case 4  % pentagon
        z = pentagon(x);

    case 5  % mad1 --> mad_un 
        z = mad_un(x);

    case 6  % mad2 --> mad_deux 
        z = mad_deux(x);
        
    case 7  % wong_deux
        z = wong_deux(x);

    case 8  % G2_10 --> G_deux_dix  
        z = G_deux_dix(x);

    case 9  % G2_20 --> G_deux_vingt  
        z = G_deux_vingt(x);

     case 10  % G2_3 --> G_deux_trois  
        z = G_deux_trois(x);

      case 11  % chenwang_f_deux 
        z = chenwang_f_deux(x);

    case 12 %chenwang_f3 --> chenwang_f_trois 
        z = chenwang_f_trois(x);

    case 13 % hs_quinze  
        z = hs_quinze(x);

    case 14 % hs_vingt_deux
        z = hs_vingt_deux(x);

    case 15 % hs_vingt_neuf 
        z = hs_vingt_neuf(x);

    case 16 % hs_trente  
        z = hs_cent_huit(x);

    case 17 % hs_quarante_trois  
        z = hs_quarante_trois(x);

    case 18 % taowang_f_deux 
        z = taowang_f_deux(x);

    case 19 % zhaowang_f_cinq 
        z = zhaowang_f_cinq(x);

     case 20 % spring
        z = spring(x);

     case 21 % barnes  
        z = barnes(x);

     case 22 % taowang_f_un 
        z = taowang_f_un(x);

     case 23 % hs_cent_quatorze   
        z = hs_cent_quatorze(x);

     case 24  % dembo 
        z = dembo(x);

     case 25  % hs_dix_neuf  
        z = hs_dix_neuf(x);
   
     case 26  % hs_vingt_trois  
        z = hs_vingt_trois(x);

     case 27   % mezmontes
        z = mezmontes(x);

     case 28   % welded_beam  
        z = welded_beam(x);

     case 29   % pressure_vessel  
        z = pressure_vessel(x);

     case 30   % speed_reducer  
        z = speed_reducer(x);

      case 31   % opteng_rbf  
        z = opteng_rbf(x);

      case 32   % opteng_benchmark_quatre
        z = opteng_benchmark_quatre(x);

      case 33   % opteng_benchmark_cinq 
        z = opteng_benchmark_cinq(x);

       case 34   % bertsimas_f_poly  
        z = bertsimas_f_poly(x);

       case 35   % gomez  
        z = gomez(x);

       case 36   % constr_branin 
        z = constr_branin(x);

       case 37   % new_branin  
        z = new_branin(x);

       case 38   % sasena  
        z = sasena(x);

       case 39   % zilong_g_quatre 
        z = zilong_g_quatre(x);

       case 40   % zilong_g_vingt_quatre
        z = zilong_g_vingt_quatre(x);        
        
        case 41   % angun  
        z = angun(x);

        case 42   % mad_six 
        z = mad_six(x);

end