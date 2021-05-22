function log_generation(Y,L,U,Y_m,L_m,U_m,ceal)
    
    global cumilative_f_evals f_best
    
    %-------------------------------------------------------------%
    % Log generation statistics
    folder = ceal{19};
    gen_data = [cumilative_f_evals,f_best,Y,L,U,Y_m,L_m,U_m];
    
    generation_file = fopen([folder,'/G_stats_GA.txt'], 'a');
    format = [repmat('%12.20f,',1, length(gen_data)) '\n']; format(end-2) = ''; % remove last delimliter
    fprintf(generation_file, format, gen_data);
    fclose(generation_file); 

end