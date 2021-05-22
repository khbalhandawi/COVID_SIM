%=========================================================================%
%                          PLOTTING FUNCTIONS                             %
%=========================================================================%
function [fig,ax1,ax2] = build_fig(run_number)
    
    fig = figure(run_number);
    fig_height = 250*2;
    set(fig, 'Position', [100, 100, 600, 800])
    
    ax1 = subplot(2,1,1,'Parent',fig); % subplot
    % set(ax,'xlim',[1,options.MaxGenerations+1]);
    title('Range of population size, Mean','interp','none')
    xlabel(ax1,'Number of function evaluations','interp','none')
    
    ax2 = subplot(2,1,2,'Parent',fig); % subplot
    % set(ax,'xlim',[1,options.MaxGenerations+1]);
    title(ax2,'Range of population score, Mean','interp','none')
    xlabel(ax2,'Number of function evaluations','interp','none')
    
end