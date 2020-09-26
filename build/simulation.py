import os
import sys

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from utils import check_folder
from config import Configuration, config_error
from environment import build_hospital
from visualiser import build_fig, build_fig_scatter, draw_tstep, draw_tstep_scatter, set_style, build_fig_SIRonly, draw_SIRonly
from population import Population_trackers, load_population

#set seed for reproducibility
#np.random.seed(100)

class Simulation():

    def __init__(self, *args, **kwargs):
        #load default config data
        self.Config = Configuration()
        #set_style(self.Config)

    def population_init(self):
        '''(re-)initializes population'''
        self.population = load_population(tstep=0, folder='population')

    def initialize_simulation(self):
        #initialize times
        self.frame = 0
        self.time = 0
        #initialize default population
        self.population_init()
        #initalise population tracker
        self.pop_tracker = Population_trackers(self.Config)
        
    def tstep(self):
        '''
        takes a time step in the simulation
        '''
        start = time.time() # start clock
        
        #======================================================================================#
        #load population data for current time frame
        self.population = load_population(tstep=self.frame, folder='population')

        #======================================================================================#
        #update population statistics
        self.pop_tracker.update_counts(self.population, self.frame)

        #======================================================================================#
        #visualise
        if self.Config.visualise and (self.frame % self.Config.visualise_every_n_frame) == 0:
            if self.Config.n_plots == 1:
                draw_tstep_scatter(self.Config, self.population, self.pop_tracker, self.frame, 
                                   self.fig, self.ax1, self.tight_bbox)
            elif self.Config.n_plots == 2:
                draw_tstep(self.Config, self.population, self.pop_tracker, self.frame, 
                           self.fig, self.spec, self.ax1, self.ax2, self.tight_bbox)

        #report stuff to console
        if (self.Config.verbose) and ((self.frame % self.Config.report_freq) == 0):
            end = time.time()
            time_elapsed =  end - start # elapsed time

            sys.stdout.write('\r')
            sys.stdout.write('%i: S: %i, I: %i, R: %i, in treatment: %i, F: %i, of total: %i, time: %.5f' 
                            %(self.frame, self.pop_tracker.susceptible[-1], self.pop_tracker.infectious[-1],
                            self.pop_tracker.recovered[-1], len(self.population[self.population[:,10] == 1]),
                            self.pop_tracker.fatalities[-1], self.Config.pop_size, 
                            time_elapsed))

        #run callback
        self.callback()

        #======================================================================================#
        #update frame
        self.frame += 1
        self.time += self.Config.dt


    def callback(self):
        '''placeholder function that can be overwritten.

        By overwriting this method any custom behavior can be implemented.
        The method is called after every simulation timestep.
        '''
        pass

    def run(self):
        '''run simulation'''

        if self.Config.visualise:
            if self.Config.n_plots == 1:
                self.fig, self.ax1, self.tight_bbox = build_fig_scatter(self.Config)
            elif self.Config.n_plots == 2:
                self.fig, self.spec, self.ax1, self.ax2, self.tight_bbox = build_fig(self.Config)

        i = 0
        
        while i < self.Config.simulation_steps:
            try:
                self.tstep()
            except KeyboardInterrupt:
                print('\nCTRL-C caught, exiting')
                sys.exit(1)

            #check whether to end if no infectious persons remain.
            #check if self.frame is above some threshold to prevent early breaking when simulation
            #starts initially with no infections.
            if self.Config.endif_no_infections and self.frame >= 300:
                if len(self.population[(self.population[:,6] == 1) | 
                                       (self.population[:,6] == 4)]) == 0:
                    i = self.Config.simulation_steps
            else:
                i += 1

        
        if self.Config.plot_last_tstep:
            self.fig_sir, self.spec_sir, self.ax1_sir = build_fig_SIRonly(self.Config)
            draw_SIRonly(self.Config, self.population, self.pop_tracker, self.frame, 
                            self.fig_sir, self.spec_sir, self.ax1_sir)

        #report outcomes
        if self.Config.verbose:
            print('\n-----stopping-----\n')
            print('total timesteps taken: %i' %self.frame)
            print('total dead: %i' %len(self.population[self.population[:,6] == 3]))
            print('total recovered: %i' %len(self.population[self.population[:,6] == 2]))
            print('total infected: %i' %len(self.population[self.population[:,6] == 1]))
            print('total infectious: %i' %len(self.population[(self.population[:,6] == 1) |
                                                            (self.population[:,6] == 4)]))
            print('total unaffected: %i' %len(self.population[self.population[:,6] == 0]))

#=============================================================================
# Main execution 
if __name__ == '__main__':

    current_path = os.getcwd() # Working directory of file
    
    #initialize
    sim = Simulation()

    #load config settings
    sim.Config.read_from_file('configuration_debug.ini')
    
    #set visuals
    sim.Config.plot_style = 'default' #can also be dark
    sim.Config.plot_text_style = 'LaTeX' #can also be LaTeX
    sim.Config.visualise = True
    sim.Config.visualise_every_n_frame = 1
    sim.Config.plot_last_tstep = True
    sim.Config.verbose = True
    sim.Config.report_freq = 50
    sim.Config.save_plot = True
    # sim.Config.marker_size = (2700 - sim.Config.pop_size) / 140
    sim.Config.marker_size = 5

    # Trace path of a single individual on grid
    sim.Config.trace_path = True
    
    sim.initialize_simulation()
    #run, hold CTRL+C in terminal to end scenario early
    sim.run()

    plt.show()