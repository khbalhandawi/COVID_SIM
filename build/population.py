'''
this file contains functions that help initialize the population
parameters for the simulation
'''

import numpy as np

def load_population(tstep=0, folder='data_tstep'):
    '''loads population data at given timestep from disk

    Function that loads the simulation data from specific files on the disk.
    Loads the state of the population matrix

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    tstep : int
        the timestep that will be saved
    ''' 
    population = np.loadtxt('%s/population_%i.bin' %(folder, tstep))
    return population

def load_ground_covered(tstep=0, folder='data_tstep'):
    '''loads population tracking data at given timestep from disk

    Function that loads the population tracking data from specific files on the disk.
    Loads the state of the ground_covered matrix

    Keyword arguments
    -----------------

    tstep : int
        the timestep that will be saved
    ''' 
    ground_covered = np.loadtxt('%s/ground_covered_%i.bin' %(folder, tstep))
    return ground_covered

def load_grid_coords(tstep=0, folder='data_tstep'):
    '''loads tracking grid coordinates from disk

    Function that loads the tracking grid coordinates from specific files on the disk.
    Loads the state of the grid_coords matrix

    Keyword arguments
    -----------------

    tstep : int
        the timestep that will be saved
    ''' 
    grid_coords = np.loadtxt('%s/grid_coords.bin' %(folder))
    return grid_coords


class Population_trackers():
    '''class used to track population parameters

    Can track population parameters over time that can then be used
    to compute statistics or to visualise. 

    TODO: track age cohorts here as well
    '''
    def __init__(self, *args, **kwargs):
        self.susceptible = []
        self.infectious = []
        self.recovered = []
        self.fatalities = []
        self.Config = args[0]
        self.distance_travelled = [0.0]
        self.total_distance = np.zeros(self.Config.pop_size) # distance travelled by individuals
        self.mean_perentage_covered = [0.0]
        self.grid_coords = load_grid_coords(folder='population')
        self.ground_covered = load_ground_covered(tstep=0, folder='population')
        self.perentage_covered = np.zeros(self.Config.pop_size) # portion of world covered by individuals
        #PLACEHOLDER - whether recovered individual can be reinfected
        self.reinfect = False 

    def update_counts(self, population, frame):
        '''docstring
        '''
        pop_size = population.shape[0]
        self.infectious.append(len(population[population[:,6] == 1]))
        self.recovered.append(len(population[population[:,6] == 2]))
        self.fatalities.append(len(population[population[:,6] == 3]))

        # Total distance travelled
        if self.Config.track_position:
            speed_vector = population[:,3:5][population[:,11] == 0] # speed of individuals within world
            distance_individuals = np.linalg.norm( speed_vector ,axis = 1) * self.Config.dt # current distance travelled 

            self.total_distance[population[:,11] == 0] += distance_individuals # cumulative distance travelled
            self.distance_travelled.append(np.mean(self.total_distance)) # mean cumulative distance

        # Compute and track ground covered
        if self.Config.track_GC and (frame % self.Config.update_every_n_frame) == 0:

            # Import ground covered
            self.ground_covered = load_ground_covered(tstep=frame, folder='population')
            #1D
            if self.ground_covered.ndim > 1:
                self.perentage_covered = np.count_nonzero(self.ground_covered,axis=1)/len(self.grid_coords[:,0])
            else:
                self.perentage_covered = np.count_nonzero(self.ground_covered)/len(self.grid_coords[:,0])

            self.mean_perentage_covered.append(np.mean(self.perentage_covered)) # mean ground covered

        # Mark recovered individuals as susceptible if reinfection enables
        if self.reinfect:
            self.susceptible.append(pop_size - (self.infectious[-1] +
                                                self.fatalities[-1]))
        else:
            self.susceptible.append(pop_size - (self.infectious[-1] +
                                                self.recovered[-1] +
                                                self.fatalities[-1]))
        
        
#=============================================================================
# Main execution 
if __name__ == '__main__':
    population = load_population(tstep=1051, folder='population')
    