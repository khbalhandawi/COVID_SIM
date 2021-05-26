import subprocess
from subprocess import PIPE,STDOUT
import numpy as np

from functionsUtilities.utils import load_matrix

#==============================================================================#
# Execute system commands and return output to console
def system_command(command):
    #CREATE_NO_WINDOW = 0x08000000 # Create no console window flag

    p = subprocess.Popen(command,shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                         ) # disable windows errors

    for line in iter(p.stdout.readline, b''):
        line = line.decode('utf-8')
        print(line.rstrip()) # print line by line
        # rstrip() to remove \n separator

    output, error = p.communicate()
    if p.returncode != 0: # crash the program
        raise Exception("cpp failed %d %s %s" % (p.returncode, output, error))

#==============================================================================#
# COVID_SIM_UI COMMAND
def application_COVID_SIM_UI( i, design_variables, parameters, output_file_n, debug = False ):
    design_variables_str = ' '.join(map(str,design_variables)) # print variables as space delimited string
    parameters_str = ' '.join(map(str,parameters)) # print parameters as space delimited string
    command = "COVID_SIM_UI %i 0 %s %s %s" %(i, design_variables_str, parameters_str, output_file_n)
    
    print(command)
    if not debug:
        system_command(command)

#==============================================================================#
# COVID_SIM_UI blackbox
def blackbox_COVID_SIM_UI(i, args, params):

    [run, output_file_base, pop_size, params_COVID_SIM_UI, read_process] = params

    output_file_n = '%s_%i.log' %(output_file_base, i)
    application_COVID_SIM_UI(i,args,params_COVID_SIM_UI,output_file_n)
    #--------------------------------------------------------------------------#
    # Get results
    output_file_path = "data/" + output_file_n

    with open(output_file_path, 'r') as fh:
        for line in fh:
            pass
        last = line

    values = last.split(',')
    [ _, _, _, _, _, infected, fatalities, mean_distance, mean_GC, _ ] = values

    # If being used to obtain stochastic profiles of disease trajectory
    if read_process:
        run_data = load_matrix('SIRF_data', folder='population')
        run_data_M = load_matrix('mean_GC_data', folder='population')
        run_data_R0 = load_matrix('mean_R0_data', folder='population')

        I = [x / pop_size for x in run_data[:,2]]
        R = [x / pop_size for x in run_data[:,3]]
        F = [x / pop_size for x in run_data[:,4]]
        M = [x * 100 for x in run_data_M[:,1]]

        return [int(infected)/pop_size, int(fatalities)/pop_size, float(mean_GC), float(mean_distance), I, F, R, M, run_data_R0]
    else:
        return [int(infected), int(fatalities), float(mean_GC), float(mean_distance)]