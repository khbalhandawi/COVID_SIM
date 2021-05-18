from StoMADS_experiment import check_folder, clear_folder
import subprocess
from subprocess import PIPE,STDOUT

#=========================================================#
#                  ROBUST SYSTEM COMMAND                  #
#=========================================================#
def system_command(command):
    ''' Execute system commands and return output to console '''
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

#=========================================================#
#                      C++ COMMAND                        #
#=========================================================#
def cpp_application( args, debug = False ):
    args_str = ' '.join(map(str,args)) # print variables as space delimited string
    command = "COVID_opt %s" %(args_str)
    
    print(command)
    if not debug:
        system_command(command)

#=========================================================#
#                          MAIN                           #
#=========================================================#
if __name__ == "__main__":

    import sys
    #==========================================================
    # input arguments   
    if (len(sys.argv) > 1):
        NOMAD_settings = {
            "healthcare_capacity"   : 90,
            "n_k"                   : int(sys.argv[1]),
            "MAX_BB_EVAL"           : 10000,
            "MIN_MESH_SIZE"         : 1e-31,
            "EPSILON"               : 1e-31,
            "nb_proc"               : int(sys.argv[2]),
            "n_k_success"           : 100,
            "config"                : sys.argv[3]
        }

        n_runs = int(sys.argv[4])

    else:
        NOMAD_settings = {
            "healthcare_capacity"   : 90,
            "n_k"                   : 4,
            "MAX_BB_EVAL"           : 10000,
            "MIN_MESH_SIZE"         : 1e-31,
            "EPSILON"               : 1e-31,
            "nb_proc"               : 8,
            "n_k_success"           : 100,
            "config"                : "default"
        }

        n_runs = 4

    # print(NOMAD_settings)

    folder = 'NOMAD_exp'
    machine = 'WORKSTATION'
    check_folder(folder)

    for i in range(n_runs):
        
        print("\n+=======================================================+")
        print("|                    EXPERIMENT %04d                    |" %(i+1))
        print("+=======================================================+\n")

        run_folder = "./%s/Run_%i" %(folder,i+1)
        check_folder(run_folder)
        clear_folder(run_folder)
        
        settings_file = run_folder + "/" + "settings.txt"
        file = open(run_folder + "/" + machine, 'w')
        file.close()

        with open(settings_file, 'a') as file:
            for key in NOMAD_settings:
                print(key, ' : ', NOMAD_settings[key], file=file)

        args = [NOMAD_settings[x] for x in NOMAD_settings.keys()]
        args += [run_folder]

        cpp_application(args)
