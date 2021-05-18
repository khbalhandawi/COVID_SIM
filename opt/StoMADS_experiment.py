
#=========================================================#
#                 Create empty directory                  #
#=========================================================#
def check_folder(folder='data/'):
	import os
	'''check if folder exists, make if not present'''
	if not os.path.exists(folder):
		os.makedirs(folder)

#=========================================================#
#                     Clear directory                     #
#=========================================================#
def clear_folder(folder='data/'):
    import os
    '''clear all files inside folder'''
    for f in os.listdir(folder):
        dirname = os.path.join(folder, f)
        if not os.path.isdir(dirname):
            os.remove(dirname)

#=========================================================#
#                          MAIN                           #
#=========================================================#
if __name__ == "__main__":

    import sys
    #==========================================================
    # input arguments   
    if (len(sys.argv) > 1):
        StoMADS_settings = {
            "n_k"                   : int(sys.argv[1]),
            "epsilon_f"             : float(sys.argv[2]),
            "Delta"                 : 1.0,
            "gamma"                 : 5.0,
            "tau"                   : 0.5,
            "OPPORTUNISTIC_EVAL"    : True,
            "MAX_BB_EVAL"           : 10000,
            "nb_proc"               : int(sys.argv[3]),
            "n_k_success"           : 100
        }

        n_runs = int(sys.argv[4])

    else:
        StoMADS_settings = {
            "n_k"                   : 1,
            "epsilon_f"             : 0.1,
            "Delta"                 : 1.0,
            "gamma"                 : 5.0,
            "tau"                   : 0.5,
            "OPPORTUNISTIC_EVAL"    : True,
            "MAX_BB_EVAL"           : 10000,
            "nb_proc"               : 1,
            "n_k_success"           : 100
        }

        n_runs = 4

    print(StoMADS_settings)

    import matlab.engine
    eng = matlab.engine.start_matlab()

    folder = 'StoMADS_exp'
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

        with open(settings_file, 'w') as file:
            print(StoMADS_settings, file=file)

        eng.StoMADS_call(StoMADS_settings['n_k'],
            StoMADS_settings['epsilon_f'],
            StoMADS_settings['Delta'],
            StoMADS_settings['gamma'],
            StoMADS_settings['tau'],
            StoMADS_settings['MAX_BB_EVAL'],
            StoMADS_settings['nb_proc'],
            StoMADS_settings['n_k_success'],
            run_folder,nargout=0,
            )