#!/usr/bin/env python3
"""Run the sample data.

See README.md in this directory for more information.
"""

import argparse
import gzip
import multiprocessing
import os
import shutil
import subprocess
import pandas as pd
import random
import numpy as np

def try_remove(f):
    try:
        os.remove(f)
    except OSError as e:
        pass

def parse_args():
    """Parse the arguments.

    On exit: Returns the result of calling argparse.parse()

    args_exe.covidsim is the name of the CovidSim executable
    args_exe.datadir is the directory with the input data
    args_exe.paramdir is the directory with the parameters in it
    args_exe.outputdir is the directory where output will be stored
    args_exe.threads is the number of threads to use
    """
    parser = argparse.ArgumentParser()
    try:
        cpu_count = len(os.sched_getaffinity(0)) # only on Linux
    except AttributeError:
        # os.sched_getaffinity isn't available
        cpu_count = int(multiprocessing.cpu_count())
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2

    work_path = os.getcwd()
    data_dir = os.path.join(work_path, "covid_sim")
    
    # Default values
    param_dir = os.path.join(data_dir, "param_files")
    output_dir = os.path.join(data_dir, "output_files")
    network_dir = os.path.join(data_dir, "network_files")
    read_only = 'N'
    first_setup = 'N'
    exe_path = "CovidSim.exe"
    country = "Canada"

    parser.add_argument(
            "--country",
            help="Country to run sample for",
            default=country)
    parser.add_argument(
            "--covidsim",
            help="Location of CovidSim binary",
            default=exe_path)
    parser.add_argument(
            "--datadir",
            help="Directory at root of input data",
            default=data_dir)
    parser.add_argument(
            "--paramdir",
            help="Directory with input parameter files",
            default=param_dir)
    parser.add_argument(
            "--outputdir",
            help="Directory to store output data",
            default=output_dir)
    parser.add_argument(
            "--networkdir",
            help="Directory to store network data and bins",
            default=network_dir)
    parser.add_argument(
            "--threads",
            help="Number of threads to use",
            default=cpu_count
            )
    parser.add_argument(
            "--firstsetup",
            help="whether to initialize network and other first time setup or not",
            default=first_setup
            )
    parser.add_argument(
            "--readonly",
            help="whether to simply read and plot final excel results",
            default=read_only
            )
    args_exe = parser.parse_args()

    return args_exe

def blackbox_CovidSim(i, args, params):

    # Setup blackbox
    args_exe = parse_args() # get default CovidSim call

    [run,pop_size,healthcare_capacity,country] = params
    args_exe.country = country # Change args
    if i != 0:
        args_exe.firstsetup == 'N' # Change args

    # Lists of places that need to be handled specially
    united_states = [ "United_States" ]
    canada = [ "Canada" ]
    usa_territories = ["Alaska", "Hawaii", "Guam", "Virgin_Islands_US", "Puerto_Rico", "American_Samoa"]
    nigeria = ["Nigeria"]

    # Location of a user supplied executable (CovidSim.exe):
    exe = args_exe.covidsim

    # Ensure output directory exists
    os.makedirs(args_exe.outputdir, exist_ok=True)

    # The admin file to use
    admin_file = os.path.join(args_exe.datadir, "admin_units",
        "{0}_admin.txt".format(args_exe.country))

    if not os.path.exists(admin_file):
        print("Unable to find admin file for country: {0}".format(args_exe.country))
        print("Data directory: {0}".format(args_exe.datadir))
        print("Looked for: {0}".format(admin_file))
        exit(1)

    # Population density file in gziped form, text file, and binary file as
    # processed by CovidSim
    if args_exe.country in united_states + canada:
        wpop_file_root = "usacan"
    elif args_exe.country in usa_territories:
        wpop_file_root = "us_terr"
    elif args_exe.country in nigeria:
        wpop_file_root = "nga_adm1"
    else:
        wpop_file_root = "eur"

    wpop_file_gz = os.path.join(
            args_exe.datadir,
            "populations",
            "wpop_{0}.txt.gz".format(wpop_file_root))
    if not os.path.exists(wpop_file_gz):
        print("Unable to find population file for country: {0}".format(args_exe.country))
        print("Data directory: {0}".format(args_exe.datadir))
        print("Looked for: {0}".format(wpop_file_gz))
        exit(1)

    wpop_file = os.path.join(
            args_exe.networkdir,
            "wpop_{0}.txt".format(wpop_file_root))
    wpop_bin = os.path.join(
            args_exe.networkdir,
            "{0}_pop_density.bin".format(args_exe.country))

    if args_exe.firstsetup == 'Y':
        try_remove(wpop_file)
        try_remove(wpop_bin)

    # gunzip wpop fie
    with gzip.open(wpop_file_gz, 'rb') as f_in:
        with open(wpop_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Configure pre-parameter file.  This file doesn't change between runs:
    if args_exe.country in united_states:
        pp_file = os.path.join(args_exe.paramdir, "preUS_R0=2.0.txt")
    elif args_exe.country in nigeria:
        pp_file = os.path.join(args_exe.paramdir, "preNGA_R0=2.0.txt")
    else:
        pp_file = os.path.join(args_exe.paramdir, "preUK_R0=2.0.txt")
    if not os.path.exists(pp_file):
        print("Unable to find pre-parameter file")
        print("Param directory: {0}".format(args_exe.paramdir))
        print("Looked for: {0}".format(pp_file))
        exit(1)

    # Configure No intervention parameter file.  This is run first
    # and provides a baseline
    no_int_file = os.path.join(args_exe.paramdir, "p_NoInt.txt")
    if not os.path.exists(no_int_file):
        print("Unable to find parameter file")
        print("Param directory: {0}".format(args_exe.paramdir))
        print("Looked for: {0}".format(no_int_file))
        exit(1)

    # Configure an intervention (controls) parameter file.
    # In reality you will run CovidSim many times with different parameter
    # controls.
    root = "PC7_CI_HQ_SD_V2"
    cf = os.path.join(args_exe.paramdir, "p_{0}.txt".format(root))
    if not os.path.exists(cf):
        print("Unable to find parameter file")
        print("Param directory: {0}".format(args_exe.paramdir))
        print("Looked for: {0}".format(cf))
        exit(1)

    school_file = None
    if args_exe.country in united_states:
        school_file = os.path.join(args_exe.datadir, "populations", "USschools.txt")

        if not os.path.exists(school_file):
            print("Unable to find school file for country: {0}".format(args_exe.country))
            print("Data directory: {0}".format(args_exe.datadir))
            print("Looked for: {0}".format(school_file))
            exit(1)

    # Some command_line settings
    r = 3.0
    rs = r/2

    # This is the temporary network that represents initial state of the
    # simulation
    network_bin = os.path.join(
            args_exe.networkdir,
            "Network_{0}_T{1}_R{2}.bin".format(args_exe.country, args_exe.threads, r))

    if args_exe.firstsetup == 'Y':
        try_remove(network_bin)

    # Run the no intervention sim.  This also does some extra setup which is one
    # off for each R.

    cf = os.path.join(args_exe.paramdir, "p_{0}.txt".format(root))
    print("Intervention: {0} {1} {2}".format(args_exe.country, root, r))
    cmd = [
            exe,
            "/c:{0}".format(args_exe.threads),
            "/A:" + admin_file
            ]
    if school_file:
        cmd.extend(["/s:" + school_file])

    # Random seeds used for model and 

    random_seed_1 = [5218636, 2181088]; # Setup seed for network (keep fixed)
    random_seed_2 = [random.randint(1e6,9e6),random.randint(1e6,9e6)];

    # output file root
    out_root = os.path.join(args_exe.outputdir,"{0}-{1}-{2}_{3}_R0={4}".format(i,run,args_exe.country, root, r))
    


    if args_exe.firstsetup == 'Y':

        # # Input variables (V1)
        # cmd.extend([
        #         "/PP:" + pp_file, # Preparam file
        #         "/P:" + cf, # Param file
        #         "/O:" + out_root,
        #         "/D:" + wpop_file, # Input (this time text) pop density
        #         "/M:" + wpop_bin, # Where to save binary pop density
        #         "/S:" + network_bin, # Where to save binary net setup
        #         "/R:{0}".format(rs),
        #         "/CLP1:"+str(args[0]), # default is 1.0 (Individual level compliance with quarantine) [1.0 - 0.9] E
        #         "/CLP3:"+str(args[2]), # default is 0.9 (Proportion of detected cases isolated) [0.1 - 0.9] T
        #         "/CLP2:"+str(args[1]), # default is 1.0 (Relative spatial contact rate given social distancing) [5.0 - 1.0] S
        #         str(random_seed_1[0]), # These four numbers are RNG seeds
        #         str(random_seed_1[1]),
        #         str(random_seed_2[0]),
        #         str(random_seed_2[1])
        #         ])

        # Input variables (V2)
        SD_scaling = args[1]*np.array([1.25, 0.1, 0.25, 1, 0.5, 1, 0.75])

        cmd.extend([
                "/PP:" + pp_file, # Preparam file
                "/P:" + cf, # Param file
                "/O:" + out_root,
                "/D:" + wpop_file, # Input (this time text) pop density
                "/M:" + wpop_bin, # Where to save binary pop density
                "/S:" + network_bin, # Where to save binary net setup
                "/R:{0}".format(rs),
                "/CLP1:"+str(args[0]), # default is 1.0 (Individual level compliance with quarantine) [1.0 - 0.9] E
                "/CLP3:"+str(args[2]), # default is 0.9 (Proportion of detected cases isolated) [0.1 - 0.9] T
                "/CLP21:"+str(SD_scaling[0]), # default is 1.0 (Relative spatial contact rate given social distancing) [5.0 - 1.0] S
                "/CLP22:"+str(SD_scaling[1]), 
                "/CLP23:"+str(SD_scaling[2]), 
                "/CLP24:"+str(SD_scaling[3]), 
                "/CLP25:"+str(SD_scaling[4]), 
                "/CLP26:"+str(SD_scaling[5]), 
                "/CLP27:"+str(SD_scaling[6]), 
                str(random_seed_1[0]), # These four numbers are RNG seeds
                str(random_seed_1[1]),
                str(random_seed_2[0]),
                str(random_seed_2[1])
                ])

    elif args_exe.firstsetup == 'N':
        
        # # Input variables (V1)
        # cmd.extend([
        #         "/PP:" + pp_file,
        #         "/P:" + cf,
        #         "/O:" + out_root,
        #         "/D:" + wpop_bin, # Binary pop density file (speedup)
        #         "/L:" + network_bin, # Network to load
        #         "/R:{0}".format(rs),
        #         "/CLP1:"+str(args[0]), # default is 1.0 (Individual level compliance with quarantine) [1.0 - 0.9] E
        #         "/CLP3:"+str(args[2]), # default is 0.9 (Proportion of detected cases isolated) [0.1 - 0.9] T
        #         "/CLP2:"+str(args[1]), # default is 1.0 (Relative spatial contact rate given social distancing) [5.0 - 1.0] S
        #         str(random_seed_1[0]), # These four numbers are RNG seeds
        #         str(random_seed_1[1]),
        #         str(random_seed_2[0]),
        #         str(random_seed_2[1])
        #         ])

        # Input variables (V2)
        SD_scaling = args[1]*np.array([1.25, 0.1, 0.25, 1, 0.5, 1, 0.75])

        cmd.extend([
                "/PP:" + pp_file,
                "/P:" + cf,
                "/O:" + out_root,
                "/D:" + wpop_bin, # Binary pop density file (speedup)
                "/L:" + network_bin, # Network to load
                "/R:{0}".format(rs),
                "/CLP1:"+str(args[0]), # default is 1.0 (Individual level compliance with quarantine) [1.0 - 0.9] E
                "/CLP3:"+str(args[2]), # default is 0.9 (Proportion of detected cases isolated) [0.1 - 0.9] T
                "/CLP21:"+str(SD_scaling[0]), # default is 1.0 (Relative spatial contact rate given social distancing) [5.0 - 1.0] S
                "/CLP22:"+str(SD_scaling[1]), 
                "/CLP23:"+str(SD_scaling[2]), 
                "/CLP24:"+str(SD_scaling[3]), 
                "/CLP25:"+str(SD_scaling[4]), 
                "/CLP26:"+str(SD_scaling[5]), 
                "/CLP27:"+str(SD_scaling[6]), 
                str(random_seed_1[0]), # These four numbers are RNG seeds
                str(random_seed_1[1]),
                str(random_seed_2[0]),
                str(random_seed_2[1])
                ])

    if args_exe.readonly == 'N':    
        print("Command line: " + " ".join(cmd))
        process = subprocess.run(cmd, check=True)

    output_file = args_exe.outputdir + "\{0}-{1}-{2}_{3}_R0={4}.avNE.severity.xls".format(i,run,args_exe.country, root, r)

    data = pd.read_csv(output_file, sep="\t")
    df = pd.DataFrame(data)

    S = df["S"].to_numpy()/pop_size
    I = df["I"].to_numpy()/pop_size
    R = df["R"].to_numpy()/pop_size

    F = df["cumDeath"].to_numpy()/pop_size
    Critical = df["Critical"].to_numpy()/pop_size

    infected = max(I)/pop_size
    fatalities = F[-1]/pop_size
    
    return [infected, fatalities, I, F, R, S, Critical]