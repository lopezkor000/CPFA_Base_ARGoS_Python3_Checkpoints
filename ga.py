#!/usr/bin/env python

import argos_util
import subprocess
import csv
import tempfile
import os
import numpy as np
import time
import argparse
import errno
import copy
from lxml import etree
import logging
#(4/19/2026) Charles Galperin required for save/load feature
from datetime import datetime # used for timestamps
import pickle # required for save/load feature
import sys # used to interrupt program
###(C.G.)
import pdb

# http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#(4/19/2026) Charles Galperin
# Alternate print() function for timestamps to be viewable in program outputs
def printTime(*args, **kwargs):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)
###(C.G.)

class ArgosRunException(Exception):
    pass


class iAntGA(object):
    def __init__(self, xml_file, pop_size=50, gens=20, elites=3,
                 mut_rate=0.1, robots=20, tags=1024, length=3600,
                 system="linux", tests_per_gen=10, terminateFlag=0, resume_file=None):

        self.xml_file = xml_file #qilu 03/26/2016
        self.system = system
        self.pop_size = pop_size
        self.gens = gens
        self.elites = elites
        self.mut_rate = mut_rate
        self.current_gen = 0
        self.robots = robots #qilu 03/26/2016
        self.tags = tags
        # Initialize population
        self.population_data=[]
        self.population = []
        self.prev_population = None
        self.system = system
        self.fitness = np.zeros(pop_size)
        self.starttime = int(time.time())
        self.length = length
        self.tests_per_gen = tests_per_gen
        self.terminateFlag =terminateFlag #qilu 01/21/2016
        self.not_evolved_idx = [-1]*self.pop_size #qilu 03/27/2016 check whether a population is from previous generation and is not modified
        self.not_evolved_count = [0]*self.pop_size #qilu 04/02
        self.prev_not_evolved_count = [0]*self.pop_size #qilu 04/02
        self.prev_fitness = np.zeros(pop_size) #qilu 03/27/2016
        
        name_and_extension = xml_file.split(".")
        XML_FILE_NAME = name_and_extension[0]

        # (4/20/2026) Charles Galperin
        # Check for progress continuation or new population
        if resume_file and os.path.exists(resume_file):
            printTime(f"Resuming from {resume_file}...")
            self.load_state(resume_file)
        else:
            printTime("Starting a fresh simulation...\n")
            for _ in range(pop_size):
                self.population.append(argos_util.uniform_rand_argos_xml(xml_file, robots, length, system))
        ###(C.G.)

        #dirstring = str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size) + "_r_" + str(robots) + "_t_" + \
        dirstring = XML_FILE_NAME +"_" + str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size) + "_r_" + \
            str(robots) +"_tag_"+str(tags)+ "_t_" + str(length) + "_k_" + str(tests_per_gen)
        self.save_dir = os.path.join("gapy_saves", dirstring)
        mkdir_p(self.save_dir)
        # (4/20/2026) Charles Galperin
        # creating a directory for checkpoints within the save directory already used for the experiment
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        mkdir_p(self.checkpoint_dir)
        ###(C.G.)

        logging.basicConfig(filename=os.path.join(self.save_dir,'iAntGA.log'),
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG
                            ) #(4/19/2026) Charles Galperin: timestamps also added to logging details

    # (4/19/2026) Charles Galperin
    # Save-and-continue functionality for long-running experimental workflows.
    #
    # This mechanism checkpoints experiment state after each generation, allowing
    # computationally intensive configurations to be executed incrementally rather
    # than as a single uninterrupted run. Experiments may be safely paused and
    # resumed, supporting large populations and extended evaluation horizons.
    #
    # Checkpoints are serialized using Python's `pickle` module to ensure data integrity.
    #
    # Saves the state of the GA
    def save_state(self):
        # Convert lxml objects to xml strings for pickling since lxml cant be pickled
        xml_strings_pop = [etree.tostring(ind) for ind in self.population]
        xml_strings_prev = None
        if self.prev_population is not None:
            xml_strings_prev = [etree.tostring(ind) for ind in self.prev_population]

        # Temporarily replace xml data with string
        original_population = self.population
        original_prev_population = self.prev_population

        self.population = xml_strings_pop
        self.prev_population = xml_strings_prev

        # Create an identifiable checkpoint name
        base_exp_name = os.path.splitext(os.path.basename(self.xml_file))[0]
        checkpoint_file_name = f"{base_exp_name}_Gen_{self.current_gen}_of_{self.gens}.pkl"
        path_to_save = os.path.join(self.checkpoint_dir, checkpoint_file_name)

        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)
        # restore with original lxml data before continuing experiment process
        self.population = original_population
        self.prev_population = original_prev_population
        printTime(f"State saved to {checkpoint_file_name}")

    # This loads a GA state from a pickle file
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            loaded_ga = pickle.load(f)
            # Update current instance attributes
            self.population = loaded_ga.population # currently strings
            self.fitness = loaded_ga.fitness
            self.current_gen = loaded_ga.current_gen
            self.not_evolved_idx = loaded_ga.not_evolved_idx
            self.not_evolved_count = loaded_ga.not_evolved_count
            self.prev_not_evolved_count = loaded_ga.prev_not_evolved_count
            # Turn loaded string data back to lxml
            self.population = [etree.fromstring(xml_str) for xml_str in self.population]
            if self.prev_population is not None:
                self.prev_population = [etree.fromstring(xml_str) for xml_str in self.prev_population]
            printTime(f"Resumed from {filename} at generation {self.current_gen}\n")
    ###(C.G.)

    def test_fitness(self, argos_xml, seed):
        argos_util.set_seed(argos_xml, seed)
        xml_str = etree.tostring(argos_xml)
        cwd = os.getcwd()
        tmpf = tempfile.NamedTemporaryFile('wb', suffix=".argos", prefix="gatmp",
                                           dir=os.path.join(cwd, "experiments"),
                                           delete=False) # 'w' changed to 'wb'
        tmpf.write(xml_str)
        tmpf.close()
        argos_args = ["argos3", "-n", "-c", tmpf.name]
        argos_run = subprocess.Popen(argos_args, stdout=subprocess.PIPE, text=True)
          
        # Wait until argos is finished
        while argos_run.poll() is None:
            time.sleep(0.5)
     
        if argos_run.returncode != 0:
            logging.error("Argos failed test")
            # when argos fails just return fitness 0
            return 0
        lines = argos_run.stdout.readlines()
        if os.path.exists(tmpf.name):
            os.unlink(tmpf.name)
        printTime(lines[-1])
        logging.info("partial fitness = %f", float(lines[-1].strip().split(",")[0]))
        return float(lines[-1].strip().split(",")[0])

    def run_ga(self):
        # (4/19/2026) Charles Galperin
        # Moved experiment setting outputs from location prior to initializing GA to here
        # so that the updated values after initialization were shown
        printTime("pop_size =" + str(pop_size))
        printTime("gens=" + str(gens))
        printTime("elites=" + str(elites))
        printTime("mut_rate=" + str(mut_rate))
        printTime("robots=" + str(robots))
        printTime("tags=" + str(tags))
        printTime("time=" + str(length / 60) + " minutes")
        printTime("Evaluations=" + str(tests_per_gen))
        print()
        ###(C.G.)
        while self.current_gen <=self.gens and self.terminateFlag == 0:
            self.run_generation()

    def run_generation(self):
        logging.info("Starting generation: " + str(self.current_gen))
        self.fitness = np.zeros(pop_size) #reset it
        seeds = [np.random.randint(2 ** 32) for _ in range(self.tests_per_gen)]
        logging.info("Seeds for generation: " + str(seeds))
        for i, p in enumerate(self.population):
            printTime("Gen: "+str(self.current_gen)+'; Population: '+str(i+1))
            for test_id in range(self.tests_per_gen):
                seed = seeds[test_id]
                logging.info("pop %d at test %d with seed %d", i, test_id, seed)
                if self.not_evolved_idx[i] == -1 or self.not_evolved_count[i] > 3:
                    self.not_evolved_count[i] =0;    
                    self.fitness[i] += self.test_fitness(p, seed)
                else: #qilu 03/27/2016 avoid recompute
                    self.fitness[i] += self.prev_fitness[self.not_evolved_idx[i]]
                    logging.info("partial fitness = %d", self.prev_fitness[self.not_evolved_idx[i]])
        # use average fitness as fitness
        for i in range(len(self.fitness)):
            logging.info("pop %d total fitness = %g", i, self.fitness[i])
            self.fitness[i] /= self.tests_per_gen
            logging.info("pop %d avg fitness = %g", i, self.fitness[i])

        # sort fitness and population
        #fitpop = sorted(zip(self.fitness, self.population), reverse=True)
        #self.fitness, self.population = map(list, zip(*fitpop))

        # (4/19/2026) Charles Galperin
        # Original method ###
        # fitpop = sorted(zip(self.fitness, self.population, self.not_evolved_count), reverse=True) #qilu 04/02 add not_evolved_count
        #
        # original method: if fitness scores were identical sorting would move to comparing lxml
        # which is a functionality that does not exist for lxml, this would lead to a crash
        # "TypeError: '<' not supported between instances of 'lxml.etree._Element' and 'lxml.etree._Element'"
        #
        # To avoid conflicts with identical fitness scores an alternate method of sorting will be implemented
        fit_pop_index = range(len(self.fitness))
        # Sort based on self.fitness only
        sorted_fit_pop_index = sorted(fit_pop_index, key=lambda x: self.fitness[x], reverse=True)
        # rebuild list after sorting
        fitpop = [(self.fitness[i], self.population[i], self.not_evolved_count[i]) for i in sorted_fit_pop_index]
        ### (C.G.) fix complete
        self.fitness, self.population, self.not_evolved_count = map(list, zip(*fitpop))

        self.save_population(seed)

        self.prev_population = copy.deepcopy(self.population)
        self.prev_fitness = copy.deepcopy(self.fitness) #qilu 03/27
        self.prev_not_evolved_count = copy.deepcopy(self.not_evolved_count) #qilu 04/02

        self.not_evolved_idx=[] #qilu 03/27/2016
        self.not_evolved_count = [] #qilu 04/02/2016
        self.population = []
        self.check_termination() #qilu 01/21/2016 add this function
        self.population_data=[] # qilu 01/21/2016 reset it
        # Add elites
        for i in range(self.elites):
            # reverse order from sort
            self.population.append(self.prev_population[i])
            self.not_evolved_idx.append(i) 
            self.not_evolved_count.append(self.prev_not_evolved_count[i] + 1)

        # Now do crossover and mutation until population is full

        num_newOffSpring = self.pop_size - self.elites
        #pdb.set_trace()
        count = 0
        for i in range(num_newOffSpring):
            if count == num_newOffSpring: break
            p1c = np.random.choice(len(self.prev_population), 2)
            p2c = np.random.choice(len(self.prev_population), 2)
            if p1c[0] <= p1c[1]:
                parent1 = self.prev_population[p1c[0]]
                idx1 = p1c[0]
            else: 
                parent1 = self.prev_population[p1c[1]]
                idx1 = p1c[1]
                
            if p2c[0] <= p2c[1]:
                parent2 = self.prev_population[p2c[0]]
                idx2 = p2c[0]
            else:
                parent2 = self.prev_population[p2c[1]]
                idx2 = p2c[1]
            #if parent1 != parent2 and np.random.uniform()<0.5: #qilu 11/26/2015
            #pdb.set_trace()
            if parent1 != parent2: #qilu 03/26/2016
                children = argos_util.uniform_crossover(xml_file, parent1, parent2, 0.5, self.system) # qilu 03/07/2016 add the crossover rate p  
            else:
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for child in children:
                argos_util.mutate_parameters(child, self.mut_rate)
                self.population.append(child)
                if argos_util.get_parameters(parent1) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx1)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx1] + 1)
                elif argos_util.get_parameters(parent2) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx2) 
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx2] + 1)
                else:
                    self.not_evolved_idx.append(-1)
                    self.not_evolved_count.append(0)
            count += 2
            while count > num_newOffSpring:
                del self.population[-1]
                del self.not_evolved_idx[-1]
                del self.not_evolved_count[-1]
                count -=1
        self.current_gen += 1
        # (4/19/2026) Charles Galperin
        # Checkpoint created after the computation of a generation
        self.save_state()
        ###(C.G.)

    def check_termination(self):
        upperBounds = [1.0, 1.0, 2.0, 20.0, 1.0, 20.0, 180.0]
        fitness_convergence_rate = 0.95
        diversity_rate=0.035
        #data_keys= self.population_data[0].keys()
        #data_keys.sort()
        data_keys= sorted(self.population_data[0].keys()) # (4/19/2026) Charles Galperin: Python3 fix
        complete_data =[]
        for data in self.population_data:
            complete_data.append([float(data[key]) for key in data_keys])
        npdata = np.array(complete_data)

        #Fitness convergence and population diversity
        means = npdata.mean(axis=0)
        stds = np.delete(npdata.std(axis=0), [7, 8])
        #pdb.set_trace()
        normalized_stds = stds/upperBounds

        current_fitness_rate = means[7]/npdata[0,7]
        current_diversity_rate = normalized_stds.max()
        if current_diversity_rate<=diversity_rate and current_fitness_rate>= fitness_convergence_rate:
            self.terminateFlag = 1
            printTime("Convergent ...\n")
        elif current_diversity_rate>diversity_rate and current_fitness_rate<fitness_convergence_rate:
            printTime('Fitness is not convergent ...')
            printTime('Fitness rate is '+str(current_fitness_rate))
            printTime('Deviation is '+str(current_diversity_rate))
        elif current_diversity_rate > diversity_rate:
            printTime('population diversity is high ...')
            printTime('The current standard deviation is '+str(current_diversity_rate)+', which is greater than '+str(diversity_rate)+' ...')
        else:
            printTime('Fitness is not convergent ...')
            printTime('The current rate of mean of fitness is '+str(current_fitness_rate)+', which is less than '+str(fitness_convergence_rate)+' ...')


    def save_population(self, seed):
        save_dir = self.save_dir
        mkdir_p(save_dir)
        filename = "gen_%d.gapy" % self.current_gen
        #population_data = []
        for f, p in zip(self.fitness, self.population):
            data = copy.deepcopy(argos_util.get_parameters(p)) 
            #data2= copy.deepcopy(argos_util.get_controller_params(p)) #qilu 07/25
            if 'PrintFinalScore' in data: 
                del data['PrintFinalScore']
            data["fitness"] = f
            data["seed"] = seed
            self.population_data.append(data)
            #population_data2.append(data2)
            #print data
        # (4/19/2026) Charles Galperin
        # Original method 'Python2 style'
        # data_keys = argos_util.PARAMETER_LIMITS.keys()
        # data_keys.append("fitness")
        # data_keys.append("seed")
        # data_keys.sort()
        #
        # Python3 fix
        data_keys = sorted(list(argos_util.PARAMETER_LIMITS.keys()) + ["fitness", "seed"])
        ###(C.G.)

        #data_keys2 = argos_util.controller_params_LIMITS.keys()
        #data_keys2.sort()
        with open(os.path.join(save_dir, filename), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.population_data) #qilu 07/27
            
#            writer2 = csv.DictWriter(csvfile, fieldnames=data_keys2, extrasaction='ignore')
#            writer2.writeheader()
#            writer2.writerows(population_data2) #qilu 07/27

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA for argos')
    parser.add_argument('-f', '--file', action='store', dest='xml_file')
    parser.add_argument('-s', '--system', action='store', dest='system')
    parser.add_argument('-r', '--robots', action='store', dest='robots', type=int)
    parser.add_argument('-m', '--mut_rate', action='store', dest='mut_rate', type=float)
    parser.add_argument('-e', '--elites', action='store', dest='elites', type=int)
    parser.add_argument('-g', '--gens', action='store', dest='gens', type=int)
    parser.add_argument('-p', '--pop_size', action='store', dest='pop_size', type=int)
    parser.add_argument('-t', '--time', action='store', dest='time', type=int)
    parser.add_argument('-k', '--tests_per_gen', action='store', dest='tests_per_gen', type=int)
    parser.add_argument('-o', '--terminateFlag', action='store', dest='terminateFlag', type=int)
    # (4/19/2026) Charles Galperin
    # flag added for loading checkpoint files
    parser.add_argument('-rf', '--resume_file', action='store', dest='resume_file', help='Path to a .pkl checkpoint file')
    ###(C.G.)
    pop_size = 50
    gens = 150
    elites = 1
    mut_rate = 0.05
    robots = 24  #robots = 16
    tags = 256 #qilu 03/26 for naming the output directory
    system = "linux"
    length = 720 # 12 minutes, length is in second. default length = 3600
    tests_per_gen = 10
    terminateFlag = 0
    
    args = parser.parse_args()
    # (4/19/2026) Charles Galperin
    # Moved output of experiment settingsto run_ga() because values did not reflect updated args here
    #print("pop_size ="+ str(pop_size))
    #print("gens="+str(gens))
    #print("elites="+ str(elites))
    #print("mut_rate="+str(mut_rate))
    #print("robots="+str(robots))
    #print("tags="+str(tags))
    #print("time="+str(length/60)+" minutes")
    #print("Evaluation="+str(tests_per_gen))
    ###(C.G.)

    #xml_file = raw_input('Choose a file name(e.g. cluster_2_mac.argos)')
    
    if args.xml_file:
        xml_file = args.xml_file
        printTime("The input file: "+xml_file)

    if args.pop_size:
        pop_size = args.pop_size

    if args.gens:
        gens = args.gens

    if args.elites:
        elites = args.elites

    if args.mut_rate:
        mut_rate = args.mut_rate

    if args.robots:
        robots = args.robots

    if args.system:
        system = args.system

    if args.time:
        length = args.time

    if args.tests_per_gen:
        tests_per_gen = args.tests_per_gen

    if args.terminateFlag:
        terminateFlag = args.terminateFlag

    # (4/19/2026) Charles Galperin
    # Logic for loading a checkpoint or starting a fresh experiment
    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print(f"ERROR: Resume file '{args.resume_file}' not found.")
            # Stop the script entirely with an error code
            sys.exit(1)
        else:
            resume_file = args.resume_file
    else:
        resume_file = None
    ###(C.G.)

    ga = iAntGA(xml_file = xml_file,
                pop_size=pop_size,
                gens=gens,
                elites=elites,
                mut_rate=mut_rate,
                robots=robots,
                tags=tags,
                length=length,
                system=system,
                tests_per_gen=tests_per_gen,
                terminateFlag = terminateFlag,
                resume_file=resume_file) # (4/19/2026) Charles Galperin: checkpoint option added
    start = time.time()
    ga.run_ga()
    stop = time.time()
    printTime('The loaded file is '+ xml_file+' ...')
    printTime('It runs '+str((stop-start)/3600.0)+ ' hours...')
