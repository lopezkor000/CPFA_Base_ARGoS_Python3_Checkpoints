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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class IslandState(object):
    def __init__(self, population, elites):
        self.population = population
        self.pop_size = len(population)
        self.elites = min(elites, self.pop_size)
        self.fitness = np.zeros(self.pop_size)
        self.prev_population = None
        self.not_evolved_idx = [-1] * self.pop_size
        self.not_evolved_count = [0] * self.pop_size
        self.prev_not_evolved_count = [0] * self.pop_size
        self.prev_fitness = np.zeros(self.pop_size)
        self.population_data = []


class iAntGA(object):
    def __init__(self, xml_file, pop_size=50, gens=20, elites=3,
                 mut_rate=0.1, robots=20, tags=1024, length=3600,
                 system="linux", tests_per_gen=10, terminateFlag=0, resume_file=None,
                 islands=None, max_workers=None, migration_elites=None, migration_interval=None):

        self.xml_file = xml_file #qilu 03/26/2016
        self.system = system
        self.pop_size = pop_size
        self.gens = gens
        self.elites = elites
        self.mut_rate = mut_rate
        self.current_gen = 0
        self.robots = robots #qilu 03/26/2016
        self.tags = tags
        self.starttime = int(time.time())
        self.length = length
        self.tests_per_gen = tests_per_gen
        self.terminateFlag = terminateFlag #qilu 01/21/2016
        self.population_data = []
        self.cpu_count = os.cpu_count() or 1
        self.islands = []
        self.island_count = 0
        self.max_workers = self.cpu_count
        self.migration_elites = 1
        self.migration_interval = 1

        # (4/20/2026) Charles Galperin
        # Check for progress continuation or new population
        if resume_file and os.path.exists(resume_file):
            printTime(f"Resuming from {resume_file}...")
            self.load_state(resume_file)
            if islands is not None and islands != self.island_count:
                printTime("Requested islands ignored for resumed checkpoint")
        else:
            printTime("Starting a fresh simulation...\n")
            population = [
                argos_util.uniform_rand_argos_xml(xml_file, robots, length, system)
                for _ in range(pop_size)
            ]
            self.island_count = self._normalize_island_count(islands, pop_size)
            self.islands = self._split_population(population, self.island_count)

        # Apply runtime overrides for concurrency and migration settings
        if max_workers is None:
            self.max_workers = self._normalize_max_workers(self.max_workers)
        else:
            self.max_workers = self._normalize_max_workers(max_workers)
        if migration_elites is not None:
            self.migration_elites = max(0, migration_elites)
        if migration_interval is not None:
            self.migration_interval = max(1, migration_interval)

        # Initialize save directories for new runs or ensure existing directories on resume
        if not hasattr(self, "save_dir"):
            name_and_extension = xml_file.split(".")
            XML_FILE_NAME = name_and_extension[0]
            dirstring = XML_FILE_NAME +"_" + str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size) + "_r_" + \
                str(robots) +"_tag_"+str(tags)+ "_t_" + str(length) + "_k_" + str(tests_per_gen)
            self.save_dir = os.path.join("gapy_saves", dirstring)
        mkdir_p(self.save_dir)
        if not hasattr(self, "checkpoint_dir"):
            self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        mkdir_p(self.checkpoint_dir)
        ###(C.G.)

        logging.basicConfig(filename=os.path.join(self.save_dir,'iAntGA.log'),
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG
                            ) #(4/19/2026) Charles Galperin: timestamps also added to logging details

    def _normalize_island_count(self, islands, pop_size):
        if islands is None or islands <= 0:
            islands = self.cpu_count
        if islands > pop_size:
            islands = pop_size
        return max(1, islands)

    def _normalize_max_workers(self, max_workers):
        if max_workers is None or max_workers <= 0:
            return self.cpu_count
        return max_workers

    def _split_population(self, population, island_count):
        islands = []
        base_size = int(len(population) / island_count)
        remainder = len(population) - (base_size * island_count)
        start = 0
        for island_id in range(island_count):
            size = base_size + (1 if island_id < remainder else 0)
            subpop = population[start:start + size]
            islands.append(IslandState(subpop, self.elites))
            start += size
        return islands

    def _restore_population_xml(self, population):
        if not population:
            return population
        sample = population[0]
        if isinstance(sample, (bytes, str)):
            return [etree.fromstring(xml_str) for xml_str in population]
        return population

    def _upgrade_legacy_state(self, loaded_ga):
        island = IslandState([], loaded_ga.elites)
        island.population = loaded_ga.population
        island.fitness = loaded_ga.fitness
        island.prev_population = loaded_ga.prev_population
        island.not_evolved_idx = loaded_ga.not_evolved_idx
        island.not_evolved_count = loaded_ga.not_evolved_count
        island.prev_not_evolved_count = loaded_ga.prev_not_evolved_count
        island.prev_fitness = loaded_ga.prev_fitness
        island.population_data = loaded_ga.population_data
        island.pop_size = len(island.population)
        return island

    def _evaluate_fitness_task(self, argos_xml, seed):
        try:
            xml_copy = etree.fromstring(etree.tostring(argos_xml))
            return self.test_fitness(xml_copy, seed)
        except Exception:
            logging.exception("Fitness evaluation failed")
            return 0, "Fitness evaluation failed"

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
        original_populations = []
        original_prev_populations = []
        for island in self.islands:
            original_populations.append(island.population)
            original_prev_populations.append(island.prev_population)
            island.population = [etree.tostring(ind) for ind in island.population]
            if island.prev_population is not None:
                island.prev_population = [etree.tostring(ind) for ind in island.prev_population]

        # Create an identifiable checkpoint name
        base_exp_name = os.path.splitext(os.path.basename(self.xml_file))[0]
        checkpoint_file_name = f"{base_exp_name}_Gen_{self.current_gen}_of_{self.gens}.pkl"
        path_to_save = os.path.join(self.checkpoint_dir, checkpoint_file_name)

        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)
        # restore with original lxml data before continuing experiment process
        for i, island in enumerate(self.islands):
            island.population = original_populations[i]
            island.prev_population = original_prev_populations[i]
        printTime(f"State saved to {checkpoint_file_name}")

    # This loads a GA state from a pickle file
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            loaded_ga = pickle.load(f)
            # Update current instance attributes
            if hasattr(loaded_ga, "islands"):
                self.__dict__.update(loaded_ga.__dict__)
            else:
                self.__dict__.update(loaded_ga.__dict__)
                self.islands = [self._upgrade_legacy_state(loaded_ga)]
                self.island_count = 1
            # Turn loaded string data back to lxml
            for island in self.islands:
                island.population = self._restore_population_xml(island.population)
                if island.prev_population is not None:
                    island.prev_population = self._restore_population_xml(island.prev_population)
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
        
        # The last line of the output is the fitness
        fitness_line = lines[-1].strip()
        fitness_value = float(fitness_line.split(",")[0])
        
        # Return the raw fitness value and the full line for detailed logging
        return fitness_value, fitness_line

    def run_ga(self):
        # (4/19/2026) Charles Galperin
        # Moved experiment setting outputs from location prior to initializing GA to here
        # so that the updated values after initialization were shown
        printTime("pop_size =" + str(self.pop_size))
        printTime("gens=" + str(self.gens))
        printTime("elites=" + str(self.elites))
        printTime("mut_rate=" + str(self.mut_rate))
        printTime("robots=" + str(self.robots))
        printTime("tags=" + str(self.tags))
        printTime("time=" + str(self.length / 60) + " minutes")
        printTime("Evaluations=" + str(self.tests_per_gen))
        printTime("islands=" + str(self.island_count))
        printTime("max_workers=" + str(self.max_workers))
        printTime("migration_elites=" + str(self.migration_elites))
        printTime("migration_interval=" + str(self.migration_interval))
        print()
        ###(C.G.)
        while self.current_gen <=self.gens and self.terminateFlag == 0:
            self.run_generation()

    def run_generation(self):
        logging.info("Starting generation: " + str(self.current_gen))
        if self.tests_per_gen <= 0:
            logging.error("tests_per_gen must be > 0")
            return
        seeds = [np.random.randint(2 ** 32) for _ in range(self.tests_per_gen)]
        logging.info("Seeds for generation: " + str(seeds))

        tasks = []
        for island_id, island in enumerate(self.islands):
            island.fitness = np.zeros(island.pop_size)
            for i, p in enumerate(island.population):
                if island.not_evolved_idx[i] == -1 or island.not_evolved_count[i] > 3:
                    island.not_evolved_count[i] = 0
                    for task_id, seed in enumerate(seeds):
                        printTime(f"Gen: {self.current_gen}; Island: {island_id + 1}; Population: {i + 1}; Task: {task_id + 1}/{self.tests_per_gen}")
                        tasks.append((island_id, i, p, seed))
                else: #qilu 03/27/2016 avoid recompute
                    island.fitness[i] = island.prev_fitness[island.not_evolved_idx[i]] * self.tests_per_gen
                    logging.info("island %d pop %d partial fitness = %g", island_id, i, island.prev_fitness[island.not_evolved_idx[i]])

        if tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {}
                for task_id, (island_id, i, p, seed) in enumerate(tasks):
                    future = executor.submit(self._evaluate_fitness_task, p, seed)
                    future_map[future] = (island_id, i, task_id)
                for future in as_completed(future_map):
                    island_id, i, task_id = future_map[future]
                    try:
                        score, fitness_line = future.result()
                        printTime(f"Completed - Gen: {self.current_gen}; Island: {island_id + 1}; Pop: {i + 1}; Task: {task_id + 1}/{len(tasks)}; Result: {fitness_line}")
                    except Exception:
                        logging.exception("Fitness task failed for island %d pop %d", island_id, i)
                        score = 0
                    self.islands[island_id].fitness[i] += score

        # use average fitness as fitness
        for island_id, island in enumerate(self.islands):
            for i in range(len(island.fitness)):
                logging.info("island %d pop %d total fitness = %g", island_id, i, island.fitness[i])
                island.fitness[i] /= self.tests_per_gen
                logging.info("island %d pop %d avg fitness = %g", island_id, i, island.fitness[i])

            # (4/19/2026) Charles Galperin
            # To avoid conflicts with identical fitness scores an alternate method of sorting will be implemented
            fit_pop_index = range(len(island.fitness))
            sorted_fit_pop_index = sorted(fit_pop_index, key=lambda x: island.fitness[x], reverse=True)
            fitpop = [(island.fitness[i], island.population[i], island.not_evolved_count[i]) for i in sorted_fit_pop_index]
            island.fitness, island.population, island.not_evolved_count = map(list, zip(*fitpop))

        seed_for_save = seeds[-1] if seeds else 0
        self.save_population(seed_for_save)

        for island in self.islands:
            island.prev_population = copy.deepcopy(island.population)
            island.prev_fitness = copy.deepcopy(island.fitness) #qilu 03/27
            island.prev_not_evolved_count = copy.deepcopy(island.not_evolved_count) #qilu 04/02

        self.check_termination() #qilu 01/21/2016 add this function
        self.population_data = [] # qilu 01/21/2016 reset it

        # Add elites and generate offspring per island
        for island in self.islands:
            island.not_evolved_idx = [] #qilu 03/27/2016
            island.not_evolved_count = [] #qilu 04/02/2016
            island.population = []
            for i in range(island.elites):
                island.population.append(island.prev_population[i])
                island.not_evolved_idx.append(i)
                island.not_evolved_count.append(island.prev_not_evolved_count[i] + 1)

            num_newOffSpring = island.pop_size - island.elites
            count = 0
            for i in range(num_newOffSpring):
                if count == num_newOffSpring:
                    break
                p1c = np.random.choice(len(island.prev_population), 2)
                p2c = np.random.choice(len(island.prev_population), 2)
                if p1c[0] <= p1c[1]:
                    parent1 = island.prev_population[p1c[0]]
                    idx1 = p1c[0]
                else:
                    parent1 = island.prev_population[p1c[1]]
                    idx1 = p1c[1]

                if p2c[0] <= p2c[1]:
                    parent2 = island.prev_population[p2c[0]]
                    idx2 = p2c[0]
                else:
                    parent2 = island.prev_population[p2c[1]]
                    idx2 = p2c[1]
                if parent1 != parent2: #qilu 03/26/2016
                    children = argos_util.uniform_crossover(self.xml_file, parent1, parent2, 0.5, self.system) # qilu 03/07/2016 add the crossover rate p
                else:
                    children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
                for child in children:
                    argos_util.mutate_parameters(child, self.mut_rate)
                    island.population.append(child)
                    if argos_util.get_parameters(parent1) == argos_util.get_parameters(child):
                        island.not_evolved_idx.append(idx1)
                        island.not_evolved_count.append(island.prev_not_evolved_count[idx1] + 1)
                    elif argos_util.get_parameters(parent2) == argos_util.get_parameters(child):
                        island.not_evolved_idx.append(idx2)
                        island.not_evolved_count.append(island.prev_not_evolved_count[idx2] + 1)
                    else:
                        island.not_evolved_idx.append(-1)
                        island.not_evolved_count.append(0)
                count += 2
                while count > num_newOffSpring:
                    del island.population[-1]
                    del island.not_evolved_idx[-1]
                    del island.not_evolved_count[-1]
                    count -= 1

        if self.migration_elites > 0 and self.island_count > 1:
            if self.current_gen + 1 >= self.migration_interval and (self.current_gen + 1) % self.migration_interval == 0:
                self._migrate_elites()

        self.current_gen += 1
        # (4/19/2026) Charles Galperin
        # Checkpoint created after the computation of a generation
        self.save_state()
        ###(C.G.)

    def _migrate_elites(self):
        elites_by_island = []
        for island in self.islands:
            elite_count = min(self.migration_elites, len(island.prev_population))
            elites = [copy.deepcopy(island.prev_population[i]) for i in range(elite_count)]
            elites_by_island.append(elites)

        for island_id, island in enumerate(self.islands):
            migrants = []
            for other_id, elites in enumerate(elites_by_island):
                if other_id == island_id:
                    continue
                for elite in elites:
                    migrants.append(copy.deepcopy(elite))

            available = island.pop_size - island.elites
            if available <= 0 or not migrants:
                continue
            if len(migrants) > available:
                migrants = migrants[:available]
            for offset, migrant in enumerate(migrants):
                target_index = island.pop_size - 1 - offset
                if target_index < island.elites:
                    break
                island.population[target_index] = migrant
                island.not_evolved_idx[target_index] = -1
                island.not_evolved_count[target_index] = 0

        logging.info("Migrated %d elites per island", self.migration_elites)

    def check_termination(self):
        upperBounds = [1.0, 1.0, 2.0, 20.0, 1.0, 20.0, 180.0]
        fitness_convergence_rate = 0.95
        diversity_rate=0.035
        if not self.population_data:
            return
        data_keys = sorted(list(argos_util.PARAMETER_LIMITS.keys()) + ["fitness", "seed"])
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
        self.population_data = []
        population_data_csv = []
        for island_id, island in enumerate(self.islands):
            for f, p in zip(island.fitness, island.population):
                data = copy.deepcopy(argos_util.get_parameters(p))
                if 'PrintFinalScore' in data:
                    del data['PrintFinalScore']
                data["fitness"] = f
                data["seed"] = seed
                self.population_data.append(data)
                data_csv = dict(data)
                data_csv["island_id"] = island_id
                population_data_csv.append(data_csv)

        data_keys = sorted(list(argos_util.PARAMETER_LIMITS.keys()) + ["fitness", "seed"])
        csv_keys = data_keys + ["island_id"]
        ###(C.G.)

        with open(os.path.join(save_dir, filename), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(population_data_csv) #qilu 07/27
            
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
    parser.add_argument('--islands', action='store', dest='islands', type=int, help='Number of islands (default: CPU count)')
    parser.add_argument('--max_workers', action='store', dest='max_workers', type=int, help='Max concurrent ARGoS runs (default: CPU count)')
    parser.add_argument('--migration_elites', action='store', dest='migration_elites', type=int, help='Elites per island to share each migration (default: 1)')
    parser.add_argument('--migration_interval', action='store', dest='migration_interval', type=int, help='Global generation interval for migration (default: 1)')
    # (4/19/2026) Charles Galperin
    # flag added for loading checkpoint files
    parser.add_argument('-rf', '--resume_file', action='store', dest='resume_file', help='Path to a .pkl checkpoint file')
    ###(C.G.)
    pop_size = 50
    gens = 150
    elites = 1
    mut_rate = 0.05
    robots = 24  #robots = 16
    tags=256 #qilu 03/26 for naming the output directory
    system = "linux"
    length = 720 # 12 minutes, length is in second. default length = 3600
    tests_per_gen= 10
    terminateFlag = 0
    islands = None
    max_workers = None
    migration_elites = None
    migration_interval = None
    
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

    if args.islands is not None:
        islands = args.islands

    if args.max_workers is not None:
        max_workers = args.max_workers

    if args.migration_elites is not None:
        migration_elites = args.migration_elites

    if args.migration_interval is not None:
        migration_interval = args.migration_interval

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
                resume_file=resume_file,
                islands=islands,
                max_workers=max_workers,
                migration_elites=migration_elites,
                migration_interval=migration_interval) # (4/19/2026) Charles Galperin: checkpoint option added
    start = time.time()
    ga.run_ga()
    stop = time.time()
    printTime('The loaded file is '+ xml_file+' ...')
    printTime('It runs '+str((stop-start)/3600.0)+ ' hours...')
