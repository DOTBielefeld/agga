# AGGA
This is an implementation of AGGA a configurator for anytime portfolios.

The data (instances, scenarios, arguments, wrapper) used for our experiments can be found [here](https://github.com/DOTBielefeld/agga_supplement)


## Installation
Install the requirements
```
pip install -r requirements.txt
```

To run configurations in parallel, we use [ray](https://www.ray.io). We require Ray version 2.3.1. By default, AGGA will make use of all available cores on a machine. To run it on a cluster, we provide a Slurm script that starts Ray before calling AGGA.

To construct the portfolios, Gurobi is needed.




## Running AGGA
### Search
Run AGGA by: 
```
python agga/main.py --scenario_file <path_to_your_scenario> --arg_file <path_to_your_argument_file>
``` 

### Arguments
Examples of how to run AGGA can be found here: [here](https://github.com/DOTBielefeld/agga_supplement)

#### Scenario
The scenario file should contain the following arguments:
+ `cutoff_time`: Time the target algorithm is allowed to run
+ `instance_file`: File containing the paths to the instances
+ `paramfile`: Parameter file of the target algorithm in PCS format
+ `wallclock_limit`: Time AGGA is allowed to run


#### Run target algorithm
To run the target algorithm, a wrapper is needed that, given the configuration, instance, and seed, returns a command string calling the target algorithm:
+ `wrapper_mod_name`: Path to the wrapper file. The wrapper file will be imported by AGGA.
+ `wrapper_class_name`: The class in wrapper_mod_name that should be called by AGGA to generate the command for the target algorithm.
+ `quality_match`: Regex that signals AGGA that a solution can be found in the output line.
+ `quality_extract`: Regex that extracts the solution quality from the previously matched line.
+ `memory_limit`: Memory limit for a target algorithm run



#### Generell
+ `seed`: Seed to be used by AGGA
+ `log_folder`: Path to a folder where AGGA should write the logs and results
+ `num_cpu`: Number of CPUs to be used
+ `localmode`: If set to true, Ray will start in local mode, i.e., not look for a running Ray instance in the background
+ `termination_criterion` [total_runtime, total_tournament_number]: Stopping criteria used by AGGA
+ `total_tournament_number` [optional]: Number of AGGA iterations if `termination_criterion` is `total_tournament_number`
+ `use_ggapp`: Create new configurations with the GGA++ model or by genetic engineering
+ `gga_rand_ratio`: Percentage of random configurations to use with the GGA++ model
+ `capping` [cost_sensitive, cap_opt, None]: Type of capping mechanism
+ `cap_model` [rf, rp, xgb]: Type of capping models to be used with cost-sensitive capping
+ `cap_start`: Time when the capping should start in seconds
+ `retrain_capping`: Whether to retrain or not the cost-sensitive capping
+ `rf` [optional]: Path to a JSON file containing the reference quality for the instances in the instance set

#### Instance set
+ `initial_instance_set_size`: Size of the starting instance set used
+ `instance_increment_size`: Number of instances that should be added in an iteration to the instance set
+ `target_reach` [optional]: Approximation of the iteration by which the full instance set should be used
+ `time_instance_set_full`: Time in seconds AGGA should use the full instance set. This may help with obtaining trajectories for all current configurations to be used in the portfolio construction

### Portfolio Construction
After the search phase of AGGA, you can construct the final portfolios by calling:

```
python agga/portfolio_construction.py --agga_dir <directory_of_AGGA_run> --scenario_path <path_to_scenario_of_AGGA_run> --T <time_horizon_of_portfolio> --confs_portfolio <number_of_confs_to_include_in_portfolio> --confs_per_front <max_number_of_points_a_trajectory_can_contribute> --restrict_confs <restrict_number_of_confs_for_mip> --conf_restriction <number_of_confs_to_consider_in_mip> --gurobi_time_limit <timelimit_for_mip>
``` 

The portfolio will be saved in the directory of the AGGA run.











