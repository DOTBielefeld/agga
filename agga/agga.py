
import numpy as np
import ray
import time
import json

from ta_result_store import TargetAlgorithmObserver

from generators.agga_conf_generation import ANYTIMEGGA
from capping.cost_sensetive_monitor import CostSensitiveMonitor
from capping.cap_opt_monitor import CAPOPT_Monitor

from tournament_dispatcher import MiniTournamentDispatcher
from tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, termination_check

from instance_sets import InstanceSet



def agga(scenario, ta_wrapper, logger):
    global_cache = TargetAlgorithmObserver.remote(scenario)

    if scenario.capping == "cost_sensitive":
        monitor = CostSensitiveMonitor.remote(1, global_cache, scenario)
        monitor.monitor.remote()
        scenario.num_cpu = scenario.num_cpu -1
    elif scenario.capping == "cap_opt":
        monitor = CAPOPT_Monitor.remote(1, global_cache, scenario)
        monitor.monitor.remote()
        scenario.num_cpu = scenario.num_cpu - 1

    instance_selector = InstanceSet(scenario.instance_set, scenario.initial_instance_set_size, set_size=len(scenario.instance_set),
                                    target_reach=scenario.target_reach, instance_increment_size=scenario.instance_increment_size)

    # load ref points if any
    if scenario.rf:
        with open(scenario.rf) as f:
            ref_points = f.read()
        ref_points = json.loads(ref_points)
        for k, v in ref_points.items():
            ref_points[k] = np.array([scenario.cutoff_time, float(v)]) * 1.1
    else:
        ref_points = {}

    gga_generator = ANYTIMEGGA(scenario, scenario.num_cpu, ref_point=ref_points, use_model=scenario.use_ggapp)

    tournament_dispatcher = MiniTournamentDispatcher()
    tasks = []
    tournaments = []
    results = ray.get(global_cache.get_results.remote())

    # creating the first tournaments and adding first conf/instance pairs to ray tasks
    points_to_run = gga_generator.return_conf([],[],{})
    instance_id, instances = instance_selector.get_subset(0)
    tournament, initial_assignments = tournament_dispatcher.init_tournament(results, points_to_run,instances, instance_id)

    tournaments.append(tournament)
    global_cache.put_tournament_update.remote(tournament)
    tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    main_loop_start = time.time()
    tournament_counter = 0
    terminate = False

    while terminate is False:

        # get some ta feedback
        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
            result_conf, result_instance, cancel_flag = result[0], result[1], result[2]

        # sometimes a ray worker may crash. We handel that here. I.e if the TA did not run to the end, we reschedule
        except (ray.exceptions.WorkerCrashedError, ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError) as e:
            logger.info(f'Crashed TA worker, {time.ctime()}, {winner}, {e}')
            # Figure out which tournament conf. belongs to
            for t in tournaments:
                conf_instance = get_tasks(t.ray_object_store, winner)
                if len(conf_instance) != 0:
                    tournament_of_c_i = t
                    break
            conf = [conf for conf in tournament_of_c_i.configurations if conf.id == conf_instance[0][0]][0]
            instance = conf_instance[0][1]

            # We check if we have killed the conf and only messed up the termination of the process
            termination_check_c_i = ray.get(global_cache.get_termination_single.remote(conf.id , instance))
            if termination_check_c_i:
                result_conf = conf
                result_instance = instance
                cancel_flag = True
                global_cache.put_result.remote(result_conf.id, result_instance, np.nan)
                logger.info(f"Canceled task with no return: {result_conf}, {result_instance}")
                print(f"Canceled task with no return: {result_conf}, {result_instance}")
            else:  # got no results: need to rescheulde
                next_task = [[conf, instance]]
                tasks = update_tasks(tasks, next_task, tournament_of_c_i, global_cache, ta_wrapper, scenario)
                logger.info(f"We have no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
                print(f"We have no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
                continue

        # get the results of the just finished run
        if result_conf.id in list(results.keys()):
            results[result_conf.id][result_instance] = ray.get(
                global_cache.get_results_single.remote(result_conf.id, result_instance))
        else:
            results[result_conf.id] = {}
            results[result_conf.id][result_instance] = ray.get(
                global_cache.get_results_single.remote(result_conf.id, result_instance))

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        result_tournament = get_tournament_membership(tournaments, result_conf)
        if cancel_flag:
            if result_conf.id in result_tournament.ray_object_store.keys():
                if result_instance in result_tournament.ray_object_store[result_conf.id].keys():
                    if result_tournament.ray_object_store[result_conf.id][result_instance] in tasks:
                        tasks.remove(result_tournament.ray_object_store[result_conf.id][result_instance])

        # Update the tournament based on result
        result_tournament, tournament_stop = tournament_dispatcher.update_tournament(results, tasks, result_conf, result_tournament,)


        # check if we are done
        if termination_check(scenario.termination_criterion, main_loop_start, scenario.wallclock_limit,
                             scenario.total_tournament_number, tournament_counter) == False:
            break

        # the tournament finished
        if tournament_stop:
            print("Iteration:", time.time() - main_loop_start, tournament_counter)
            tournament_counter += 1

            # Remove that old tournament
            tournaments.remove(result_tournament)
            global_cache.put_tournament_history.remote(result_tournament)

            # In case we are almost done we allow for last iterations on the whole instance set to get good information
            # for the portfolio construction
            if time.time() - main_loop_start > scenario.wallclock_limit - scenario.time_instance_set_full:
                instance_id, instances = result_tournament.instance_set_id + 1 , scenario.instance_set
                # no monitor for the last full instances sets to get full feedback
                ray.kill(monitor)
            else:             # Get the instances for the new tournament
                instance_id, instances = instance_selector.get_subset(result_tournament.instance_set_id + 1)

            # get the results
            trajectories = ray.get(global_cache.get_intermediate_output.remote())
            terminations = ray.get(global_cache.get_termination_history.remote())

            # gen new confs
            points_to_run = gga_generator.return_conf(result_tournament, trajectories, terminations)

            # Create new tournament with new confs and instances
            new_tournament, initial_assignments_new_tournament = tournament_dispatcher.init_tournament(results,
                                                                                                       points_to_run,
                                                                                                       instances,
                                                                                                       instance_id)
            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            tournaments.append(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache, ta_wrapper,
                                 scenario)

            global_cache.put_tournament_update.remote(new_tournament)
            global_cache.remove_tournament.remote(result_tournament)

            logger.info(f"Final results tournament {result_tournament}")
            logger.info(f"New tournament {new_tournament}")
        else:
            # If the tournament does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = tournament_dispatcher.next_tournament_run(results, result_tournament, result_conf)
            tasks = update_tasks(tasks, next_task, result_tournament, global_cache, ta_wrapper, scenario)
            #logger.info(f"New Task {next_task}, {result_tournament}")
            global_cache.put_tournament_update.remote(result_tournament)

    # safe some stuff in the end
    global_cache.save_rt_results.remote()
    global_cache.save_tournament_history.remote()
    global_cache.save_trajectory.remote()

    logger.info("DONE")
    time.sleep(30)
    [ray.cancel(t) for t in not_ready]




