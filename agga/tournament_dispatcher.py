import numpy as np
import uuid
import copy

from pool import Tournament
from tournament_performance import get_instances_no_results

class MiniTournamentDispatcher:

    def init_tournament(self, results, configurations, instance_partition, instance_partition_id):
        """
        Create a new tournament out of the given configurations and list of instances.
        :param results: Results cache.
        :param configurations: List. Configurations for the tournament
        :param instance_partition: List. List of instances
        :param instance_partition_id: Id of the instance set.
        :return: Tournament, first conf/instance assignment to run
        """

        # Get the configurations that has seen the most instances before
        conf_instances_ran = []
        most_run_conf = []
        for conf in configurations:
            if conf.id in list(results.keys()):
                conf_instances_ran = list(results[conf.id].keys())
                most_run_conf.append(conf)

        # Get instances the conf with the most runs has not been run on before
        possible_first_instances = [i for i in instance_partition if i not in conf_instances_ran]

        # If there are instances the conf with the most runs has not seen we select on of them to be the first instance
        # all confs should be run on
        if possible_first_instances:
            first_instance = np.random.choice(possible_first_instances)
            initial_instance_conf_assignments = [[conf, first_instance] for conf in configurations]
            best_finisher = []
        # An empty list of possible instances means that the conf with the most runs has seen all instances in the
        # instance set. In that case we can choose any instance for the confs that have not seen all instances.
        # We also have a free core then to which we assign a extra conf/instance pair where both are chosen at random
        else:
            amount_of_assignments_needed = len(configurations)

            # confs that I can use for the assigemnt
            configurations_not_run_on_all = configurations
            for conf in most_run_conf:
                configurations_not_run_on_all.remove(conf)

            # We do one assigment off all possible confs with the same instance and then fill the rest slots...
            first_instance = np.random.choice(instance_partition)
            initial_instance_conf_assignments = [[conf, first_instance] for conf in configurations_not_run_on_all]

            # Fill the rest slots
            # need to choose amount_of_assignments_needed - len(initial_instance_conf_assignments) conf/instance assihemtns
            # that we have not yet choosen.
            extra_instances = copy.deepcopy(instance_partition)
            extra_instances.remove(first_instance)

            if amount_of_assignments_needed - len(initial_instance_conf_assignments) > len(extra_instances ) * len(configurations_not_run_on_all):
                raise Exception('To fill all cpu slot we do not have enough conf/instance possebilites')

            while len(initial_instance_conf_assignments) < amount_of_assignments_needed:
                extra_instance = np.random.choice(extra_instances)
                extra_confs = np.random.choice(configurations_not_run_on_all)
                extra_assignmet = [extra_confs, extra_instance]
                if extra_assignmet not in initial_instance_conf_assignments:
                    initial_instance_conf_assignments.append(extra_assignmet)

            best_finisher = most_run_conf

        configuration_ids = [c.id for c in configurations] + [b.id for b in best_finisher if len(best_finisher) >= 1]

        return Tournament(uuid.uuid4(), best_finisher, [], configurations, configuration_ids, {}, instance_partition,
                          instance_partition_id), \
               initial_instance_conf_assignments


    def update_tournament(self, results, tasks, finished_conf, tournament):
        """
        Given a finishing conf we update the tournament if necessary. I.e the finishing conf has seen all instances of
        the tournament. In that case, it is moved either to the best or worst finishers. best finishers are ordered.
        Worst finishers are not
        :param results: Ray cache object.
        :param finished_conf: Configuration that finished or was canceled
        :param tournament: Tournament the finish conf was a member of
        :param number_winner: Int that determines the number of winners per tournament
        :return: updated tournament, stopping signal
        """
        evaluated_instances = results[finished_conf.id].keys()

        # We figure out if there are still tasks the finished configuration is still running on for which we have a results
        # but have not returned through a ray.wait()
        still_running_task_for_conf = [sr for sr in tasks if sr in list(tournament.ray_object_store[finished_conf.id].values())]

        # A conf can only become a best finisher if it has seen all instances of the tournament and is not running any
        # other conf/instance pairs. i.e the result we process here is the last one
        if set(evaluated_instances) == set(tournament.instance_set) and len(still_running_task_for_conf) == 0:
            # We can than remove the conf from further consideration
            if finished_conf in tournament.configurations:
                tournament.configurations.remove(finished_conf)
            else:
                raise Exception(f"Messed up the confs in the tournaments")
            tournament.best_finisher.append(finished_conf)

        # If there are no configurations left we end the tournament
        if len(tournament.configurations) == 0:
            stop = True
        else:
            stop = False

        return tournament, stop

    def next_tournament_run(self, results, tournament, finished_conf):
        """
        Decided which conf/instance pair to run next. Rule: If the configuration that has just finished was not killed
        nor saw all instances, it is assigned a new instance at random. Else, the configuration with the lowest runtime
        so far is selected.
        :param results: Ray cache
        :param tournament: The tournament we opt to create a new task for
        :param finished_conf: Configuration that just finished before
        :return: configuration, instance pair to run next
        """
        next_possible_conf = {}

        # For each conf still in the running we need to figure out on which instances it already ran or is still
        # running on to get for each conf the instances it still can run on
        for conf in tournament.configurations:
            already_run = get_instances_no_results(results, conf.id, tournament.instance_set)

            not_running_currently = get_instances_no_results(tournament.ray_object_store, conf.id,
                                                             tournament.instance_set)
            not_running_currently = [c for c in not_running_currently if c in already_run]

            if len(not_running_currently) > 0:
                next_possible_conf[conf.id] = not_running_currently
        # If there are no configuration that need to see new instances we create a dummy task to give the still running
        # conf/instance pairs time to finish.
        if len(next_possible_conf) == 0:
            configuration = None
            next_instance = None
        else:
        # If the previous run conf has not seen all instances and did not time out it is selected to run again
            if finished_conf.id in list(next_possible_conf.keys()):
                next_conf_id = finished_conf.id
            else:
                next_conf_id = np.random.choice(list(next_possible_conf.keys()))

            configuration = [c for c in tournament.configurations if c.id == next_conf_id][0]
            next_possible_instance = next_possible_conf[next_conf_id]
            next_instance = np.random.choice(next_possible_instance)

        return [[configuration, next_instance]]




