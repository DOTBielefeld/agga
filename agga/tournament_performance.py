import numpy as np
import copy


def get_conf_time_out(results, configuration_id, instances_set):
    """
    Determine if a configuration timed out on any instance in a set
    :param results:  dic. Run results of configuration
    :param configuration_id: int.
    :param instances_set: Set of instances.
    :return: None if no results are present, True if conf timed out on any instance, False if it dit not.
    """
    if configuration_id in list(results.keys()):
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]
        if np.isnan(conf_results_instances).any():
            return True
        else:
            return False
    else:
        return None


def get_censored_runtime_for_instance_set(results, configuration_id, instances_set):
    """
    For a configuration compute the total runtime needed only for instances in a set. If there are no results for the
    conf return 0. Note that runs that were canceled by the monitor are not included since we count them as nan's
    :param results: Dic of results: {conf_id: {instance: runtime}}
    :param configuration_id: Id of the configuration
    :param instances_set: List of instances
    :return: Runtime of the configuration on instances
    """
    if configuration_id in results.keys():
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]
        runtime = np.nansum(list(conf_results_instances))

    return runtime

def get_runtime_for_instance_set_with_timeout(results, configuration_id, instances_set, timeout, par_penalty=1):
    """
    For a configuration compute the total runtime needed only for instances in a set. If there are no results for the
    conf return 0. Note that runs that were canceled by the monitor are not included since we count them as nan's
    :param results: Dic of results: {conf_id: {instance: runtime}}
    :param configuration_id: Id of the configuration
    :param instances_set: List of instances
    :return: Runtime of the configuration on instances
    """
    if configuration_id in results.keys():
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]

        runtime = np.nansum(list(conf_results_instances)) + np.count_nonzero(np.isnan(list(conf_results_instances))) * (timeout * par_penalty)

    return runtime


def get_censored_runtime_of_configuration(results, configuration_id):
    """
    Get total runtime of a conf not conditioned on an instance set. Note that runs that were canceled by the monitor
    are not included since we count them as nan's
    :param results: Dic of results: {conf_id: {instance: runtime}}
    :param configuration_id: Id of the configuration
    :return:
    """
    if configuration_id in results.keys():
        conf_results = results[configuration_id]
        runtime = np.nansum(list(conf_results.values()))
    return runtime



def get_instances_no_results(results, configuration_id, instance_set):
    """
    For a configuration get a list of instances we have no results for yet
    :param results: Dic of results: {conf_id: {instance: runtime}}
    :param configuration_id: Id of the configuration
    :param instance_set: List of instances
    :return: List of configuration the conf has not been run on
    """
    not_run_on= copy.deepcopy(instance_set)

    if configuration_id in results.keys():
        configuration_results = results[configuration_id]

        instances_run_on = configuration_results.keys()

        for iro in instances_run_on:
            if iro in not_run_on:
                not_run_on.remove(iro)

    return not_run_on

def compute_pareto_front(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return costs
