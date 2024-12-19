import ray
import logging
import time
import numpy as np

def calculateArea(trajectory, finalEffort, shift=None):
    """
    This is from https://github.com/souzamarcelo/capopt
    """
    lastEffort = trajectory[0][0]
    lastValue = trajectory[0][1]
    area = 0
    for point in trajectory[1:]:
        if point[0] <= finalEffort:
            x = point[0] - lastEffort
            y = (lastValue + shift)
            area += (x * y)
            lastEffort = point[0]
            lastValue = point[1]
    x = finalEffort - lastEffort
    y = (lastValue + shift)
    area += (x * y)
    return area

@ray.remote(num_cpus=1)
class CAPOPT_Monitor:
    def __init__(self, sleep_time, cache, scenario):
        """
        :param sleep_time: Int. Wake up and check whether runtime is exceeded
        :param cache: Ray cache
        :param number_of_finisher: Int.
        :return: conf/instance that are killed.
        """
        self.sleep_time = sleep_time
        self.cache = cache
        self.time_out = scenario.cutoff_time
        self.scenario = scenario
        self.kill_store = {}
        self.results_store = {}

        logging.basicConfig(filename=f'{scenario.log_folder}/monitor.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def monitor(self):
        logging.info("Starting monitor")

        while True:
            # instead of an elite set we use the confs of the last pool/iteration
            last_pool = ray.get(self.cache.get_last_tournament.remote())
            if last_pool:
                # only get the new intermediate results
                new_feedback = {}
                for conf in last_pool.configuration_ids:
                    if conf not in new_feedback:
                        new_feedback[conf] = {}

                    for i in last_pool.instance_set:
                        r = ray.get(self.cache.get_intermediate_single.remote(conf, i))
                        new_feedback[conf][i] = r

                start = time.time()
                quality_lb = {}
                quality_ub = {}
                # get the quality bounds to calc the area
                for c, instance_dic in new_feedback.items():
                    for i, t in instance_dic.items():
                        if len(t) != 0:
                            if i not in quality_lb.keys():
                                quality_lb[i] = t[-1][1]
                                quality_ub[i] = t[0][1]
                            else:
                                if quality_lb[i] > t[-1][1]:
                                    quality_lb[i] = t[-1][1]
                                if quality_ub[i] < t[0][1]:
                                    quality_ub[i] = t[0][1]

                # calculate the ref area we compare current trajectories against
                area_ref = {}
                for c, instance_dic in new_feedback.items():
                    for i, t in instance_dic.items():
                        # only get area if conf is finished on that instance
                        r = ray.get(self.cache.get_results_single.remote(c, i))
                        if r or np.isnan(r):
                            if len(t) != 0:
                                t.insert(0, [0, quality_ub[i]])
                                area = calculateArea(t, self.time_out, -quality_lb[i])
                                if i not in area_ref.keys():
                                    area_ref[i] = area
                                elif area < area_ref[i]:
                                    area_ref[i] = area

                # Get the current tournaments that are in the cache
                tournaments = ray.get(self.cache.get_tournament.remote())

                for t in tournaments:
                    if t.instance_set_id != -1 and len(t.ray_object_store) != 0:
                        for conf in t.configurations:
                            # Here we figured out which instances the conf is still running and which one it already finished
                            instances_conf_finished = []
                            for i in t.instance_set:
                                r = ray.get(self.cache.get_results_single.remote(conf.id, i))
                                if r or np.isnan(r):
                                    instances_conf_finished.append(i)

                            instances_conf_planned = list(t.ray_object_store[conf.id].keys())
                            instances_conf_still_runs = [c for c in instances_conf_planned if c not in instances_conf_finished]

                            for instance in instances_conf_still_runs:
                               killed = self.kill_store.get(conf.id, {}).get(instance, False)

                               if killed == False and instance in area_ref.keys():
                                    trajectory_sofar = ray.get(self.cache.get_intermediate_single.remote(conf.id, instance))
                                    if trajectory_sofar:
                                        # insert artificial starting point so all trajectories start at 0
                                        # if we would not do this we would get a smaller area for trajectories that start later.
                                        # but this is not what we want. For a good anytime behaviour you want solutions fast..
                                        trajectory_sofar.insert(0, [0, max(trajectory_sofar[0][1], quality_ub[instance])])
                                        area = calculateArea(trajectory_sofar, trajectory_sofar[-1][0], -quality_lb[instance])
                                        logging.info(f"Capping check {conf.id} {instance} {area} {area_ref}")

                                        if area > area_ref[instance]:
                                            print("CAPPING",conf.id, instance, area, area_ref, -quality_lb[instance])
                                            print(trajectory_sofar)
                                            self.cache.put_termination_history.remote(conf.id, instance)
                                            [ray.cancel(t.ray_object_store[conf.id][instance])]

                                            if conf.id not in self.kill_store.keys():
                                                self.kill_store[conf.id] = {}
                                            self.kill_store[conf.id] = {instance: True}

            time.sleep(self.sleep_time)

    def termination_check(self, conf_id, instance, termination_history):
        """
        Check if we have killed a conf/instance pair already. Return True if we did not.
        :param conf_id:
        :param instance:
        :return:
        """
        if conf_id not in termination_history:
            return True
        elif instance not in termination_history[conf_id]:
            return True
        else:
            return False







