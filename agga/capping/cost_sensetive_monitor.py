import ray
import logging
import time
import numpy as np
import json
from .cappingmodel import CappingModel

@ray.remote(num_cpus=1)
class CostSensitiveMonitor:
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
        self.cutoff = scenario.cap_start
        self.train_counter = 0
        self.instances_with_results = []
        self.retrain = scenario.retrain_capping

        logging.basicConfig(filename=f'{scenario.log_folder}/monitor.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def train_model(self):
        # train the capping model
        self.monitor_model = CappingModel(self.scenario)

        run_data = ray.get(self.cache.get_intermediate_output.remote())
        self.train_data = {}

        # only train on noncapped runs
        for conf, instance_trajectories in run_data.items():
            for instance, traj in instance_trajectories.items():
                killed = self.kill_store.get(conf, {}).get(instance, False)

                if killed == False:
                    if conf not in self.train_data.keys():
                        self.train_data[conf] = {}
                    self.train_data[conf][instance] = traj

                    # get instances that have at least one proper result
                    if len(traj) != 0 and instance not in self.instances_with_results:
                        self.instances_with_results.append(instance)

        self.monitor_model.fit(self.train_data, self.cutoff)

        with open(f"{self.scenario.log_folder}/model_train_data_{self.train_counter}.json", 'a') as f:
            history = {str(k):v for k,v in self.train_data.items()}
            json.dump(history, f, indent=2)

        self.train_counter = self.train_counter + 1

    def monitor(self):
        logging.info("Starting monitor")

        # before we train the capping model we need data so we wait...
        time.sleep(3600)
        logging.info("Training model")
        self.train_model()
        time_since_last_train = time.time()

        while True:
            # Get the current tournaments that are in the cache
            tournaments = ray.get(self.cache.get_tournament.remote())

            for t in tournaments:
                if t.instance_set_id != -1:
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
                            start_time = ray.get(self.cache.get_start_single.remote(conf.id, instance))
                            if start_time:
                                conf_runtime = time.time() - start_time
                            else:
                                continue

                            killed = self.kill_store.get(conf.id, {}).get(instance, False)

                            if conf_runtime >= self.cutoff and killed == False and instance in self.instances_with_results:
                                trajectory_sofar = ray.get(self.cache.get_intermediate_single.remote(conf.id, instance))
                                trajectory_sofar = {conf.id: {instance: trajectory_sofar}}
                                cutoff_prediciton = self.monitor_model.predict_batch(trajectory_sofar, self.cutoff)

                                logging.info(f"{trajectory_sofar} {cutoff_prediciton}")
                                if cutoff_prediciton[conf.id][instance] == 0:
                                    self.cache.put_termination_history.remote(conf.id, instance)
                                    [ray.cancel(t.ray_object_store[conf.id][instance])]

                                    if conf.id not in self.kill_store.keys():
                                        self.kill_store[conf.id] = {}
                                    self.kill_store[conf.id] = {instance: True}

            if time.time() - time_since_last_train > 3600 * 2 and self.retrain:
                logging.info("Retrain monitor")
                self.train_model()
                time_since_last_train = time.time()

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







