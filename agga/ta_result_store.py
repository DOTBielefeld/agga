import ray
import logging
import json
from log_setup import TournamentEncoder


@ray.remote(num_cpus=1)
class TargetAlgorithmObserver:

    def __init__(self, scenario):
        self.intermediate_output = {}
        self.results = {}
        self.start_time = {}
        self.tournament_history = {}
        self.termination_history = {}
        self.tournaments = {}
        self.scenario = scenario
        self.last_tournament = None

        logging.basicConfig(filename=f'{self.scenario.log_folder}/Target_Algorithm_Cache.logger', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def put_intermediate_output(self, conf_id, instance_id, value):
        logging.info(f"Getting intermediate_output: {conf_id}, {instance_id}, {value} ")

        if conf_id not in self.intermediate_output:
            self.intermediate_output[conf_id] = {}

        if instance_id not in self.intermediate_output[conf_id]:
            self.intermediate_output[conf_id][instance_id] = []

        if value is not None:
            self.intermediate_output[conf_id][instance_id] = self.intermediate_output[conf_id][instance_id] + [value]

    def get_intermediate_output(self):
        return self.intermediate_output

    def get_intermediate_single(self, conf_id, instance_id):
        result = False
        if conf_id in list(self.intermediate_output.keys()):
            if instance_id in list(self.intermediate_output[conf_id].keys()):
                result = self.intermediate_output[conf_id][instance_id]
        return result


    def put_result(self, conf_id, instance_id, result):
        logging.info(f"Getting final result: {conf_id}, {instance_id}, {result}")
        if conf_id not in self.results:
            self.results[conf_id] = {}

        if instance_id not in self.results[conf_id]:
            self.results[conf_id][instance_id] = result

    def get_results(self):
        logging.info(f"Publishing results")
        return self.results

    def get_results_single(self, conf_id, instance_id):
        result = False
        if conf_id in list(self.results.keys()):
            if instance_id in list(self.results[conf_id].keys()):
                result = self.results[conf_id][instance_id]
        return result

    def put_start(self,conf_id, instance_id, start):
        logging.info(f"Getting start: {conf_id}, {instance_id}, {start} ")
        if conf_id not in self.start_time:
            self.start_time[conf_id] = {}

        if instance_id not in self.start_time[conf_id]:
            self.start_time[conf_id][instance_id] = start

    def get_start(self):
        logging.info(f"Publishing start")
        return self.start_time

    def get_start_single(self, conf_id, instance_id):
        result = False
        if conf_id in list(self.start_time.keys()):
            if instance_id in list(self.start_time[conf_id].keys()):
                result = self.start_time[conf_id][instance_id]
        return result

    def put_tournament_history(self, tournament):
        self.tournament_history[tournament.id ] = tournament
        self.last_tournament = tournament

    def get_tournament_history(self):
        return self.tournament_history

    def get_last_tournament(self):
        return self.last_tournament

    def put_tournament_update(self, tournament):
        self.tournaments[tournament.id] = tournament

    def remove_tournament(self,tournament):
        self.tournaments.pop(tournament.id)

    def get_tournament(self):
        return list(self.tournaments.values())

    def put_termination_history(self, conf_id, instance_id):
        if conf_id not in self.termination_history:
            self.termination_history[conf_id] = []

        if instance_id not in self.termination_history[conf_id]:
            self.termination_history[conf_id].append(instance_id)
        else:
            logging.info(f"This should not happen: we kill something we already killed")

    def get_termination_history(self):
        return self.termination_history

    def get_termination_single(self, conf_id, instance_id):
        termination = False
        if conf_id in list(self.termination_history.keys()):
            if instance_id in list(self.termination_history[conf_id]):
                termination = True
        return termination

    def save_rt_results(self):
        with open(f"{self.scenario.log_folder}/run_history.json", 'a') as f:
            history = {str(k):v for k,v in self.results.items()}
            json.dump(history, f, indent=2)

    def save_trajectory(self):
        with open(f"{self.scenario.log_folder}/run_history_trajectory.json", 'a') as f:
            history = {str(k):v for k,v in self.intermediate_output.items()}
            json.dump(history, f, indent=2)

    def save_tournament_history(self):
        with open(f"{self.scenario.log_folder}/tournament_history.json", 'a') as f:
            history = {str(k): v for k, v in self.tournament_history.items()}
            json.dump(history, f, indent=4, cls=TournamentEncoder)

