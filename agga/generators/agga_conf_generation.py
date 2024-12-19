import copy
import sys
import logging
import gurobipy as gp
from gurobipy import GRB
import uuid
import numpy as np
from pymoo.indicators.hv import HV

from agga.pool import Tournament, Configuration
from agga.generators.random_point_generator import random_point, reset_conditionals
from agga.generators.default_point_generator import default_point, check_no_goods, check_conditionals
from agga.generators.variable_graph_point_generator import variable_graph_structure, graph_crossover
from agga.generators.ggapp_surrogate import GGAppSurr
from agga.generators.mip_help import calc_area_for_front, conf_area_map, get_number_points_conf
from agga.tournament_performance import compute_pareto_front

np.set_printoptions(threshold=sys.maxsize)



class ANYTIMEGGA:
    def __init__(self, scenario, competitive_pool_size, ref_point=None, mutation_prob=0.2, random_prob=0.1, discard_portion=0.3, use_model=False): #0.3

        self.scenario = scenario
        self.mutation_prob = mutation_prob
        self.random_prob = random_prob
        self.t = 0
        self.discard_portion = discard_portion
        self.competitive_pool_size = competitive_pool_size

        self.competitive_pool = [random_point(scenario, uuid.uuid4()) for _ in range(competitive_pool_size -1 )] + [default_point(scenario, uuid.uuid4())]
        self.finisher_pool = []
        self.instances_with_results = []
        self.last_size = 0

        self.hv_store = {}
        self.use_model = use_model

        if self.use_model:

            model_conf = {
                  "bootstrap": True,
                  "max_depth": 10,
                  "max_features": 1.0,
                  "max_features_o": 0.92,
                  "min_gain": 0.41,
                  "min_samples_leaf": 4,
                  "min_samples_split": 6,
                  "n_estimators": 50,
                  "num_pct": 12,
                  "pruned": True,
                  "q": 0.10
                }

            self.gga_sur = GGAppSurr(self.scenario, self.scenario.seed, **model_conf)

        if type(ref_point) is dict:
            self.ref_point = {k: np.array(v) for k, v in ref_point.items()}
        elif ref_point is None:
            self.ref_point = {}


    def get_ref(self, tournament, results):

        for i in tournament.instance_set:
            trajectories_for_instance = {}
            for conf in tournament.configuration_ids:
                if i in results[conf].keys():
                    trajectories_for_instance[conf] = results[conf][i]

            worst_quality = -1
            for tra in trajectories_for_instance.values():
                for point in tra:
                    if point[1] > worst_quality:
                        worst_quality = point[1]

            if worst_quality != -1:
                new_ref_point = np.array([self.scenario.cutoff_time, worst_quality]) * 1.1

                if i not in self.ref_point.keys() or new_ref_point[1] > self.ref_point[i][1]:
                    self.ref_point[i] = new_ref_point
                    logging.info(f"refpoint update {i} {self.ref_point[i]}")
                if i not in self.instances_with_results:
                    self.instances_with_results.append(i)


    def impute_no_result(self, tournament, trajectories):
        # Some runs may fail. We just assign an empty trajectory to be safe.
        for conf in tournament.configuration_ids:
            if conf not in trajectories.keys():
                trajectories[conf] = {}

        for i in tournament.instance_set:
            for conf in tournament.configuration_ids:
                if i not in trajectories[conf].keys():
                   trajectories[conf][i] = []
                   print("imputing something")
        return trajectories


    def select_confs_hv(self, n_to_select , conf_ids, instances, results):

        conf_ids = copy.deepcopy(conf_ids)
        instances = copy.deepcopy(instances)
        trajectories = {}
        pareto_fronts = {}

        if n_to_select < len(instances):
            print("this may fail")

        # Get and transform pareto fronts
        instances_no_result = []
        for instance in instances:

            ref_point_norm = self.ref_point[instance]

            results_for_instance = {conf: results[conf][instance] for conf in conf_ids}

            conf_instance_results_full = np.array([point for conf_r in results_for_instance.values() for point in conf_r])
            pareto_front_instance = compute_pareto_front(conf_instance_results_full, return_mask=False)

            if len(pareto_front_instance) != 0:
                pareto_front_instance_norm = ((pareto_front_instance - 0) / (ref_point_norm - 0))
            else:
                instances_no_result.append(instance)
                continue

            pareto_fronts[instance] = sorted(pareto_front_instance_norm.tolist(), key=lambda x: x[0])
            trajectories[instance] = {conf: ((((np.array(traj) - 0) / (ref_point_norm - 0))).tolist() if len(traj) != 0 else traj) for conf, traj in results_for_instance.items()}

        for instance in instances_no_result:
            instances.remove(instance)

        ref = [[1, 1]]

        w = {}
        pc = {}
        for instance, count in zip(instances, range(1, len(instances) + 1)):
            w[count] = calc_area_for_front(pareto_fronts[instance], ref)
            pc[count] = conf_area_map(pareto_fronts[instance], trajectories[instance])

        # remove conf that are never on the pareto front...
        n_p_confs = get_number_points_conf(pc, conf_ids)
        nc = [c for c, v in n_p_confs.items() if v == 0]
        for c_nc in nc:
            del n_p_confs[c_nc]
            conf_ids.remove(c_nc)

        # if we have less confs to select from than need to be selected we return all confs on the pareto front
        if len(conf_ids) <=  n_to_select:
            selection = conf_ids

            if len(conf_ids) == 0:
                raise Exception("We have nothing to return")
        else: # in this case we have more confs on the pareto front and want to keep only a subset
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()

            model = gp.Model("hv selection", env=env)
            m = len(instances) + 1
            n = [len(f) + 1 for f in pareto_fronts.values()]
            c_confs = len(conf_ids) + 1
            constraints_violation = []

            x_ij = model.addVars(
                [
                    (i, j, k)
                    for k in range(1, m)
                    for i in range(1, n[k - 1])
                    for j in range(1, i + 1)
                ],
                vtype=GRB.BINARY,
                name="x_ijk",
            )

            y_c = model.addVars([y for y in range(1, c_confs)], vtype=GRB.BINARY, name="y_c", )

            # 3
            for k in range(1, m):
                n_k = n[k - 1]
                for i in range(2, n_k):
                    for j in range(1, i):
                        model.addConstr(gp.quicksum(x_ij[r, r, k] for r in range(j, i + 1)) >= x_ij[i, j, k])

            # one point for every instance
            for k in range(1, m):
                n_k = n[k - 1]
                vio = model.addConstr(gp.quicksum(x_ij[r, r, k] for r in range(1, n_k)) >= 1)
                constraints_violation.append(vio)

            for k in range(1, m):
                for r in range(1, n[k - 1]):
                    model.addConstr(x_ij[r, r, k] <= gp.quicksum(
                        y_c[c] * pc[k][r][conf_id] for c, conf_id in zip(range(1, c_confs), conf_ids)))

            model.addConstr(gp.quicksum(y_c[c] for c in range(1, c_confs)) == n_to_select)

            model.setObjective(gp.quicksum(
                w[k][j][i] * x_ij[i, j, k] for k in range(1, m) for i in range(1, n[k - 1]) for j in range(1, i + 1)),
                               GRB.MAXIMIZE)

            model.optimize()

            # map model results back to conf ids
            selection = []
            for v, c in zip(y_c.values(), range(len(y_c))):
                if int(round(v.x)) == 1:
                    selection.append(conf_ids[c])

        logging.info(f"Selection {selection}")
        return selection


    def discard_confs_competitive(self, instances, results):

        current_conf_ids = [conf.id for conf in copy.deepcopy(self.competitive_pool)]

        conf_to_discard = []
        confs_to_keep = []
        n_to_select = int(round(len(self.competitive_pool) * (1 - self.discard_portion)))
        if n_to_select != 0:

            while n_to_select > 0:
                confs_on_front_best_hv = self.select_confs_hv(n_to_select, current_conf_ids, instances, results)

                for conf in confs_on_front_best_hv:
                    current_conf_ids.remove(conf)

                confs_to_keep = confs_to_keep + confs_on_front_best_hv
                n_to_select = n_to_select - len(confs_on_front_best_hv)

                if len(confs_on_front_best_hv) == 0:
                    n_to_select = 0

            for conf in current_conf_ids:
                if conf not in confs_to_keep:
                    conf_to_discard.append(conf)
        logging.info(f"discard {conf_to_discard}")
        return conf_to_discard

    def create_new_conf(self, mean_hv):

        # select parents based on mean hv
        if sum([1 for x in mean_hv.values() if x > 0]) > 1:
            parent_ids = np.random.choice(list(mean_hv.keys()), p=[hv / sum(mean_hv.values()) for hv in mean_hv.values()], size=2, replace=False)
            parents = []
            for conf in self.competitive_pool + self.finisher_pool:
                if conf.id in parent_ids:
                    parents.append(conf)

            parent_1 = parents[0]
            parent_2 = parents[1]
        else:
            parent_1 = random_point(self.scenario, uuid.uuid4())
            parent_2 = random_point(self.scenario, uuid.uuid4())


        no_good = True
        while no_good:

            rn = np.random.uniform()
            if rn < self.random_prob:
                config_setting = random_point(self.scenario, uuid.uuid4()).conf
            else:
                graph_structure = variable_graph_structure(self.scenario)
                config_setting = graph_crossover(graph_structure, parent_1, parent_2, self.scenario)

                possible_mutations = random_point(self.scenario, uuid.uuid4())
                for param, value in config_setting.items():
                    rn = np.random.uniform()
                    if rn < self.mutation_prob:
                        config_setting[param] = possible_mutations.conf[param]

            identity = uuid.uuid4()
            configuration = Configuration(identity, config_setting, [])

            cond_vio = check_conditionals(self.scenario, configuration.conf)
            if cond_vio:
                configuration.conf = reset_conditionals(self.scenario, configuration.conf, cond_vio)

            no_good = check_no_goods(self.scenario, configuration.conf)

        logging.info(f"New {configuration} {parent_1} {parent_2}")
        return configuration


    def transfer_confs_pools(self, results):

        # move confs from one pool to the other
        pool_copy = copy.deepcopy(self.competitive_pool)
        for conf in pool_copy:
             if all(x in results[conf.id].keys() and results[conf.id][x] for x in self.instances_with_results):
                self.competitive_pool.remove(conf)
                self.finisher_pool.append(conf)

        # figure out which confs in the finisher pool are on the pareto front of any instance
        confs_on_front =  []
        for instance in self.instances_with_results:
            results_for_instance = {conf.id: results[conf.id][instance] for conf in  self.finisher_pool}
            conf_instance_results_full = np.array([point for conf_r in results_for_instance.values() for point in conf_r])
            full_pareto_front_on_instance = compute_pareto_front(conf_instance_results_full, return_mask=False)

            for front_point in full_pareto_front_on_instance:
                for conf, trajectory in results_for_instance.items():
                    if front_point.tolist() in trajectory:
                        if conf not in confs_on_front:
                            confs_on_front.append(conf)

        # need to recalc the hv improvement
        conf_ids = [conf.id for conf in self.finisher_pool if conf.id not in confs_on_front]
        _ = self.mean_hypervolume(conf_ids, self.instances_with_results, results)

        # we only keep the confs that have a point on any pareto front in the finisher pool
        self.finisher_pool = [conf for conf in self.finisher_pool if conf.id in confs_on_front]

        logging.info(f"Current finisher pool {self.finisher_pool}")

    def mean_hypervolume(self, conf_ids, instances, results):

        mean_hv = {}
        for instance in instances:
            ref_point = self.ref_point[instance]

            ind_hv = HV(ref_point= [1, 1])
            for conf in conf_ids:

                # this is some shenanigance such that we have not to recompute the hv every time
                conf_instance_hv_obtained = conf in self.hv_store and instance in self.hv_store[conf]

                if conf_instance_hv_obtained:
                    hv = self.hv_store[conf][instance]
                else:
                    conf_instance_results_trajectory = np.array(results[conf][instance])

                    # zero hv in case not solved
                    if len(conf_instance_results_trajectory) == 0:
                        hv = 0
                    else:
                        conf_instance_pareto_front = compute_pareto_front(conf_instance_results_trajectory, return_mask=False)
                        conf_instance_pareto_front_norm = (conf_instance_pareto_front - 0) / (ref_point - 0)
                        hv = ind_hv(conf_instance_pareto_front_norm)

                    self.hv_store.setdefault(conf, {})[instance] = hv

                mean_hv.setdefault(conf, []).append(hv)

        return {k: sum(v)/(len(instances)) for k,v in mean_hv.items()}


    def return_conf(self, tournament, trajectories, terminations):

        self.t = self.t + 1

        logging.info(f"ref {self.ref_point}")

        if self.t > 1:
            logging.info(f"pool before {self.competitive_pool}")
            trajectories = self.impute_no_result(tournament, copy.deepcopy(trajectories))

            self.get_ref(copy.deepcopy(tournament), trajectories)
            logging.info(f"Instances done {self.instances_with_results}")

            # we have reached the max number of instances. we start transfering confs
            if len(tournament.instance_set) == len(self.scenario.instance_set):
                self.transfer_confs_pools(trajectories)

            discard = self.discard_confs_competitive(self.instances_with_results, trajectories)
            self.competitive_pool[:] = [x for x in self.competitive_pool if x.id not in discard ]

            pool_combined = self.competitive_pool + self.finisher_pool
            conf_ids = [c.id for c in pool_combined]
            mean_hv_current_pool = self.mean_hypervolume(conf_ids, self.instances_with_results, trajectories)

            if self.use_model:
                # get hv for confs that we discard so we can train the gga++ surrogate also on these
                _ = self.mean_hypervolume(discard, self.instances_with_results, trajectories)

                # only use instances that have results for gga++
                result_tournament_c = copy.deepcopy(tournament)
                result_tournament_c.instance_set = self.instances_with_results

                # to safe sometime we do not update gga++ always
                if self.t < 15 or self.last_size != len(self.instances_with_results) or (self.t % 7 == 0 and self.last_size == len(self.instances_with_results) ) :
                    self.gga_sur.update(result_tournament_c, result_tournament_c.best_finisher, self.hv_store, terminations, samples=300)

                num_new_confs = self.competitive_pool_size - len(self.competitive_pool)
                num_model_conf = int(num_new_confs * (1 - self.scenario.gga_rand_ration))

                # we create more confs than we need
                new_confs = [self.create_new_conf(mean_hv_current_pool) for _ in range(num_new_confs * 3)]

                # we let gga++ rank the confs
                pred = self.gga_sur.predict(new_confs)[0]
                sugg_sorted = np.argsort(-pred)
                best_idx = sugg_sorted[:num_model_conf]
                new_confs = list(np.array(new_confs)[best_idx])[:num_model_conf]

                random_confs = [random_point(self.scenario, uuid.uuid4()) for _ in range(num_new_confs - len(new_confs))]
                new_confs = new_confs + random_confs
            else:
                new_confs = [self.create_new_conf(mean_hv_current_pool) for _ in
                             range(self.competitive_pool_size - len(self.competitive_pool))]

            self.last_size = len(self.instances_with_results)
            self.competitive_pool = self.competitive_pool + new_confs
            logging.info(f"pool after {self.competitive_pool}")

            return copy.deepcopy(self.competitive_pool)
        else:
            return copy.deepcopy(self.competitive_pool)














