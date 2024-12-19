import sys
import os
sys.path.append(os.getcwd())

import copy

from pymoo.indicators.hv import HV
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json
import os
import math
import itertools
from itertools import permutations
import gc
import time

from scenario import Scenario

from generators.default_point_generator import check_no_goods, check_conditionals

from log_setup import TournamentEncoder
from tournament_performance import compute_pareto_front
from agga.generators.mip_help import calc_area_for_front, conf_t_active, dom_points, pre_comp_area
import argparse



def down_select_points(points, ref, num_points):
    # incase a conf generated more points that should be used we select only a subset of points that
    # maximise the hypervolume

    # these points are normalised between 0 and T
    points = sorted(points, key=lambda x: x[0])
    n = len(points) + 1

    w = calc_area_for_front(points, ref)
    m = gp.Model("hv selection")

    x_ij = m.addVars(
        [
            (i, j)
            for i in range(1, n)
            for j in range(1, i + 1)
        ],
        vtype=GRB.BINARY,
        name="x_ij",
    )

    for i in range(2, n):
        for j in range(1, i):
            m.addConstr(gp.quicksum(x_ij[r, r] for r in range(j, i + 1)) >= x_ij[i, j])

    m.addConstr(gp.quicksum(x_ij[r, r] for r in range(1, n)) == num_points)

    m.setObjective(gp.quicksum(w[j][i] * x_ij[i, j] for i in range(1, n) for j in range(1, i + 1)), GRB.MAXIMIZE)
    m.optimize()

    selection = []
    for (c_s, ts), c in zip(x_ij.items(), range(len(x_ij))):
        if c_s[0] == c_s[1] and int(round(ts.x)) == 1:
            selection.append(points[c_s[0] - 1])

    return selection

def construct_portfolio(run_history, ref_for_norm, full_confs, scenario_path, instances, conf_ids, n_to_select, num_points_per_front, T, gurobi_time_limit,max_select,extend, info_dic):

    ref = T + 1

    # recalc the ref points for only the confs and instances we consider
    points_per_instance_and_conf = {}
    points_per_instance = {}
    quality_ref_upper = {}
    quality_ref_lower = {}
    before_norm = {}

    for conf in conf_ids:
        for i in instances:
            if conf in run_history and i in run_history[conf] and len(run_history[conf][i]) != 0:
                trajectory = run_history[conf][i]

                if i not in before_norm:
                    before_norm[i] = {}
                before_norm[i][conf] = run_history[conf][i]

                if i not in quality_ref_upper and len(trajectory) != 0:
                    quality_ref_upper[i] = trajectory[0][1]
                elif len(trajectory) != 0:
                    if trajectory[0][1] > quality_ref_upper[i]:
                        quality_ref_upper[i] = trajectory[0][1]

                if i not in quality_ref_lower and len(trajectory) != 0:
                    quality_ref_lower[i] = trajectory[-1][1] -0.1
                elif len(trajectory) != 0:
                    if trajectory[-1][1] < quality_ref_lower[i]:
                        quality_ref_lower[i] = trajectory[-1][1] - 0.1

    info_dic["before_norm"] = before_norm
    info_dic["ref_upper"] = quality_ref_upper
    info_dic["ref_lower"] = quality_ref_lower

    # Preprocess by normalizing and restricting the number of points in a trajectory.
    for instance in instances:
        ref_upper = quality_ref_upper[instance]
        ref_lower = quality_ref_lower[instance]

        # norm the quality between 0 and 60
        normalised_trajectories_for_i = {
            conf: [
                [0.001 if x[0] == 0 else x[0], ((x[1] - ref_lower) / (ref_upper - ref_lower)) * T]
                for x in run_history[conf][instance]
            ]
            for conf in conf_ids
        }

        # incase a conf has to many point on the pareto front for an instance
        # we select the most important ones
        for conf, ta in normalised_trajectories_for_i.items():
            if len(ta)  > num_points_per_front:
                normalised_trajectories_for_i[conf] = down_select_points(ta, [[ref,ref]], num_points_per_front)

        # for each instance get the final points from which we can select from
        points_per_instance[instance] = sorted([point for value in normalised_trajectories_for_i.values() for point in value])

        # also store for each instance and conf the final points
        points_per_instance_and_conf[instance] = {conf: (np.array(traj).tolist() if len(traj) != 0 else traj)
                          for conf, traj in normalised_trajectories_for_i.items()}

    # precompute the area as descibed in the paper
    area = {}
    for instance, count in zip(instances, range(1, len(instances) + 1)):
        area[count] = pre_comp_area(points_per_instance_and_conf[instance], T +1)

    # setup variables and get parameters
    model = gp.Model("hv selection")
    m = len(instances) + 1
    c_confs = len(conf_ids) + 1
    n_select = n_to_select - 1

    v_cct = model.addVars(
        [
            (c, cc, t)
            for c in range(1, c_confs)
            for cc in range(1, c_confs)
            for t in range(1, T + 1)
        ],
        vtype=GRB.BINARY,
        name="v_cct",
    )

    a_ct = model.addVars(
        [
            (c, t)
            for c in range(1, c_confs)
            for t in range(T + 1 )
        ],
        vtype=GRB.BINARY,
        name="a_ct",
    )

    # for every timestep a conf is selected
    for t in range(1, T + 1):
        model.addConstr(gp.quicksum(a_ct[c, t] for c in range(1, c_confs)) == 1)

    # pic at most k confs
    if max_select:
        model.addConstr(gp.quicksum(
            v_cct[c, cc, t] for c in range(1, c_confs) for cc in range(1, c_confs) for t in range(1, T + 1)) <= n_select)
    else:
        model.addConstr(gp.quicksum(
            v_cct[c, cc, t] for c in range(1, c_confs) for cc in range(1, c_confs) for t in range(1, T + 1)) == n_select)

    # can not switch back once we have used a conf
    for c in range(1, c_confs):
        model.addConstr(
            gp.quicksum(v_cct[c, cc, t] for cc in range(1, c_confs) for t in range(1, T + 1)) + a_ct[c, T - 1] <= 1)

    # can not switch back once we have used a conf
    for c in range(1, c_confs):
        model.addConstr(gp.quicksum(v_cct[cc, c, t] for cc in range(1, c_confs) for t in range(1, T + 1)) + a_ct[c, 1] <= 1)

    # we can only do one switch per timestep
    for t in range(1, T + 1):
        model.addConstr(gp.quicksum(v_cct[cc, c, t] for cc in range(1, c_confs) for c in range(1, c_confs)) <= 1)

    # we can not switch from conf 1 to conf 1
    for t in range(1, T + 1):
        for c in range(1, c_confs):
            model.addConstr(v_cct[c, c, t] == 0)

    # either switch or stay with conf
    for t in range(1, T ):
        for c in range(1, c_confs):
            model.addConstr(
                a_ct[c, t] - gp.quicksum(v_cct[c, cc, t] for cc in range(1, c_confs) if c != cc) + gp.quicksum(
                    v_cct[cc, c, t] for cc in range(1, c_confs) if c != cc) == a_ct[c, t + 1], name=f"l{t}{c}")

    # we can not switch on the last step
    for c in range(1, c_confs):
        for cc in range(1, c_confs):
            model.addConstr(v_cct[c, cc, T] == 0)

    # we can not switch on first timestep
    for c in range(1, c_confs):
        for cc in range(1, c_confs):
            model.addConstr(v_cct[c, cc, 1] == 0)

    model.setObjective(gp.quicksum(a_ct[c, t] * area[k][conf_id][t] for t in range(1, T) for k in range(1, m) for c, conf_id in zip(range(1, c_confs), conf_ids)), GRB.MAXIMIZE)

    model.Params.TimeLimit = gurobi_time_limit
    model.Params.SoftMemLimit = 16 * 2
    model.setParam('Threads', 4)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == 2:
        solution_value = model.ObjVal
        info_dic["opt_value"] = solution_value

    # map model output back to confs
    portfolio_assignment = {}
    for (c_s, ts), c_a in a_ct.items():
        if int(round(c_a.x)) == 1:
            portfolio_assignment[ts] = conf_ids[c_s - 1]

    # kind of a dummy scenario but we need it to get the condionals
    scenario = Scenario(scenario_path)

    # get the full confs and sage
    conf_per_step = {}
    for time_t, best_conf_timestep in portfolio_assignment.items():
        if best_conf_timestep in full_confs.keys():

            clean_conf = copy.copy(full_confs[best_conf_timestep])
            # Check conditionals and turn off parameters if violated
            cond_vio = check_conditionals(scenario, clean_conf)
            for cv in cond_vio:
                clean_conf.pop(cv, None)

            conf_per_step[time_t] = {"id": best_conf_timestep, "conf": clean_conf}
        else:
            raise Exception(f"Something went wrong with the portfolio construction {best_conf_timestep}")

    else:
        with open(f"{agga_dir}/portfolio_nc_{n_to_select}_{extend}.json", 'w') as f:
            json.dump(conf_per_step, f, indent=4, cls=TournamentEncoder)
            f.write(os.linesep)

    with open(f"{agga_dir}/portfolio_info_{n_to_select}_{extend}.json", "w") as f:
        json.dump(info_dic, f)


def parse_args():
    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("Args for portfolio")

    hp.add_argument('--agga_dir', default="", type=str)
    hp.add_argument('--scenario_path', default="", type=str)
    hp.add_argument('--T', default=60, type=int)
    hp.add_argument('--confs_portfolio', default=3, type=int)
    hp.add_argument('--confs_per_front', default=5, type=int)
    hp.add_argument('--restrict_confs', type=lambda x: (str(x).lower() == 'true'), default=False)
    hp.add_argument('--conf_restriction', default=40, type=int)
    hp.add_argument('--gurobi_time_limit', default=60*60*2, type=int)
    hp.add_argument('--extend', default="", type=str)
    hp.add_argument('--max_select', type=lambda x: (str(x).lower() == 'true'), default=False)

    return vars(parser.parse_args())


if __name__ == "__main__":
    portfolio_args = parse_args()

    agga_dir = portfolio_args["agga_dir"]
    scenario_path = portfolio_args["scenario_path"]
    T = portfolio_args["T"]
    confs_portfolio = portfolio_args["confs_portfolio"]
    confs_per_front = portfolio_args["confs_per_front"]
    restrict_confs = portfolio_args["restrict_confs"]
    conf_restriction = portfolio_args["conf_restriction"]
    gurobi_time_limit = portfolio_args["gurobi_time_limit"]
    max_select = portfolio_args["max_select"]
    extend = portfolio_args["extend"]

    print(agga_dir)
    with open(f'{agga_dir}/run_history_trajectory.json') as f:
        run_history = json.load(f)
        print(len(run_history))

    with open(f'{agga_dir}/tournament_history.json') as f:
        confs = json.load(f)

    # Iterate through configurations and their instances and get the pareto front of each
    # trajectory and also the number of solved instances and ref points
    refs_upper = {}
    refs_lower = {}
    instance_finish_count = {}
    info_dic = {}

    for conf, instance_run in run_history.items():
        success = 0
        for i, trajectory in instance_run.items():
            run_history[conf][i] = compute_pareto_front(np.array(trajectory), return_mask=False).tolist()

            if i not in refs_upper and len(trajectory) != 0:
                refs_upper[i] = [T, trajectory[0][1]]
            elif len(trajectory) != 0:
                if trajectory[0][1] > refs_upper[i][1]:
                    refs_upper[i][1] = trajectory[0][1]

            if i not in refs_lower and len(trajectory) != 0:
                refs_lower[i] = [0, trajectory[-1][1]]
            elif len(trajectory) != 0:
                if trajectory[-1][1] < refs_lower[i][1]:
                    refs_lower[i][1] = trajectory[-1][1]

            if len(trajectory) != 0:
                success += 1
        instance_finish_count[conf] = success

    # get all confs that we have ever ran
    # we need this to map the mip results to actual confs
    full_conf_store = {}
    for _, ks in confs.items():
        for c in ks["best_finisher"]:
            if c["id"] not in full_conf_store.keys():
                full_conf_store[c["id"]] = c["conf"]

    # we only continue with the confs that have seen the most instances
    confs_finished_most_i = [k for k, v in instance_finish_count.items() if float(v) >= max(instance_finish_count.values())]

    # We provide the possibility to restrict the number of configurations
    # the model should consider for constructing the portfolio by only including the most promising configurations
    # based on the hypervolume.
    # Compute the mean HV for all configurations that have seen the most instances.
    if restrict_confs:
        mean_hv_confs = {}
        for conf in confs_finished_most_i:
            hv_l = []
            for instance in refs_upper.keys():

                ref_point_upper = np.array(refs_upper[instance]) * 1.1
                ref_point_lower = np.array(refs_lower[instance])

                if conf in run_history and instance in run_history[conf]:
                    conf_instance_result = run_history[conf][instance]
                else:
                    run_history[conf][instance] = []

                pareto_front_of_conf = compute_pareto_front(np.array(run_history[conf][instance]), return_mask=False)

                if len(pareto_front_of_conf) == 0:
                    pareto_front_of_conf = ref_point_upper
                    hv = 0
                else:
                    ind_hv = HV(ref_point=[1, 1])
                    pareto_front_of_conf_norm = (pareto_front_of_conf - ref_point_lower) / (ref_point_upper - ref_point_lower)
                    hv = ind_hv(pareto_front_of_conf_norm)
                hv_l.append(hv)

            mean_hv_confs[conf] = sum(hv_l) / len(hv_l)

        if len(confs_finished_most_i) > conf_restriction:
            confs_finished_most_i = sorted(mean_hv_confs, key=lambda k: mean_hv_confs[k], reverse=True)[:conf_restriction]
            info_dic["hv_best_confs"] =  {conf: mean_hv_confs[conf] for conf in sorted(mean_hv_confs, key=mean_hv_confs.get, reverse=True)[:conf_restriction]}

    info_dic["finished_most_i_best_hv"] = confs_finished_most_i

    construct_portfolio(run_history, refs_upper, full_conf_store, scenario_path, list(refs_upper.keys()), confs_finished_most_i, confs_portfolio, confs_per_front, T, gurobi_time_limit,max_select,extend,  info_dic)

